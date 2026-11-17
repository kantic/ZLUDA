use super::module;
use cuda_types::cuda::*;
use hip_runtime_sys::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::{cell::RefCell, ffi::c_void, ptr, sync::Mutex};
use zluda_common::{FromCuda, ZludaObject};

// Maximum depth of the per-thread context stack (cuCtxPushCurrent nesting depth)
const MAX_CONTEXT_STACK_DEPTH: usize = 64;

// Per-thread context stack.
// This type is deliberately `Copy` with a fixed capacity: a thread-local without drop glue never
// registers a TLS destructor, so it stays accessible until the thread is gone. This matters,
// because glibc's `exit()` runs TLS destructors of the main thread *before* atexit handlers and
// C++ static destructors, and those (e.g. MXNet's engine) release their CUDA resources.
#[derive(Clone, Copy)]
pub(crate) struct ContextStack {
    len: usize,
    entries: [(CUcontext, hipDevice_t); MAX_CONTEXT_STACK_DEPTH],
}

impl ContextStack {
    const fn new() -> Self {
        Self {
            len: 0,
            entries: [(CUcontext(ptr::null_mut()), 0); MAX_CONTEXT_STACK_DEPTH],
        }
    }

    fn last(&self) -> Option<&(CUcontext, hipDevice_t)> {
        self.entries[..self.len].last()
    }

    fn pop(&mut self) -> Option<(CUcontext, hipDevice_t)> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(self.entries[self.len])
    }

    fn push(&mut self, entry: (CUcontext, hipDevice_t)) -> Result<(), CUerror> {
        if self.len == MAX_CONTEXT_STACK_DEPTH {
            return Err(CUerror::OUT_OF_MEMORY);
        }
        self.entries[self.len] = entry;
        self.len += 1;
        Ok(())
    }

    // Replaces the top of the stack, or pushes onto an empty stack
    fn replace_last(&mut self, entry: (CUcontext, hipDevice_t)) {
        if self.len == 0 {
            self.entries[0] = entry;
            self.len = 1;
        } else {
            self.entries[self.len - 1] = entry;
        }
    }
}

const _: () = assert!(!std::mem::needs_drop::<RefCell<ContextStack>>());

thread_local! {
    pub(crate) static STACK: RefCell<ContextStack> = const { RefCell::new(ContextStack::new()) };
}

pub(crate) struct Context {
    pub(crate) device: hipDevice_t,
    pub(crate) alloc_stream: hipStream_t,
    pub(crate) state: Mutex<ContextState>,
}

pub(crate) struct ContextState {
    pub(crate) ref_count: u32,
    pub(crate) flags: u32,
    pub(crate) modules: FxHashSet<CUmodule>,
    pub(crate) storage: FxHashMap<usize, StorageData>,
}

unsafe impl Send for ContextState {}

pub(crate) struct StorageData {
    pub(crate) value: usize,
    pub(crate) reset_cb: Option<extern "system" fn(CUcontext, *mut c_void, *mut c_void)>,
    pub(crate) handle: CUcontext,
}

impl ContextState {
    pub(crate) fn new() -> Self {
        ContextState {
            ref_count: 0,
            flags: 0,
            modules: FxHashSet::default(),
            storage: FxHashMap::default(),
        }
    }

    pub(crate) fn reset(&mut self) -> CUresult {
        for (key, data) in self.storage.iter_mut() {
            if let Some(_cb) = data.reset_cb {
                // TODO: check that these callbacks do not call into the CUDA driver
                // since this could result in a recursive mutex lock.
                _cb(data.handle, *key as *mut c_void, data.value as *mut c_void);
            }
        }
        self.ref_count = 0;
        self.flags = 0;
        // drop all modules and return first error if any
        let result = self.modules.drain().fold(Ok(()), |res: CUresult, hmod| {
            match (res, zluda_common::drop_checked::<module::Module>(hmod)) {
                (Err(e), _) => Err(e),
                (_, Err(e)) => Err(e),
                _ => Ok(()),
            }
        });
        self.storage.clear();
        result
    }
}

impl Context {
    pub(crate) fn new(device: hipDevice_t) -> Result<Self, CUerror> {
        let mut alloc_stream = unsafe { std::mem::zeroed() };
        unsafe { hipStreamCreateWithFlags(&mut alloc_stream, hipStreamNonBlocking) }?;
        Ok(Self {
            device,
            alloc_stream,
            state: Mutex::new(ContextState::new()),
        })
    }

    pub(crate) fn with_state(&self, fn_: impl FnOnce(&ContextState) -> CUresult) -> CUresult {
        match self.state.lock() {
            Ok(guard) => fn_(&*guard),
            Err(_) => CUresult::ERROR_UNKNOWN,
        }
    }

    pub(crate) fn with_state_mut(
        &self,
        fn_: impl FnOnce(&mut ContextState) -> CUresult,
    ) -> CUresult {
        match self.state.lock() {
            Ok(mut guard) => fn_(&mut *guard),
            Err(_) => CUresult::ERROR_UNKNOWN,
        }
    }
}

impl ZludaObject for Context {
    const COOKIE: usize = 0x5f867c6d9cb73315;

    type Error = CUerror;
    type CudaHandle = CUcontext;

    fn drop_checked(&mut self) -> CUresult {
        Ok(())
    }
}

// This type is used in lieu of Option<&Context> because of trait coherence rules.
// We can't implement FromCuda for Option<&T> in this crate because Option<T> is defined in the standard library.
pub(crate) enum MaybeContextRef<'a> {
    None,
    Some(&'a Context),
}

impl<'a>
    zluda_common::FromCuda<
        'a,
        <Context as ZludaObject>::CudaHandle,
        <Context as ZludaObject>::Error,
    > for MaybeContextRef<'a>
{
    fn from_cuda(
        handle: &'a <Context as ZludaObject>::CudaHandle,
    ) -> Result<MaybeContextRef<'a>, <Context as ZludaObject>::Error> {
        match zluda_common::as_ref(handle) {
            Some(ctx) => Ok(MaybeContextRef::Some(ctx.as_result()?)),
            None => Ok(MaybeContextRef::None),
        }
    }
}

// Accesses the thread-local context stack. `STACK` has no destructor, so this should never
// fail, but never panic: a panic can't unwind across `extern "C"` and would abort the process.
fn with_stack<T>(f: impl FnOnce(&RefCell<ContextStack>) -> T) -> Result<T, CUerror> {
    STACK.try_with(f).map_err(|_| CUerror::DEINITIALIZED)
}

pub(crate) fn get_current_device() -> Result<hipDevice_t, CUerror> {
    with_stack(|stack| {
        stack
            .try_borrow()
            .map_err(|_| CUerror::UNKNOWN)
            .and_then(|s| {
                s.last()
                    .ok_or(CUerror::INVALID_CONTEXT)
                    .map(|(_, dev)| *dev)
            })
    })?
}

pub(crate) fn get_current_context() -> Result<CUcontext, CUerror> {
    if let Some(ctx) = with_stack(|stack| {
        stack
            .try_borrow()
            .ok()
            .and_then(|stack| stack.last().map(|(ctx, _)| *ctx))
    })? {
        return Ok(ctx);
    }
    Err(CUerror::INVALID_CONTEXT)
}

pub(crate) unsafe fn get_limit(pvalue: *mut usize, limit: hipLimit_t) -> hipError_t {
    unsafe { hipDeviceGetLimit(pvalue, limit) }
}

pub(crate) fn set_limit(limit: hipLimit_t, value: usize) -> hipError_t {
    unsafe { hipDeviceSetLimit(limit, value) }
}

pub(crate) fn set_cache_config(config: CUfunc_cache) -> CUresult {
    get_current_context()?;
    match config {
        CUfunc_cache::CU_FUNC_CACHE_PREFER_NONE
        | CUfunc_cache::CU_FUNC_CACHE_PREFER_SHARED
        | CUfunc_cache::CU_FUNC_CACHE_PREFER_L1
        | CUfunc_cache::CU_FUNC_CACHE_PREFER_EQUAL => Ok(()),
        _ => Err(CUerror::INVALID_VALUE),
    }
}

pub(crate) fn synchronize() -> hipError_t {
    unsafe { hipDeviceSynchronize() }
}

pub(crate) fn synchronize_v2(ctx: &Context) -> hipError_t {
    let mut current_device = 0;
    unsafe { hipGetDevice(&mut current_device)? };
    let _dev_drop = if current_device != ctx.device {
        unsafe { hipSetDevice(ctx.device) }?;
        Some(SetDeviceOnDrop(current_device))
    } else {
        None
    };
    unsafe { hipDeviceSynchronize() }
}

struct SetDeviceOnDrop(i32);

impl Drop for SetDeviceOnDrop {
    fn drop(&mut self) {
        unsafe { hipSetDevice(self.0) }.ok();
    }
}

// cuCtxSetCurrent replaces the top of the context stack, cuCtxPushCurrent and cuCtxCreate push
// onto it. Both are equivalent to a pop when called with a null context
#[derive(Clone, Copy, PartialEq, Eq)]
enum StackOp {
    Push,
    ReplaceTop,
}

pub(crate) fn set_current(raw_ctx: CUcontext) -> CUresult {
    set_current_impl(raw_ctx, StackOp::ReplaceTop)
}

fn set_current_impl(raw_ctx: CUcontext, op: StackOp) -> CUresult {
    let new_device = if raw_ctx.0 == ptr::null_mut() {
        with_stack(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some((_, old_device)) = stack.pop() {
                if let Some((_, new_device)) = stack.last() {
                    if old_device != *new_device {
                        return Some(*new_device);
                    }
                }
            }
            None
        })?
    } else {
        let ctx: &Context = FromCuda::<_, CUerror>::from_cuda(&raw_ctx)?;
        let device = ctx.device;
        with_stack(move |stack| {
            let mut stack = stack.try_borrow_mut().map_err(|_| CUerror::UNKNOWN)?;
            let last_device = stack.last().map(|(_, dev)| *dev);
            match op {
                StackOp::Push => stack.push((raw_ctx, device))?,
                StackOp::ReplaceTop => stack.replace_last((raw_ctx, device)),
            }
            Ok::<_, CUerror>(match last_device {
                None => Some(device),
                Some(last_device) if last_device != device => Some(device),
                _ => None,
            })
        })??
    };
    if let Some(dev) = new_device {
        unsafe { hipSetDevice(dev)? };
    }
    Ok(())
}

pub(crate) fn get_current(pctx: &mut CUcontext) -> CUresult {
    match get_current_context() {
        Ok(ctx) => *pctx = ctx,
        Err(_) => *pctx = CUcontext(ptr::null_mut()),
    }
    CUresult::SUCCESS
}

pub(crate) fn get_device(dev: &mut hipDevice_t) -> CUresult {
    get_device_v2(dev, MaybeContextRef::None)
}

pub(crate) fn get_device_v2(dev: &mut hipDevice_t, ctx: MaybeContextRef) -> CUresult {
    let device = match ctx {
        MaybeContextRef::Some(ctx) => ctx.device,
        MaybeContextRef::None => {
            let cu_ctx: CUcontext = get_current_context()?;
            let ctx: &Context = FromCuda::<_, CUerror>::from_cuda(&cu_ctx)?;
            ctx.device
        }
    };
    *dev = device;
    Ok(())
}

pub(crate) unsafe fn push_current(ctx: CUcontext) -> CUresult {
    if ctx == CUcontext(ptr::null_mut()) {
        return CUresult::ERROR_INVALID_VALUE;
    }
    set_current_impl(ctx, StackOp::Push)
}

pub(crate) unsafe fn push_current_v2(ctx: CUcontext) -> CUresult {
    push_current(ctx)
}

pub(crate) unsafe fn pop_current(result: Option<&mut CUcontext>) -> CUresult {
    let old_ctx_and_new_device = with_stack(|stack| {
        let mut stack = stack.borrow_mut();
        stack
            .pop()
            .map(|(ctx, _)| (ctx, stack.last().map(|(_, dev)| *dev)))
    })?;
    let ctx = match old_ctx_and_new_device {
        Some((old_ctx, new_device)) => {
            if let Some(new_device) = new_device {
                hipSetDevice(new_device)?;
            }
            old_ctx
        }
        None => return CUresult::ERROR_INVALID_CONTEXT,
    };
    if let Some(out) = result {
        *out = ctx;
    }
    Ok(())
}

pub(crate) unsafe fn create_v2(
    ctx: &mut CUcontext,
    _flags: ::core::ffi::c_uint,
    dev: cuda_types::cuda::CUdevice,
) -> CUresult {
    let handle = Context::wrap(Context::new(dev)?);
    // TODO: optimize
    set_current_impl(handle, StackOp::Push)?;
    *ctx = handle;
    Ok(())
}

pub(crate) unsafe fn create_v4(
    ctx: &mut CUcontext,
    _ctx_create_params: Option<&mut cuda_types::cuda::CUctxCreateParams>,
    flags: ::core::ffi::c_uint,
    dev: cuda_types::cuda::CUdevice,
) -> CUresult {
    create_v2(ctx, flags, dev)
}

pub(crate) unsafe fn destroy_v2(ctx: CUcontext) -> CUresult {
    zluda_common::drop_checked::<Context>(ctx)
}

pub(crate) unsafe fn pop_current_v2(ctx: Option<&mut CUcontext>) -> CUresult {
    pop_current(ctx)
}

pub(crate) unsafe fn get_stream_priority_range(
    least_priority: *mut ::core::ffi::c_int,
    greatest_priority: *mut ::core::ffi::c_int,
) -> hipError_t {
    hipDeviceGetStreamPriorityRange(least_priority, greatest_priority)
}

pub(crate) unsafe fn set_flags(flags: ::core::ffi::c_uint) -> CUresult {
    let cu_ctx = get_current_context()?;
    let ctx: &Context = FromCuda::<_, CUerror>::from_cuda(&cu_ctx)?;
    ctx.with_state_mut(|state| {
        state.flags = flags;
        Ok(())
    })
}

pub(crate) unsafe fn get_api_version(
    _ctx: CUcontext,
    version: &mut ::core::ffi::c_uint,
) -> CUresult {
    // That's what original CUDA driver does
    *version = 3020;
    Ok(())
}

pub(crate) unsafe fn get_shared_memory_config(config: &mut CUsharedconfig) -> CUresult {
    *config = CUsharedconfig::CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
    Ok(())
}

pub(crate) unsafe fn set_shared_memory_config(_: CUsharedconfig) -> CUresult {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::CudaApi;
    use cuda_macros::test_cuda;
    use std::mem;

    #[test_cuda]
    fn empty_pop_fails(api: impl CudaApi) {
        api.cuInit(0);
        assert_eq!(
            api.cuCtxPopCurrent_v2_unchecked(&mut unsafe { mem::zeroed() }),
            CUresult::ERROR_INVALID_CONTEXT
        );
    }

    #[test_cuda]
    fn pop_into_null_succeeds(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut unsafe { mem::zeroed() }, 0, 0);
        api.cuCtxPopCurrent_v2(ptr::null_mut());
    }

    #[test_cuda]
    fn set_current_replaces_top_of_stack(api: impl CudaApi) {
        api.cuInit(0);
        let mut ctx1 = unsafe { mem::zeroed() };
        let mut ctx2 = unsafe { mem::zeroed() };
        api.cuCtxCreate_v2(&mut ctx1, 0, 0);
        api.cuCtxCreate_v2(&mut ctx2, 0, 0);
        // stack: [ctx1, ctx2] -> [ctx1, ctx1]
        api.cuCtxSetCurrent(ctx1);
        let mut popped = unsafe { mem::zeroed() };
        api.cuCtxPopCurrent_v2(&mut popped);
        assert_eq!(popped, ctx1);
        let mut current = unsafe { mem::zeroed() };
        api.cuCtxGetCurrent(&mut current);
        assert_eq!(current, ctx1);
        api.cuCtxPopCurrent_v2(&mut popped);
        assert_eq!(popped, ctx1);
        assert_eq!(
            api.cuCtxPopCurrent_v2_unchecked(&mut popped),
            CUresult::ERROR_INVALID_CONTEXT
        );
    }

    #[test_cuda]
    fn set_current_does_not_grow_stack(api: impl CudaApi) {
        api.cuInit(0);
        let mut ctx = unsafe { mem::zeroed() };
        api.cuCtxCreate_v2(&mut ctx, 0, 0);
        for _ in 0..1000 {
            api.cuCtxSetCurrent(ctx);
        }
        let mut popped = unsafe { mem::zeroed() };
        api.cuCtxPopCurrent_v2(&mut popped);
        assert_eq!(
            api.cuCtxPopCurrent_v2_unchecked(&mut popped),
            CUresult::ERROR_INVALID_CONTEXT
        );
    }

    #[test_cuda]
    fn set_cache_config_accepts_known_values(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut unsafe { mem::zeroed() }, 0, 0);
        for config in [
            CUfunc_cache::CU_FUNC_CACHE_PREFER_NONE,
            CUfunc_cache::CU_FUNC_CACHE_PREFER_SHARED,
            CUfunc_cache::CU_FUNC_CACHE_PREFER_L1,
            CUfunc_cache::CU_FUNC_CACHE_PREFER_EQUAL,
        ] {
            api.cuCtxSetCacheConfig(config);
        }
    }

    #[test_cuda]
    fn set_cache_config_rejects_unknown_value(api: impl CudaApi) {
        api.cuInit(0);
        api.cuCtxCreate_v2(&mut unsafe { mem::zeroed() }, 0, 0);
        assert_eq!(
            api.cuCtxSetCacheConfig_unchecked(CUfunc_cache(4)),
            CUresult::ERROR_INVALID_VALUE
        );
    }

    #[test_cuda]
    fn set_cache_config_requires_current_context(api: impl CudaApi) {
        api.cuInit(0);
        assert_eq!(
            api.cuCtxSetCacheConfig_unchecked(CUfunc_cache::CU_FUNC_CACHE_PREFER_NONE),
            CUresult::ERROR_INVALID_CONTEXT
        );
    }
}
