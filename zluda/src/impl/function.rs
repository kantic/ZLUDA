use crate::r#impl::driver;
use cuda_types::cuda::{CUerror, CUfunction, CUfunction_attribute, CUkernel};
use dark_api::FunctionArgInfo;
use hip_runtime_sys::*;
use std::{mem, ptr, sync::Mutex};

pub(crate) struct Function {
    pub(crate) base: hipFunction_t,
    pub(crate) sm_version: u32,
    pub(crate) explicit_args_size_align: Option<Vec<FunctionArgInfo>>,
    /// Launch configuration set through the legacy (pre-CUDA 4.0) execution control API
    pub(crate) legacy: Mutex<LegacyLaunch>,
}

/// Kernel parameters, block shape and dynamic shared memory size set with cuParamSet*,
/// cuFuncSetBlockShape and cuFuncSetSharedSize for cuLaunch/cuLaunchGrid. Wolfram's CUDALink
/// launches kernels this way.
pub(crate) struct LegacyLaunch {
    params: Vec<u8>,
    params_size: Option<usize>,
    block: [u32; 3],
    shared_mem_bytes: u32,
}

impl Default for LegacyLaunch {
    fn default() -> Self {
        Self {
            params: Vec::new(),
            params_size: None,
            block: [1, 1, 1],
            shared_mem_bytes: 0,
        }
    }
}

impl<'a, E: zluda_common::CudaErrorType> zluda_common::FromCuda<'a, CUfunction, E>
    for &'a Function
{
    fn from_cuda(cu_func: &'a CUfunction) -> Result<Self, E> {
        Ok(unsafe { mem::transmute(*cu_func) })
    }
}

impl<'a, E: zluda_common::CudaErrorType> zluda_common::FromCuda<'a, CUkernel, E> for &'a Function {
    fn from_cuda(cu_func: &'a CUkernel) -> Result<Self, E> {
        Ok(unsafe { mem::transmute(*cu_func) })
    }
}

impl<'a, E: zluda_common::CudaErrorType> zluda_common::FromCuda<'a, *mut CUfunction, E>
    for &'a mut &'a Function
{
    fn from_cuda(cu_func: &'a *mut CUfunction) -> Result<Self, E> {
        let cu_func = unsafe { cu_func.as_mut() }.ok_or_else(|| E::INVALID_VALUE)?;
        Ok(unsafe { mem::transmute(cu_func) })
    }
}

impl<'a, E: zluda_common::CudaErrorType> zluda_common::FromCuda<'a, *mut CUkernel, E>
    for &'a mut &'a Function
{
    fn from_cuda(cu_func: &'a *mut CUkernel) -> Result<Self, E> {
        let cu_func = unsafe { cu_func.as_mut() }.ok_or_else(|| E::INVALID_VALUE)?;
        Ok(unsafe { mem::transmute(cu_func) })
    }
}

pub(crate) fn get_attribute(
    pi: &mut i32,
    cu_attrib: CUfunction_attribute,
    func: &Function,
) -> Result<(), CUerror> {
    match cu_attrib {
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION => {
            *pi = func.sm_version as i32;
            return Ok(());
        }
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION => {
            let (major_cap, minor_cap) = driver::global_state()?.compute_capability;
            *pi = major_cap * 10 + minor_cap;
            return Ok(());
        }
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE => {
            *pi = 0;
            return Ok(());
        }
        _ => {}
    }
    unsafe { hipFuncGetAttribute(pi, mem::transmute(cu_attrib), func.base) }?;
    if cu_attrib == CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS {
        *pi = (*pi).max(1);
    }
    Ok(())
}

const CU_LAUNCH_PARAM_BUFFER_POINTER: usize = 0x01;
const CU_LAUNCH_PARAM_BUFFER_SIZE: usize = 0x02;
const CU_LAUNCH_PARAM_END: usize = 0x00;
const HIP_LAUNCH_PARAM_END: usize = 0x03;

pub(crate) unsafe fn launch_kernel(
    f: &Function,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: hipStream_t,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> hipError_t {
    if let Some(extra) = extra.cast::<[usize; 5]>().as_ref() {
        match extra {
            [CU_LAUNCH_PARAM_BUFFER_POINTER, pointer, CU_LAUNCH_PARAM_BUFFER_SIZE, size, CU_LAUNCH_PARAM_END] =>
            {
                let mut hip_extra: [_; 5] = [
                    CU_LAUNCH_PARAM_BUFFER_POINTER,
                    *pointer,
                    CU_LAUNCH_PARAM_BUFFER_SIZE,
                    *size,
                    HIP_LAUNCH_PARAM_END,
                ];
                hipModuleLaunchKernel(
                    f.base,
                    grid_dim_x,
                    grid_dim_y,
                    grid_dim_z,
                    block_dim_x,
                    block_dim_y,
                    block_dim_z,
                    shared_mem_bytes,
                    stream,
                    kernel_params,
                    hip_extra.as_mut_ptr().cast(),
                )
            }
            _ => hipError_t::ErrorInvalidValue,
        }
    } else {
        hipModuleLaunchKernel(
            f.base,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            block_dim_x,
            block_dim_y,
            block_dim_z,
            shared_mem_bytes,
            stream,
            kernel_params,
            extra,
        )
    }
}

fn with_legacy<T>(
    func: &Function,
    f: impl FnOnce(&mut LegacyLaunch) -> Result<T, CUerror>,
) -> Result<T, CUerror> {
    let mut legacy = func.legacy.lock().map_err(|_| CUerror::UNKNOWN)?;
    f(&mut legacy)
}

fn set_param_bytes(func: &Function, offset: i32, bytes: &[u8]) -> Result<(), CUerror> {
    let offset = usize::try_from(offset).map_err(|_| CUerror::INVALID_VALUE)?;
    with_legacy(func, |legacy| {
        let end = offset + bytes.len();
        if legacy.params.len() < end {
            legacy.params.resize(end, 0);
        }
        legacy.params[offset..end].copy_from_slice(bytes);
        Ok(())
    })
}

pub(crate) unsafe fn param_set_size(func: &Function, numbytes: u32) -> Result<(), CUerror> {
    with_legacy(func, |legacy| {
        legacy.params_size = Some(numbytes as usize);
        Ok(())
    })
}

pub(crate) unsafe fn param_seti(func: &Function, offset: i32, value: u32) -> Result<(), CUerror> {
    set_param_bytes(func, offset, &value.to_ne_bytes())
}

pub(crate) unsafe fn param_setf(func: &Function, offset: i32, value: f32) -> Result<(), CUerror> {
    set_param_bytes(func, offset, &value.to_ne_bytes())
}

pub(crate) unsafe fn param_setv(
    func: &Function,
    offset: i32,
    ptr: *mut ::core::ffi::c_void,
    numbytes: u32,
) -> Result<(), CUerror> {
    if ptr.is_null() && numbytes > 0 {
        return Err(CUerror::INVALID_VALUE);
    }
    let bytes = if numbytes == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(ptr.cast::<u8>(), numbytes as usize)
    };
    set_param_bytes(func, offset, bytes)
}

pub(crate) unsafe fn set_block_shape(func: &Function, x: i32, y: i32, z: i32) -> Result<(), CUerror> {
    let dim = |d: i32| u32::try_from(d).ok().filter(|d| *d > 0).ok_or(CUerror::INVALID_VALUE);
    let block = [dim(x)?, dim(y)?, dim(z)?];
    with_legacy(func, |legacy| {
        legacy.block = block;
        Ok(())
    })
}

pub(crate) unsafe fn set_shared_size(func: &Function, bytes: u32) -> Result<(), CUerror> {
    with_legacy(func, |legacy| {
        legacy.shared_mem_bytes = bytes;
        Ok(())
    })
}

pub(crate) unsafe fn launch_grid_async(
    func: &Function,
    grid_width: i32,
    grid_height: i32,
    stream: hipStream_t,
) -> Result<(), CUerror> {
    let dim = |d: i32| u32::try_from(d).ok().filter(|d| *d > 0).ok_or(CUerror::INVALID_VALUE);
    let (grid_x, grid_y) = (dim(grid_width)?, dim(grid_height)?);
    // Copy the configuration, so that the lock isn't held during the launch
    let (mut params, block, shared_mem_bytes) = with_legacy(func, |legacy| {
        let mut params = legacy.params.clone();
        if let Some(size) = legacy.params_size {
            params.resize(size, 0);
        }
        Ok((params, legacy.block, legacy.shared_mem_bytes))
    })?;
    let mut size = params.len();
    let mut extra = [
        CU_LAUNCH_PARAM_BUFFER_POINTER as *mut ::core::ffi::c_void,
        params.as_mut_ptr().cast(),
        CU_LAUNCH_PARAM_BUFFER_SIZE as *mut ::core::ffi::c_void,
        ptr::from_mut(&mut size).cast(),
        CU_LAUNCH_PARAM_END as *mut ::core::ffi::c_void,
    ];
    launch_kernel(
        func,
        grid_x,
        grid_y,
        1,
        block[0],
        block[1],
        block[2],
        shared_mem_bytes,
        stream,
        ptr::null_mut(),
        extra.as_mut_ptr(),
    )?;
    Ok(())
}

pub(crate) unsafe fn launch_grid(func: &Function, grid_width: i32, grid_height: i32) -> Result<(), CUerror> {
    launch_grid_async(func, grid_width, grid_height, hipStream_t(ptr::null_mut()))
}

pub(crate) unsafe fn launch(func: &Function) -> Result<(), CUerror> {
    launch_grid(func, 1, 1)
}

pub(crate) unsafe fn set_attribute(
    func: &Function,
    attribute: CUfunction_attribute,
    value: i32,
) -> hipError_t {
    match attribute {
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION => {
            return hipError_t::ErrorNotSupported;
        }
        _ => {}
    }
    hipFuncSetAttribute(func.base.0.cast(), hipFuncAttribute(attribute.0), value)
}

pub(crate) unsafe fn set_cache_config(
    _func: &Function,
    _config: cuda_types::cuda::CUfunc_cache,
) -> hipError_t {
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::tests::CudaApi;
    use cuda_macros::test_cuda;
    use cuda_types::cuda::{CUdevice_attribute, CUfunction_attribute};
    use std::mem;

    #[test_cuda]
    fn check_versions(api: impl CudaApi) {
        api.cuInit(0);
        let mut ctx = unsafe { mem::zeroed() };
        api.cuCtxCreate_v2(&mut ctx, 0, 0);
        let mut major = 0;
        api.cuDeviceGetAttribute(
            &mut major,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            0,
        );
        let mut minor = 0;
        api.cuDeviceGetAttribute(
            &mut minor,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            0,
        );
        let ptx = c".version 8.7
        .target sm_60
        .address_size 64

        .entry foobar() { ret; }
        ";

        let mut module = unsafe { mem::zeroed() };
        api.cuModuleLoadData(&mut module, ptx.as_ptr().cast());
        let mut func = unsafe { mem::zeroed() };
        api.cuModuleGetFunction(&mut func, module, c"foobar".as_ptr().cast());
        let mut binary_version = 0;
        api.cuFuncGetAttribute(
            &mut binary_version,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION,
            func,
        );
        let mut ptx_version = 0;
        api.cuFuncGetAttribute(
            &mut ptx_version,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION,
            func,
        );
        assert_eq!(binary_version, major * 10 + minor);
        assert_eq!(ptx_version, 60);
    }

    #[test_cuda]
    fn legacy_launch_api(api: impl CudaApi) {
        api.cuInit(0);
        let mut ctx = unsafe { mem::zeroed() };
        api.cuCtxCreate_v2(&mut ctx, 0, 0);
        // out[blockIdx.y * gridDim.x * ntid.x + blockIdx.x * ntid.x + tid.x] = value + tid.x
        let ptx = c".version 6.5
        .target sm_60
        .address_size 64

        .entry store(.param .u64 out, .param .u32 value)
        {
            .reg .u64 addr;
            .reg .u32 v, tid, ntid, ctaid, ctaidy, nctaid, idx;
            .reg .u64 off;
            ld.param.u64 addr, [out];
            ld.param.u32 v, [value];
            mov.u32 tid, %tid.x;
            mov.u32 ntid, %ntid.x;
            mov.u32 ctaid, %ctaid.x;
            mov.u32 ctaidy, %ctaid.y;
            mov.u32 nctaid, %nctaid.x;
            mad.lo.u32 idx, ctaidy, nctaid, ctaid;
            mad.lo.u32 idx, idx, ntid, tid;
            add.u32 v, v, tid;
            mul.wide.u32 off, idx, 4;
            add.u64 addr, addr, off;
            st.global.u32 [addr], v;
            ret;
        }
        ";
        let mut module = unsafe { mem::zeroed() };
        api.cuModuleLoadData(&mut module, ptx.as_ptr().cast());
        let mut func = unsafe { mem::zeroed() };
        api.cuModuleGetFunction(&mut func, module, c"store".as_ptr().cast());
        let mut out = unsafe { mem::zeroed() };
        api.cuMemAlloc_v2(&mut out, 16 * 4);
        api.cuFuncSetBlockShape(func, 4, 1, 1);
        api.cuFuncSetSharedSize(func, 0);
        let mut ptr = out.0 as u64;
        api.cuParamSetv(func, 0, std::ptr::from_mut(&mut ptr).cast(), 8);
        api.cuParamSeti(func, 8, 100);
        api.cuParamSetSize(func, 12);
        api.cuLaunchGrid(func, 2, 2);
        api.cuCtxSynchronize();
        let mut result = [0u32; 16];
        api.cuMemcpyDtoH_v2(result.as_mut_ptr().cast(), out, 16 * 4);
        let expected: Vec<u32> = (0..16).map(|i| 100 + i % 4).collect();
        assert_eq!(result.to_vec(), expected);
    }
}
