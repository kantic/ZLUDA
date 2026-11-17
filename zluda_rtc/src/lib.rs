//! NVRTC shim for ZLUDA.
//!
//! NVRTC compiles CUDA C++ to PTX or to NVIDIA machine code (CUBIN). ZLUDA can only consume
//! PTX, so this library wraps the real NVIDIA NVRTC and:
//! * rewrites real-architecture compilation targets (`sm_XX`) to virtual ones (`compute_XX`),
//!   clamped to an architecture supported by NVRTC, so that NVRTC always produces PTX
//! * returns the PTX from `nvrtcGetCUBINSize`/`nvrtcGetCUBIN`. ZLUDA's `cuModuleLoadData`
//!   accepts PTX, so applications that compile to CUBIN and load it with the driver API
//!   (e.g. MXNet and PyTorch) work unmodified
//!
//! Everything else is forwarded to the real NVRTC, which is found in this order:
//! 1. `ZLUDA_NVRTC_LIB` environment variable
//! 2. well-known CUDA installation paths

#![allow(non_snake_case, non_camel_case_types)]

use std::{
    ffi::{c_char, c_int, c_void, CStr, CString},
    mem,
    path::{Path, PathBuf},
    sync::OnceLock,
};

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct nvrtcResult(pub c_int);

impl nvrtcResult {
    pub const SUCCESS: nvrtcResult = nvrtcResult(0);
    pub const ERROR_INVALID_INPUT: nvrtcResult = nvrtcResult(3);
    pub const ERROR_INVALID_OPTION: nvrtcResult = nvrtcResult(5);
    pub const ERROR_INTERNAL_ERROR: nvrtcResult = nvrtcResult(11);
}

pub type nvrtcProgram = *mut c_void;

macro_rules! nvrtc_functions {
    ($($name:ident ( $($arg:ident : $arg_type:ty),* ) -> $ret:ty;)*) => {
        #[allow(dead_code)]
        struct Nvrtc {
            _lib: libloading::Library,
            $($name: Option<unsafe extern "C" fn($($arg_type),*) -> $ret>,)*
        }

        impl Nvrtc {
            unsafe fn new(lib: libloading::Library) -> Self {
                Self {
                    $($name: lib.get::<unsafe extern "C" fn($($arg_type),*) -> $ret>(concat!(stringify!($name), "\0").as_bytes()).ok().map(|f| *f),)*
                    _lib: lib,
                }
            }
        }
    };
}

// Forwarding is generated for all of these, except the functions with custom implementations
// below (nvrtcGetErrorString, nvrtcCompileProgram, nvrtcGetCUBINSize, nvrtcGetCUBIN), which
// call into the real library explicitly
nvrtc_functions! {
    nvrtcGetErrorString(result: nvrtcResult) -> *const c_char;
    nvrtcVersion(major: *mut c_int, minor: *mut c_int) -> nvrtcResult;
    nvrtcGetNumSupportedArchs(num_archs: *mut c_int) -> nvrtcResult;
    nvrtcGetSupportedArchs(supported_archs: *mut c_int) -> nvrtcResult;
    nvrtcCreateProgram(prog: *mut nvrtcProgram, src: *const c_char, name: *const c_char, num_headers: c_int, headers: *const *const c_char, include_names: *const *const c_char) -> nvrtcResult;
    nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;
    nvrtcCompileProgram(prog: nvrtcProgram, num_options: c_int, options: *const *const c_char) -> nvrtcResult;
    nvrtcGetPTXSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;
    nvrtcGetCUBINSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    nvrtcGetCUBIN(prog: nvrtcProgram, cubin: *mut c_char) -> nvrtcResult;
    nvrtcGetNVVMSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    nvrtcGetNVVM(prog: nvrtcProgram, nvvm: *mut c_char) -> nvrtcResult;
    nvrtcGetLTOIRSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    nvrtcGetLTOIR(prog: nvrtcProgram, ltoir: *mut c_char) -> nvrtcResult;
    nvrtcGetOptiXIRSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    nvrtcGetOptiXIR(prog: nvrtcProgram, optixir: *mut c_char) -> nvrtcResult;
    nvrtcGetProgramLogSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult;
    nvrtcAddNameExpression(prog: nvrtcProgram, name_expression: *const c_char) -> nvrtcResult;
    nvrtcGetLoweredName(prog: nvrtcProgram, name_expression: *const c_char, lowered_name: *mut *const c_char) -> nvrtcResult;
    nvrtcGetPCHHeapSize(size: *mut usize) -> nvrtcResult;
    nvrtcSetPCHHeapSize(size: usize) -> nvrtcResult;
    nvrtcGetPCHCreateStatus(prog: nvrtcProgram) -> nvrtcResult;
    nvrtcGetPCHHeapSizeRequired(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    nvrtcSetFlowCallback(prog: nvrtcProgram, callback: *mut c_void, payload: *mut c_void) -> nvrtcResult;
}

const NVRTC_ENV_VAR: &str = "ZLUDA_NVRTC_LIB";

const NVRTC_CANDIDATES: &[&str] = &[
    "/usr/local/cuda/lib64/libnvrtc.so.12",
    "/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.12",
    "/usr/lib/x86_64-linux-gnu/libnvrtc.so.12",
    "/usr/lib64/libnvrtc.so.12",
];

fn self_path() -> Option<PathBuf> {
    unsafe {
        let mut info: libc::Dl_info = mem::zeroed();
        if libc::dladdr(self_path as *const c_void, &mut info) == 0 || info.dli_fname.is_null() {
            return None;
        }
        Some(PathBuf::from(CStr::from_ptr(info.dli_fname).to_str().ok()?))
    }
}

fn is_self(path: &Path) -> bool {
    match (
        self_path().and_then(|p| p.canonicalize().ok()),
        path.canonicalize(),
    ) {
        (Some(self_path), Ok(path)) => self_path == path,
        _ => false,
    }
}

unsafe fn load_nvrtc() -> Option<Nvrtc> {
    let env_path = std::env::var_os(NVRTC_ENV_VAR).map(PathBuf::from);
    let candidates = env_path
        .into_iter()
        .chain(NVRTC_CANDIDATES.iter().map(PathBuf::from));
    for candidate in candidates {
        if !candidate.exists() || is_self(&candidate) {
            continue;
        }
        // RTLD_LOCAL: the real NVRTC must not take part in global symbol resolution,
        // otherwise its symbols could interpose ours
        if let Ok(lib) = libloading::os::unix::Library::open(
            Some(&candidate),
            libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_LOCAL,
        ) {
            return Some(Nvrtc::new(lib.into()));
        }
    }
    None
}

fn nvrtc() -> Option<&'static Nvrtc> {
    static NVRTC: OnceLock<Option<Nvrtc>> = OnceLock::new();
    NVRTC.get_or_init(|| unsafe { load_nvrtc() }).as_ref()
}

macro_rules! forward {
    ($($name:ident ( $($arg:ident : $arg_type:ty),* ) -> $ret:ty;)*) => {
        $(
            #[no_mangle]
            pub unsafe extern "C" fn $name($($arg: $arg_type),*) -> $ret {
                match nvrtc().and_then(|nvrtc| nvrtc.$name) {
                    Some(f) => f($($arg),*),
                    None => nvrtcResult::ERROR_INTERNAL_ERROR,
                }
            }
        )*
    };
}

forward! {
    nvrtcVersion(major: *mut c_int, minor: *mut c_int) -> nvrtcResult;
    nvrtcGetNumSupportedArchs(num_archs: *mut c_int) -> nvrtcResult;
    nvrtcGetSupportedArchs(supported_archs: *mut c_int) -> nvrtcResult;
    nvrtcCreateProgram(prog: *mut nvrtcProgram, src: *const c_char, name: *const c_char, num_headers: c_int, headers: *const *const c_char, include_names: *const *const c_char) -> nvrtcResult;
    nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;
    nvrtcGetPTXSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;
    nvrtcGetNVVMSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    nvrtcGetNVVM(prog: nvrtcProgram, nvvm: *mut c_char) -> nvrtcResult;
    nvrtcGetLTOIRSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    nvrtcGetLTOIR(prog: nvrtcProgram, ltoir: *mut c_char) -> nvrtcResult;
    nvrtcGetOptiXIRSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    nvrtcGetOptiXIR(prog: nvrtcProgram, optixir: *mut c_char) -> nvrtcResult;
    nvrtcGetProgramLogSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult;
    nvrtcAddNameExpression(prog: nvrtcProgram, name_expression: *const c_char) -> nvrtcResult;
    nvrtcGetLoweredName(prog: nvrtcProgram, name_expression: *const c_char, lowered_name: *mut *const c_char) -> nvrtcResult;
    nvrtcGetPCHHeapSize(size: *mut usize) -> nvrtcResult;
    nvrtcSetPCHHeapSize(size: usize) -> nvrtcResult;
    nvrtcGetPCHCreateStatus(prog: nvrtcProgram) -> nvrtcResult;
    nvrtcGetPCHHeapSizeRequired(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    nvrtcSetFlowCallback(prog: nvrtcProgram, callback: *mut c_void, payload: *mut c_void) -> nvrtcResult;
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetErrorString(result: nvrtcResult) -> *const c_char {
    if let Some(f) = nvrtc().and_then(|nvrtc| nvrtc.nvrtcGetErrorString) {
        return f(result);
    }
    match result {
        nvrtcResult::SUCCESS => c"NVRTC_SUCCESS".as_ptr(),
        nvrtcResult::ERROR_INTERNAL_ERROR => {
            c"NVRTC_ERROR_INTERNAL_ERROR (ZLUDA could not load NVIDIA NVRTC, set ZLUDA_NVRTC_LIB to the path of the NVIDIA libnvrtc.so)".as_ptr()
        }
        _ => c"NVRTC_ERROR unknown".as_ptr(),
    }
}

fn supported_archs(nvrtc: &Nvrtc) -> Vec<c_int> {
    let (Some(get_num), Some(get_archs)) =
        (nvrtc.nvrtcGetNumSupportedArchs, nvrtc.nvrtcGetSupportedArchs)
    else {
        return Vec::new();
    };
    let mut num = 0;
    if unsafe { get_num(&mut num) } != nvrtcResult::SUCCESS || num <= 0 {
        return Vec::new();
    }
    let mut archs = vec![0; num as usize];
    if unsafe { get_archs(archs.as_mut_ptr()) } != nvrtcResult::SUCCESS {
        return Vec::new();
    }
    archs.sort_unstable();
    archs
}

/// Map an architecture (e.g. "sm_86", "compute_85", "sm_90a") to a virtual architecture
/// supported by NVRTC, e.g. "compute_86" or "compute_80"
fn to_virtual_arch(arch: &str, supported: &[c_int]) -> Option<String> {
    let number = arch
        .strip_prefix("sm_")
        .or_else(|| arch.strip_prefix("compute_"))
        .or_else(|| arch.strip_prefix("lto_"))?;
    let digits = number
        .find(|c: char| !c.is_ascii_digit())
        .map_or(number, |end| &number[..end]);
    let requested = digits.parse::<c_int>().ok()?;
    let arch = if supported.is_empty() {
        requested
    } else {
        // Highest supported architecture that is not newer than the requested one
        supported
            .iter()
            .copied()
            .filter(|a| *a <= requested)
            .max()
            .unwrap_or(supported[0])
    };
    Some(format!("compute_{}", arch))
}

fn rewrite_options(options: &[&[u8]], supported: &[c_int]) -> Vec<CString> {
    const ARCH_FLAGS: &[&str] = &["--gpu-architecture", "-arch"];
    let mut result = Vec::with_capacity(options.len());
    let mut iter = options.iter().peekable();
    while let Some(option) = iter.next() {
        let option_str = String::from_utf8_lossy(option);
        let mut rewritten = None;
        for flag in ARCH_FLAGS {
            if let Some(value) = option_str
                .strip_prefix(flag)
                .and_then(|rest| rest.strip_prefix('='))
            {
                rewritten = to_virtual_arch(value, supported).map(|arch| format!("{flag}={arch}"));
                break;
            }
            if option_str == *flag {
                // Value in a separate option
                if let Some(value) = iter.peek().map(|v| String::from_utf8_lossy(v).into_owned()) {
                    if let Some(arch) = to_virtual_arch(&value, supported) {
                        iter.next();
                        rewritten = Some(format!("{flag}={arch}"));
                    }
                }
                break;
            }
        }
        match rewritten {
            Some(new_option) => result.push(CString::new(new_option).unwrap()),
            None => result.push(CString::new(*option).unwrap()),
        }
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcCompileProgram(
    prog: nvrtcProgram,
    num_options: c_int,
    options: *const *const c_char,
) -> nvrtcResult {
    let Some(nvrtc) = nvrtc() else {
        return nvrtcResult::ERROR_INTERNAL_ERROR;
    };
    let Some(compile) = nvrtc.nvrtcCompileProgram else {
        return nvrtcResult::ERROR_INTERNAL_ERROR;
    };
    if num_options < 0 || (num_options > 0 && options.is_null()) {
        return nvrtcResult::ERROR_INVALID_INPUT;
    }
    let original = (0..num_options as usize)
        .map(|i| {
            let option = *options.add(i);
            if option.is_null() {
                None
            } else {
                Some(CStr::from_ptr(option).to_bytes())
            }
        })
        .collect::<Option<Vec<_>>>();
    let Some(original) = original else {
        return nvrtcResult::ERROR_INVALID_OPTION;
    };
    let rewritten = rewrite_options(&original, &supported_archs(nvrtc));
    let pointers = rewritten.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
    compile(prog, pointers.len() as c_int, pointers.as_ptr())
}

// Programs are always compiled to PTX (see nvrtcCompileProgram), so return PTX instead of CUBIN.
// ZLUDA's module loading functions accept PTX in place of a CUBIN.
#[no_mangle]
pub unsafe extern "C" fn nvrtcGetCUBINSize(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult {
    match nvrtc().and_then(|nvrtc| nvrtc.nvrtcGetPTXSize) {
        Some(f) => f(prog, size),
        None => nvrtcResult::ERROR_INTERNAL_ERROR,
    }
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetCUBIN(prog: nvrtcProgram, cubin: *mut c_char) -> nvrtcResult {
    match nvrtc().and_then(|nvrtc| nvrtc.nvrtcGetPTX) {
        Some(f) => f(prog, cubin),
        None => nvrtcResult::ERROR_INTERNAL_ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUPPORTED: &[c_int] = &[50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 120];

    fn rewrite(options: &[&str]) -> Vec<String> {
        let options = options.iter().map(|s| s.as_bytes()).collect::<Vec<_>>();
        rewrite_options(&options, SUPPORTED)
            .into_iter()
            .map(|s| s.into_string().unwrap())
            .collect()
    }

    #[test]
    fn rewrites_real_arch() {
        assert_eq!(
            rewrite(&["--gpu-architecture=sm_86", "-default-device"]),
            ["--gpu-architecture=compute_86", "-default-device"]
        );
    }

    #[test]
    fn clamps_unsupported_arch() {
        assert_eq!(rewrite(&["-arch=sm_85"]), ["-arch=compute_80"]);
        assert_eq!(rewrite(&["-arch=compute_85"]), ["-arch=compute_80"]);
        assert_eq!(rewrite(&["-arch=sm_30"]), ["-arch=compute_50"]);
    }

    #[test]
    fn rewrites_separate_value() {
        assert_eq!(
            rewrite(&["-arch", "sm_90a", "-lineinfo"]),
            ["-arch=compute_90", "-lineinfo"]
        );
    }

    #[test]
    fn keeps_other_options() {
        assert_eq!(
            rewrite(&["--std=c++17", "-I/usr/include", "-architecture"]),
            ["--std=c++17", "-I/usr/include", "-architecture"]
        );
    }
}
