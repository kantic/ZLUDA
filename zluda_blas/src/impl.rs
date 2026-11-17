use cuda_types::cublas::*;
use hip_runtime_sys::hipStream_t;
use libloading::Library;
use rocblas_sys::*;
use std::ffi::OsStr;
use std::{mem, ptr, sync::OnceLock};
use zluda_common::os;
use zluda_common::{constants, from_cuda_object, ZludaObject};

fn rocblas() -> Result<&'static super::RocblasVtable, rocblas_error> {
    static LOCK: OnceLock<Result<super::RocblasVtable, rocblas_error>> = OnceLock::new();
    let unwrapped: &Result<super::RocblasVtable, rocblas_error> =
        LOCK.get_or_init(|| unsafe { super::RocblasVtable::new() });
    unwrapped.as_ref().map_err(|x| *x)
}

// This is repr(c) because it needs to be compatible with BlasHandle in zluda_blaslt.
#[repr(C)]
pub struct Handle {
    blas_lt: usize,
    blas_lt_library: Library,
    rocm_handle: rocblas_handle,
}

impl ZludaObject for Handle {
    const COOKIE: usize = constants::BLAS_HANDLE_COOKIE;
    type Error = cublasError_t;
    type CudaHandle = cublasHandle_t;
    fn drop_checked(&mut self) -> cublasStatus_t {
        unsafe {
            self.blas_lt_library
                .get::<unsafe extern "system" fn(usize) -> cublasStatus_t>(b"cublasLtDestroy\0")
        }
        // Ignore any errors from destroying embedded blas_lt handle. It may have already been destroyed.
        .map(|cublaslt_destroy| unsafe { cublaslt_destroy(self.blas_lt) })
        .ok();
        unsafe { rocblas()?.rocblas_destroy_handle(self.rocm_handle) }?;
        Ok(())
    }
}

from_cuda_object!(Handle);

#[cfg(debug_assertions)]
pub(crate) fn unimplemented() -> cublasStatus_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
pub(crate) fn unimplemented() -> cublasStatus_t {
    cublasStatus_t::ERROR_NOT_SUPPORTED
}

#[cfg(windows)]
pub(crate) fn get_cublaslt_file_name(file: &OsStr) -> Option<String> {
    let file = file.to_string_lossy().to_lowercase();
    if file.starts_with("cublas") {
        Some("cublasLt".to_owned() + &file["cublas".len()..])
    } else {
        None
    }
}

#[cfg(unix)]
pub(crate) fn get_cublaslt_file_name(_file: &OsStr) -> Option<&'static str> {
    Some("libcublaslt.so\0")
}

pub(crate) fn create_v2(handle: &mut cublasHandle_t) -> cublasStatus_t {
    let mut self_path = os::self_path().ok_or(cublasError_t::INTERNAL_ERROR)?;
    let file = self_path.file_name().ok_or(cublasError_t::INTERNAL_ERROR)?;
    let cublaslt_file_name = get_cublaslt_file_name(file).ok_or(cublasError_t::INTERNAL_ERROR)?;
    self_path.pop();
    self_path.push(cublaslt_file_name);
    let blas_lt_library = unsafe { libloading::Library::new(self_path) }
        .map_err(|_| cublasError_t::INTERNAL_ERROR)?;
    let mut blas_lt: usize = 0;
    let cublaslt_create = unsafe {
        blas_lt_library
            .get::<unsafe extern "system" fn(*mut usize) -> cublasStatus_t>(b"cublasLtCreate\0")
    }
    .map_err(|_| cublasError_t::INTERNAL_ERROR)?;
    unsafe { (cublaslt_create)(&mut blas_lt) }?;
    let mut rocm_handle = rocblas_handle(ptr::null_mut());
    unsafe { rocblas()?.rocblas_create_handle(&mut rocm_handle) }?;
    *handle = Handle::wrap(Handle {
        blas_lt: blas_lt as usize,
        blas_lt_library,
        rocm_handle,
    });
    Ok(())
}

pub(crate) fn get_status_name(status: cublasStatus_t) -> *const ::core::ffi::c_char {
    match status {
        cublasStatus_t::SUCCESS => c"CUBLAS_STATUS_SUCCESS".as_ptr(),
        cublasStatus_t::ERROR_NOT_INITIALIZED => c"CUBLAS_STATUS_NOT_INITIALIZED".as_ptr(),
        cublasStatus_t::ERROR_ALLOC_FAILED => c"CUBLAS_STATUS_ALLOC_FAILED".as_ptr(),
        cublasStatus_t::ERROR_INVALID_VALUE => c"CUBLAS_STATUS_INVALID_VALUE".as_ptr(),
        cublasStatus_t::ERROR_ARCH_MISMATCH => c"CUBLAS_STATUS_ARCH_MISMATCH".as_ptr(),
        cublasStatus_t::ERROR_MAPPING_ERROR => c"CUBLAS_STATUS_MAPPING_ERROR".as_ptr(),
        cublasStatus_t::ERROR_EXECUTION_FAILED => c"CUBLAS_STATUS_EXECUTION_FAILED".as_ptr(),
        cublasStatus_t::ERROR_INTERNAL_ERROR => c"CUBLAS_STATUS_INTERNAL_ERROR".as_ptr(),
        cublasStatus_t::ERROR_NOT_SUPPORTED => c"CUBLAS_STATUS_NOT_SUPPORTED".as_ptr(),
        cublasStatus_t::ERROR_LICENSE_ERROR => c"CUBLAS_STATUS_LICENSE_ERROR".as_ptr(),
        _ => c"CUBLAS_STATUS_UNKNOWN".as_ptr(),
    }
}

pub(crate) fn get_status_string(status: cublasStatus_t) -> *const ::core::ffi::c_char {
    if cfg!(windows) && status.is_err() && status.err() == rocblas().err().map(Into::into) {
        return c"rocblas.dll could not be found. Please install HIP SDK: https://zluda.readthedocs.io/latest/hip_sdk.html".as_ptr();
    }
    get_status_name(status)
}

pub(crate) fn xerbla(_sr_name: *const ::core::ffi::c_char, _info: ::core::ffi::c_int) -> () {
    todo!()
}

pub(crate) fn get_cudart_version() -> usize {
    todo!()
}

pub(crate) fn set_math_mode(handle: &Handle, mode: rocblas_math_mode) -> cublasStatus_t {
    unsafe { rocblas()?.rocblas_set_math_mode(handle.rocm_handle, mode) }?;
    Ok(())
}

pub(crate) fn sgemm_strided_batched(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const f32,
    a: *const f32,
    lda: ::core::ffi::c_int,
    stride_a: ::core::ffi::c_longlong,
    b: *const f32,
    ldb: ::core::ffi::c_int,
    stride_b: ::core::ffi::c_longlong,
    beta: *const f32,
    c: *mut f32,
    ldc: ::core::ffi::c_int,
    stride_c: ::core::ffi::c_longlong,
    batch_count: ::core::ffi::c_int,
) -> cublasStatus_t {
    unsafe {
        rocblas()?.rocblas_sgemm_strided_batched(
            handle.rocm_handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            stride_a,
            b,
            ldb,
            stride_b,
            beta,
            c,
            ldc,
            stride_c,
            batch_count,
        )
    }?;
    Ok(())
}

pub(crate) fn sgemm_v2(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const f32,
    a: *const f32,
    lda: ::core::ffi::c_int,
    b: *const f32,
    ldb: ::core::ffi::c_int,
    beta: *const f32,
    c: *mut f32,
    ldc: ::core::ffi::c_int,
) -> cublasStatus_t {
    unsafe {
        rocblas()?.rocblas_sgemm(
            handle.rocm_handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        )
    }?;
    Ok(())
}

pub(crate) fn destroy_v2(handle: cublasHandle_t) -> cublasStatus_t {
    zluda_common::drop_checked::<Handle>(handle)
}

pub(crate) unsafe fn set_pointer_mode_v2(
    handle: &Handle,
    mode: rocblas_pointer_mode,
) -> rocblas_status {
    rocblas()?.rocblas_set_pointer_mode(handle.rocm_handle, mode)
}

pub(crate) unsafe fn set_stream_v2(handle: &Handle, stream: hipStream_t) -> rocblas_status {
    rocblas()?.rocblas_set_stream(handle.rocm_handle, stream)
}

pub(crate) unsafe fn set_workspace_v2(
    handle: &Handle,
    workspace: *mut ::core::ffi::c_void,
    size: usize,
) -> rocblas_status {
    rocblas()?.rocblas_set_workspace(handle.rocm_handle, workspace, size)
}

pub(crate) unsafe fn get_math_mode(handle: &Handle, mode: &mut cublasMath_t) -> rocblas_status {
    let mut roc_mode = mem::zeroed();
    rocblas()?.rocblas_get_math_mode(handle.rocm_handle, &mut roc_mode)?;
    *mode = zluda_common::FromCuda::from_cuda(&roc_mode)?;
    Ok(())
}

pub(crate) unsafe fn gemm_ex(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const ::core::ffi::c_void,
    a: *const ::core::ffi::c_void,
    a_type: rocblas_datatype,
    lda: ::core::ffi::c_int,
    b: *const ::core::ffi::c_void,
    b_type: rocblas_datatype,
    ldb: ::core::ffi::c_int,
    beta: *const ::core::ffi::c_void,
    c: *mut ::core::ffi::c_void,
    c_type: rocblas_datatype,
    ldc: ::core::ffi::c_int,
    compute_type: rocblas_datatype,
    algo: rocblas_gemm_algo,
) -> rocblas_status {
    rocblas()?.rocblas_gemm_ex(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        a_type,
        lda,
        b,
        b_type,
        ldb,
        beta,
        c,
        c_type,
        ldc,
        c,
        c_type,
        ldc,
        compute_type,
        algo,
        0,
        0,
    )
}

pub(crate) unsafe fn hgemm(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const cuda_types::cublas::__half,
    a: *const cuda_types::cublas::__half,
    lda: ::core::ffi::c_int,
    b: *const cuda_types::cublas::__half,
    ldb: ::core::ffi::c_int,
    beta: *const cuda_types::cublas::__half,
    c: *mut cuda_types::cublas::__half,
    ldc: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_hgemm(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha.cast(),
        a.cast(),
        lda,
        b.cast(),
        ldb,
        beta.cast(),
        c.cast(),
        ldc,
    )
}

pub(crate) unsafe fn gemm_batched_ex(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const ::core::ffi::c_void,
    a_array: *const *const ::core::ffi::c_void,
    a_type: rocblas_datatype,
    lda: ::core::ffi::c_int,
    b_array: *const *const ::core::ffi::c_void,
    b_type: rocblas_datatype,
    ldb: ::core::ffi::c_int,
    beta: *const ::core::ffi::c_void,
    c_array: *const *mut ::core::ffi::c_void,
    c_type: rocblas_datatype,
    ldc: ::core::ffi::c_int,
    batch_count: ::core::ffi::c_int,
    compute_type: rocblas_datatype,
    algo: rocblas_gemm_algo,
) -> rocblas_status {
    rocblas()?.rocblas_gemm_batched_ex(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a_array.cast(),
        a_type,
        lda,
        b_array.cast(),
        b_type,
        ldb,
        beta,
        c_array.cast(),
        c_type,
        ldc,
        c_array.cast_mut().cast(),
        c_type,
        ldc,
        batch_count,
        compute_type,
        algo,
        -1,
        0,
    )
}

pub(crate) unsafe fn gemm_strided_batched_ex(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const ::core::ffi::c_void,
    a: *const ::core::ffi::c_void,
    a_type: rocblas_datatype,
    lda: ::core::ffi::c_int,
    stride_a: ::core::ffi::c_longlong,
    b: *const ::core::ffi::c_void,
    b_type: rocblas_datatype,
    ldb: ::core::ffi::c_int,
    stride_b: ::core::ffi::c_longlong,
    beta: *const ::core::ffi::c_void,
    c: *mut ::core::ffi::c_void,
    c_type: rocblas_datatype,
    ldc: ::core::ffi::c_int,
    stride_c: ::core::ffi::c_longlong,
    batch_count: ::core::ffi::c_int,
    compute_type: rocblas_datatype,
    algo: rocblas_gemm_algo,
) -> rocblas_status {
    rocblas()?.rocblas_gemm_strided_batched_ex(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        a_type,
        lda,
        stride_a,
        b,
        b_type,
        ldb,
        stride_b,
        beta,
        c,
        c_type,
        ldc,
        stride_c,
        c,
        c_type,
        ldc,
        stride_c,
        batch_count,
        compute_type,
        algo,
        -1,
        0,
    )
}

pub(crate) unsafe fn set_vector(
    n: ::core::ffi::c_int,
    elem_size: ::core::ffi::c_int,
    x: *const ::core::ffi::c_void,
    incx: ::core::ffi::c_int,
    device_ptr: *mut ::core::ffi::c_void,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_set_vector(n, elem_size, x, incx, device_ptr, incy)
}

pub(crate) unsafe fn get_vector(
    n: ::core::ffi::c_int,
    elem_size: ::core::ffi::c_int,
    x: *const ::core::ffi::c_void,
    incx: ::core::ffi::c_int,
    y: *mut ::core::ffi::c_void,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_get_vector(n, elem_size, x, incx, y, incy)
}

#[cfg(test)]
mod tests {
    use crate::tests::CublasApi;
    use crate::tests::CublasLtApi;
    use cuda_macros::test_cuda;

    #[test_cuda]
    fn create_destroy(api: impl CublasApi) {
        let mut handle = unsafe { std::mem::zeroed() };
        api.cublasCreate_v2(&mut handle);
        api.cublasDestroy_v2(handle);
    }

    #[test_cuda]
    fn can_pass_into_blaslt(api: impl CublasApi) {
        use cuda_types::cublas::cublasComputeType_t;
        use cuda_types::cuda::cudaDataType_t;

        let mut handle = unsafe { std::mem::zeroed() };
        api.cublasCreate_v2(&mut handle);

        let mut pref = unsafe { std::mem::zeroed() };
        api.blaslt().cublasLtMatmulPreferenceCreate(&mut pref);

        let mut matmul_desc = unsafe { std::mem::zeroed() };
        api.blaslt().cublasLtMatmulDescCreate(
            &mut matmul_desc,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType_t::CUDA_R_32F,
        );

        let m: u64 = 64;
        let n: u64 = 64;
        let k: u64 = 64;

        let mut a_desc = unsafe { std::mem::zeroed() };
        api.blaslt().cublasLtMatrixLayoutCreate(
            &mut a_desc,
            cudaDataType_t::CUDA_R_32F,
            m,
            k,
            m as i64,
        );

        let mut b_desc = unsafe { std::mem::zeroed() };
        api.blaslt().cublasLtMatrixLayoutCreate(
            &mut b_desc,
            cudaDataType_t::CUDA_R_32F,
            k,
            n,
            k as i64,
        );

        let mut c_desc = unsafe { std::mem::zeroed() };
        api.blaslt().cublasLtMatrixLayoutCreate(
            &mut c_desc,
            cudaDataType_t::CUDA_R_32F,
            m,
            n,
            m as i64,
        );

        let mut d_desc = unsafe { std::mem::zeroed() };
        api.blaslt().cublasLtMatrixLayoutCreate(
            &mut d_desc,
            cudaDataType_t::CUDA_R_32F,
            m,
            n,
            m as i64,
        );

        let mut heuristic_result = unsafe { std::mem::zeroed() };
        let mut return_algo_count: i32 = 0;
        api.blaslt().cublasLtMatmulAlgoGetHeuristic(
            unsafe { std::mem::transmute(handle) },
            matmul_desc,
            a_desc,
            b_desc,
            c_desc,
            d_desc,
            pref,
            1,
            &mut heuristic_result,
            &mut return_algo_count,
        );

        api.blaslt().cublasLtMatrixLayoutDestroy(d_desc);
        api.blaslt().cublasLtMatrixLayoutDestroy(c_desc);
        api.blaslt().cublasLtMatrixLayoutDestroy(b_desc);
        api.blaslt().cublasLtMatrixLayoutDestroy(a_desc);
        api.blaslt().cublasLtMatmulDescDestroy(matmul_desc);
        api.blaslt().cublasLtMatmulPreferenceDestroy(pref);
        api.cublasDestroy_v2(handle);
    }

    #[test_cuda]
    fn lt_destroy_blas_handle(api: impl CublasApi) {
        let mut handle = unsafe { std::mem::zeroed() };
        api.cublasCreate_v2(&mut handle);
        api.blaslt()
            .cublasLtDestroy(unsafe { std::mem::transmute(handle) });
        let mut math_mode = unsafe { std::mem::zeroed() };
        api.cublasGetMathMode(handle, &mut math_mode);
        api.cublasDestroy_v2(handle);
    }
}

pub(crate) unsafe fn set_atomics_mode(
    handle: &Handle,
    mode: rocblas_atomics_mode,
) -> rocblas_status {
    rocblas()?.rocblas_set_atomics_mode(handle.rocm_handle, mode)
}

pub(crate) unsafe fn get_atomics_mode(
    handle: &Handle,
    mode: *mut rocblas_atomics_mode,
) -> rocblas_status {
    rocblas()?.rocblas_get_atomics_mode(handle.rocm_handle, mode)
}

pub(crate) unsafe fn get_pointer_mode_v2(
    handle: &Handle,
    mode: *mut rocblas_pointer_mode,
) -> rocblas_status {
    rocblas()?.rocblas_get_pointer_mode(handle.rocm_handle, mode)
}

pub(crate) unsafe fn get_stream_v2(handle: &Handle, stream: *mut hipStream_t) -> rocblas_status {
    rocblas()?.rocblas_get_stream(handle.rocm_handle, stream)
}

// We report the version of cuBLAS shipped with CUDA 12.8
const CUBLAS_VER_MAJOR: i32 = 12;
const CUBLAS_VER_MINOR: i32 = 8;
const CUBLAS_VER_PATCH: i32 = 4;

pub(crate) unsafe fn get_version_v2(_handle: &Handle, version: &mut ::core::ffi::c_int) -> cublasStatus_t {
    *version = CUBLAS_VER_MAJOR * 10000 + CUBLAS_VER_MINOR * 100 + CUBLAS_VER_PATCH;
    Ok(())
}

pub(crate) unsafe fn get_property(
    type_: cuda_types::cublas::libraryPropertyType,
    value: &mut ::core::ffi::c_int,
) -> cublasStatus_t {
    *value = match type_ {
        cuda_types::cublas::libraryPropertyType::MAJOR_VERSION => CUBLAS_VER_MAJOR,
        cuda_types::cublas::libraryPropertyType::MINOR_VERSION => CUBLAS_VER_MINOR,
        cuda_types::cublas::libraryPropertyType::PATCH_LEVEL => CUBLAS_VER_PATCH,
        _ => return cublasStatus_t::ERROR_INVALID_VALUE,
    };
    Ok(())
}

macro_rules! dot {
    ($($name:ident => $roc:ident: $t:ty),*) => {
        $(
            pub(crate) unsafe fn $name(
                handle: &Handle,
                n: ::core::ffi::c_int,
                x: *const $t,
                incx: ::core::ffi::c_int,
                y: *const $t,
                incy: ::core::ffi::c_int,
                result: *mut $t,
            ) -> rocblas_status {
                rocblas()?.$roc(handle.rocm_handle, n, x, incx, y, incy, result)
            }
        )*
    };
}

dot!(sdot_v2 => rocblas_sdot: f32, ddot_v2 => rocblas_ddot: f64);

pub(crate) unsafe fn dgemm_v2(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const f64,
    a: *const f64,
    lda: ::core::ffi::c_int,
    b: *const f64,
    ldb: ::core::ffi::c_int,
    beta: *const f64,
    c: *mut f64,
    ldc: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_dgemm(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        lda,
        b,
        ldb,
        beta,
        c,
        ldc,
    )
}

pub(crate) unsafe fn dgemm_strided_batched(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const f64,
    a: *const f64,
    lda: ::core::ffi::c_int,
    stride_a: ::core::ffi::c_longlong,
    b: *const f64,
    ldb: ::core::ffi::c_int,
    stride_b: ::core::ffi::c_longlong,
    beta: *const f64,
    c: *mut f64,
    ldc: ::core::ffi::c_int,
    stride_c: ::core::ffi::c_longlong,
    batch_count: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_dgemm_strided_batched(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a,
        lda,
        stride_a,
        b,
        ldb,
        stride_b,
        beta,
        c,
        ldc,
        stride_c,
        batch_count,
    )
}

pub(crate) unsafe fn hgemm_strided_batched(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const cuda_types::cublas::__half,
    a: *const cuda_types::cublas::__half,
    lda: ::core::ffi::c_int,
    stride_a: ::core::ffi::c_longlong,
    b: *const cuda_types::cublas::__half,
    ldb: ::core::ffi::c_int,
    stride_b: ::core::ffi::c_longlong,
    beta: *const cuda_types::cublas::__half,
    c: *mut cuda_types::cublas::__half,
    ldc: ::core::ffi::c_int,
    stride_c: ::core::ffi::c_longlong,
    batch_count: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_hgemm_strided_batched(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha.cast(),
        a.cast(),
        lda,
        stride_a,
        b.cast(),
        ldb,
        stride_b,
        beta.cast(),
        c.cast(),
        ldc,
        stride_c,
        batch_count,
    )
}

// cublasSgemmEx: alpha, beta and compute type are always f32, matrices can be of lower precision
pub(crate) unsafe fn sgemm_ex(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const f32,
    a: *const ::core::ffi::c_void,
    a_type: rocblas_datatype,
    lda: ::core::ffi::c_int,
    b: *const ::core::ffi::c_void,
    b_type: rocblas_datatype,
    ldb: ::core::ffi::c_int,
    beta: *const f32,
    c: *mut ::core::ffi::c_void,
    c_type: rocblas_datatype,
    ldc: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_gemm_ex(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha.cast(),
        a,
        a_type,
        lda,
        b,
        b_type,
        ldb,
        beta.cast(),
        c,
        c_type,
        ldc,
        c,
        c_type,
        ldc,
        rocblas_datatype::rocblas_datatype_f32_r,
        rocblas_gemm_algo::rocblas_gemm_algo_standard,
        0,
        0,
    )
}

macro_rules! syrk {
    ($($name:ident => $roc:ident: $t:ty),*) => {
        $(
            pub(crate) unsafe fn $name(
                handle: &Handle,
                uplo: rocblas_fill,
                trans: rocblas_operation,
                n: ::core::ffi::c_int,
                k: ::core::ffi::c_int,
                alpha: *const $t,
                a: *const $t,
                lda: ::core::ffi::c_int,
                beta: *const $t,
                c: *mut $t,
                ldc: ::core::ffi::c_int,
            ) -> rocblas_status {
                rocblas()?.$roc(handle.rocm_handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
            }
        )*
    };
}

syrk!(ssyrk_v2 => rocblas_ssyrk: f32, dsyrk_v2 => rocblas_dsyrk: f64);

macro_rules! trmm {
    ($($name:ident => $roc:ident: $t:ty),*) => {
        $(
            pub(crate) unsafe fn $name(
                handle: &Handle,
                side: rocblas_side,
                uplo: rocblas_fill,
                trans: rocblas_operation,
                diag: rocblas_diagonal,
                m: ::core::ffi::c_int,
                n: ::core::ffi::c_int,
                alpha: *const $t,
                a: *const $t,
                lda: ::core::ffi::c_int,
                b: *const $t,
                ldb: ::core::ffi::c_int,
                c: *mut $t,
                ldc: ::core::ffi::c_int,
            ) -> rocblas_status {
                rocblas()?.$roc(
                    handle.rocm_handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb, c, ldc,
                )
            }
        )*
    };
}

trmm!(strmm_v2 => rocblas_strmm: f32, dtrmm_v2 => rocblas_dtrmm: f64);

macro_rules! trsm {
    ($($name:ident => $roc:ident: $t:ty),*) => {
        $(
            pub(crate) unsafe fn $name(
                handle: &Handle,
                side: rocblas_side,
                uplo: rocblas_fill,
                trans: rocblas_operation,
                diag: rocblas_diagonal,
                m: ::core::ffi::c_int,
                n: ::core::ffi::c_int,
                alpha: *const $t,
                a: *const $t,
                lda: ::core::ffi::c_int,
                b: *mut $t,
                ldb: ::core::ffi::c_int,
            ) -> rocblas_status {
                rocblas()?.$roc(handle.rocm_handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
            }
        )*
    };
}

trsm!(strsm_v2 => rocblas_strsm: f32, dtrsm_v2 => rocblas_dtrsm: f64);

// cuComplex/cuDoubleComplex (float2/double2) have the same layout as
// rocblas_float_complex/rocblas_double_complex, so complex pointers are passed through with a cast.

macro_rules! axpy {
    ($($name:ident => $roc:ident: $t:ty as $rt:ty),*) => {
        $(
            pub(crate) unsafe fn $name(
                handle: &Handle,
                n: ::core::ffi::c_int,
                alpha: *const $t,
                x: *const $t,
                incx: ::core::ffi::c_int,
                y: *mut $t,
                incy: ::core::ffi::c_int,
            ) -> rocblas_status {
                rocblas()?.$roc(handle.rocm_handle, n, alpha.cast::<$rt>(), x.cast::<$rt>(), incx, y.cast::<$rt>(), incy)
            }
        )*
    };
}

axpy!(saxpy_v2 => rocblas_saxpy: f32 as f32, daxpy_v2 => rocblas_daxpy: f64 as f64);

macro_rules! dotu {
    ($($name:ident => $roc:ident: $t:ty as $rt:ty),*) => {
        $(
            pub(crate) unsafe fn $name(
                handle: &Handle,
                n: ::core::ffi::c_int,
                x: *const $t,
                incx: ::core::ffi::c_int,
                y: *const $t,
                incy: ::core::ffi::c_int,
                result: *mut $t,
            ) -> rocblas_status {
                rocblas()?.$roc(handle.rocm_handle, n, x.cast::<$rt>(), incx, y.cast::<$rt>(), incy, result.cast::<$rt>())
            }
        )*
    };
}

dotu!(
    cdotu_v2 => rocblas_cdotu: cuComplex as rocblas_float_complex,
    zdotu_v2 => rocblas_zdotu: cuDoubleComplex as rocblas_double_complex
);

// cuBLAS accepts a NULL beta in geam when B isn't needed (cuSOLVER passes one when it only
// transposes A), rocBLAS rejects it. With host pointers, substitute a zero. With device pointers
// there's no zero to point to, so the call is passed through unchanged.
unsafe fn geam_beta<T>(handle: &Handle, beta: *const T, zero: &T) -> Result<*const T, rocblas_error> {
    if !beta.is_null() {
        return Ok(beta);
    }
    let mut mode = rocblas_pointer_mode::rocblas_pointer_mode_host;
    rocblas()?.rocblas_get_pointer_mode(handle.rocm_handle, &mut mode)?;
    Ok(if mode == rocblas_pointer_mode::rocblas_pointer_mode_host {
        zero
    } else {
        beta
    })
}

macro_rules! geam {
    ($($name:ident => $roc:ident: $t:ty as $rt:ty),*) => {
        $(
            pub(crate) unsafe fn $name(
                handle: &Handle,
                transa: rocblas_operation,
                transb: rocblas_operation,
                m: ::core::ffi::c_int,
                n: ::core::ffi::c_int,
                alpha: *const $t,
                a: *const $t,
                lda: ::core::ffi::c_int,
                beta: *const $t,
                b: *const $t,
                ldb: ::core::ffi::c_int,
                c: *mut $t,
                ldc: ::core::ffi::c_int,
            ) -> rocblas_status {
                let zero: $t = std::mem::zeroed();
                let beta = geam_beta(handle, beta, &zero)?;
                rocblas()?.$roc(
                    handle.rocm_handle,
                    transa,
                    transb,
                    m,
                    n,
                    alpha.cast::<$rt>(),
                    a.cast::<$rt>(),
                    lda,
                    beta.cast::<$rt>(),
                    b.cast::<$rt>(),
                    ldb,
                    c.cast::<$rt>(),
                    ldc,
                )
            }
        )*
    };
}

geam!(
    sgeam => rocblas_sgeam: f32 as f32,
    dgeam => rocblas_dgeam: f64 as f64,
    cgeam => rocblas_cgeam: cuComplex as rocblas_float_complex,
    zgeam => rocblas_zgeam: cuDoubleComplex as rocblas_double_complex
);

macro_rules! gemv {
    ($($name:ident => $roc:ident: $t:ty as $rt:ty),*) => {
        $(
            pub(crate) unsafe fn $name(
                handle: &Handle,
                trans: rocblas_operation,
                m: ::core::ffi::c_int,
                n: ::core::ffi::c_int,
                alpha: *const $t,
                a: *const $t,
                lda: ::core::ffi::c_int,
                x: *const $t,
                incx: ::core::ffi::c_int,
                beta: *const $t,
                y: *mut $t,
                incy: ::core::ffi::c_int,
            ) -> rocblas_status {
                rocblas()?.$roc(
                    handle.rocm_handle,
                    trans,
                    m,
                    n,
                    alpha.cast::<$rt>(),
                    a.cast::<$rt>(),
                    lda,
                    x.cast::<$rt>(),
                    incx,
                    beta.cast::<$rt>(),
                    y.cast::<$rt>(),
                    incy,
                )
            }
        )*
    };
}

gemv!(
    sgemv_v2 => rocblas_sgemv: f32 as f32,
    dgemv_v2 => rocblas_dgemv: f64 as f64,
    cgemv_v2 => rocblas_cgemv: cuComplex as rocblas_float_complex,
    zgemv_v2 => rocblas_zgemv: cuDoubleComplex as rocblas_double_complex
);

macro_rules! complex_gemm {
    ($($name:ident => $roc:ident: $t:ty as $rt:ty),*) => {
        $(
            pub(crate) unsafe fn $name(
                handle: &Handle,
                transa: rocblas_operation,
                transb: rocblas_operation,
                m: ::core::ffi::c_int,
                n: ::core::ffi::c_int,
                k: ::core::ffi::c_int,
                alpha: *const $t,
                a: *const $t,
                lda: ::core::ffi::c_int,
                b: *const $t,
                ldb: ::core::ffi::c_int,
                beta: *const $t,
                c: *mut $t,
                ldc: ::core::ffi::c_int,
            ) -> rocblas_status {
                rocblas()?.$roc(
                    handle.rocm_handle,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    alpha.cast::<$rt>(),
                    a.cast::<$rt>(),
                    lda,
                    b.cast::<$rt>(),
                    ldb,
                    beta.cast::<$rt>(),
                    c.cast::<$rt>(),
                    ldc,
                )
            }
        )*
    };
}

complex_gemm!(
    cgemm_v2 => rocblas_cgemm: cuComplex as rocblas_float_complex,
    zgemm_v2 => rocblas_zgemm: cuDoubleComplex as rocblas_double_complex
);

// Level 1 and rank-1 update functions not covered by the macros above, in 32-bit and 64-bit
// integer (ILP64) variants. They map one-to-one to rocBLAS; LAPACK-style code (e.g. cuSOLVER's
// LU and SVD) uses them.

pub(crate) unsafe fn isamax_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f32,
    incx: ::core::ffi::c_int,
    result: *mut ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_isamax(handle.rocm_handle, n, x.cast::<f32>(), incx, result)
}

pub(crate) unsafe fn isamax_v2_64(
    handle: &Handle,
    n: i64,
    x: *const f32,
    incx: i64,
    result: *mut i64,
) -> rocblas_status {
    rocblas()?.rocblas_isamax_64(handle.rocm_handle, n, x.cast::<f32>(), incx, result)
}

pub(crate) unsafe fn isamin_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f32,
    incx: ::core::ffi::c_int,
    result: *mut ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_isamin(handle.rocm_handle, n, x.cast::<f32>(), incx, result)
}

pub(crate) unsafe fn isamin_v2_64(
    handle: &Handle,
    n: i64,
    x: *const f32,
    incx: i64,
    result: *mut i64,
) -> rocblas_status {
    rocblas()?.rocblas_isamin_64(handle.rocm_handle, n, x.cast::<f32>(), incx, result)
}

pub(crate) unsafe fn sscal_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    alpha: *const f32,
    x: *mut f32,
    incx: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_sscal(handle.rocm_handle, n, alpha.cast::<f32>(), x.cast::<f32>(), incx)
}

pub(crate) unsafe fn sscal_v2_64(
    handle: &Handle,
    n: i64,
    alpha: *const f32,
    x: *mut f32,
    incx: i64,
) -> rocblas_status {
    rocblas()?.rocblas_sscal_64(handle.rocm_handle, n, alpha.cast::<f32>(), x.cast::<f32>(), incx)
}

pub(crate) unsafe fn sswap_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *mut f32,
    incx: ::core::ffi::c_int,
    y: *mut f32,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_sswap(handle.rocm_handle, n, x.cast::<f32>(), incx, y.cast::<f32>(), incy)
}

pub(crate) unsafe fn sswap_v2_64(
    handle: &Handle,
    n: i64,
    x: *mut f32,
    incx: i64,
    y: *mut f32,
    incy: i64,
) -> rocblas_status {
    rocblas()?.rocblas_sswap_64(handle.rocm_handle, n, x.cast::<f32>(), incx, y.cast::<f32>(), incy)
}

pub(crate) unsafe fn scopy_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f32,
    incx: ::core::ffi::c_int,
    y: *mut f32,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_scopy(handle.rocm_handle, n, x.cast::<f32>(), incx, y.cast::<f32>(), incy)
}

pub(crate) unsafe fn scopy_v2_64(
    handle: &Handle,
    n: i64,
    x: *const f32,
    incx: i64,
    y: *mut f32,
    incy: i64,
) -> rocblas_status {
    rocblas()?.rocblas_scopy_64(handle.rocm_handle, n, x.cast::<f32>(), incx, y.cast::<f32>(), incy)
}

pub(crate) unsafe fn saxpy_v2_64(
    handle: &Handle,
    n: i64,
    alpha: *const f32,
    x: *const f32,
    incx: i64,
    y: *mut f32,
    incy: i64,
) -> rocblas_status {
    rocblas()?.rocblas_saxpy_64(handle.rocm_handle, n, alpha.cast::<f32>(), x.cast::<f32>(), incx, y.cast::<f32>(), incy)
}

pub(crate) unsafe fn idamax_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f64,
    incx: ::core::ffi::c_int,
    result: *mut ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_idamax(handle.rocm_handle, n, x.cast::<f64>(), incx, result)
}

pub(crate) unsafe fn idamax_v2_64(
    handle: &Handle,
    n: i64,
    x: *const f64,
    incx: i64,
    result: *mut i64,
) -> rocblas_status {
    rocblas()?.rocblas_idamax_64(handle.rocm_handle, n, x.cast::<f64>(), incx, result)
}

pub(crate) unsafe fn idamin_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f64,
    incx: ::core::ffi::c_int,
    result: *mut ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_idamin(handle.rocm_handle, n, x.cast::<f64>(), incx, result)
}

pub(crate) unsafe fn idamin_v2_64(
    handle: &Handle,
    n: i64,
    x: *const f64,
    incx: i64,
    result: *mut i64,
) -> rocblas_status {
    rocblas()?.rocblas_idamin_64(handle.rocm_handle, n, x.cast::<f64>(), incx, result)
}

pub(crate) unsafe fn dscal_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    alpha: *const f64,
    x: *mut f64,
    incx: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_dscal(handle.rocm_handle, n, alpha.cast::<f64>(), x.cast::<f64>(), incx)
}

pub(crate) unsafe fn dscal_v2_64(
    handle: &Handle,
    n: i64,
    alpha: *const f64,
    x: *mut f64,
    incx: i64,
) -> rocblas_status {
    rocblas()?.rocblas_dscal_64(handle.rocm_handle, n, alpha.cast::<f64>(), x.cast::<f64>(), incx)
}

pub(crate) unsafe fn dswap_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *mut f64,
    incx: ::core::ffi::c_int,
    y: *mut f64,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_dswap(handle.rocm_handle, n, x.cast::<f64>(), incx, y.cast::<f64>(), incy)
}

pub(crate) unsafe fn dswap_v2_64(
    handle: &Handle,
    n: i64,
    x: *mut f64,
    incx: i64,
    y: *mut f64,
    incy: i64,
) -> rocblas_status {
    rocblas()?.rocblas_dswap_64(handle.rocm_handle, n, x.cast::<f64>(), incx, y.cast::<f64>(), incy)
}

pub(crate) unsafe fn dcopy_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f64,
    incx: ::core::ffi::c_int,
    y: *mut f64,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_dcopy(handle.rocm_handle, n, x.cast::<f64>(), incx, y.cast::<f64>(), incy)
}

pub(crate) unsafe fn dcopy_v2_64(
    handle: &Handle,
    n: i64,
    x: *const f64,
    incx: i64,
    y: *mut f64,
    incy: i64,
) -> rocblas_status {
    rocblas()?.rocblas_dcopy_64(handle.rocm_handle, n, x.cast::<f64>(), incx, y.cast::<f64>(), incy)
}

pub(crate) unsafe fn daxpy_v2_64(
    handle: &Handle,
    n: i64,
    alpha: *const f64,
    x: *const f64,
    incx: i64,
    y: *mut f64,
    incy: i64,
) -> rocblas_status {
    rocblas()?.rocblas_daxpy_64(handle.rocm_handle, n, alpha.cast::<f64>(), x.cast::<f64>(), incx, y.cast::<f64>(), incy)
}

pub(crate) unsafe fn icamax_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const cuComplex,
    incx: ::core::ffi::c_int,
    result: *mut ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_icamax(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, result)
}

pub(crate) unsafe fn icamax_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuComplex,
    incx: i64,
    result: *mut i64,
) -> rocblas_status {
    rocblas()?.rocblas_icamax_64(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, result)
}

pub(crate) unsafe fn icamin_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const cuComplex,
    incx: ::core::ffi::c_int,
    result: *mut ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_icamin(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, result)
}

pub(crate) unsafe fn icamin_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuComplex,
    incx: i64,
    result: *mut i64,
) -> rocblas_status {
    rocblas()?.rocblas_icamin_64(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, result)
}

pub(crate) unsafe fn cscal_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    alpha: *const cuComplex,
    x: *mut cuComplex,
    incx: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_cscal(handle.rocm_handle, n, alpha.cast::<rocblas_float_complex>(), x.cast::<rocblas_float_complex>(), incx)
}

pub(crate) unsafe fn cscal_v2_64(
    handle: &Handle,
    n: i64,
    alpha: *const cuComplex,
    x: *mut cuComplex,
    incx: i64,
) -> rocblas_status {
    rocblas()?.rocblas_cscal_64(handle.rocm_handle, n, alpha.cast::<rocblas_float_complex>(), x.cast::<rocblas_float_complex>(), incx)
}

pub(crate) unsafe fn cswap_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *mut cuComplex,
    incx: ::core::ffi::c_int,
    y: *mut cuComplex,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_cswap(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy)
}

pub(crate) unsafe fn cswap_v2_64(
    handle: &Handle,
    n: i64,
    x: *mut cuComplex,
    incx: i64,
    y: *mut cuComplex,
    incy: i64,
) -> rocblas_status {
    rocblas()?.rocblas_cswap_64(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy)
}

pub(crate) unsafe fn ccopy_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const cuComplex,
    incx: ::core::ffi::c_int,
    y: *mut cuComplex,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_ccopy(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy)
}

pub(crate) unsafe fn ccopy_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuComplex,
    incx: i64,
    y: *mut cuComplex,
    incy: i64,
) -> rocblas_status {
    rocblas()?.rocblas_ccopy_64(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy)
}

pub(crate) unsafe fn caxpy_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    alpha: *const cuComplex,
    x: *const cuComplex,
    incx: ::core::ffi::c_int,
    y: *mut cuComplex,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_caxpy(handle.rocm_handle, n, alpha.cast::<rocblas_float_complex>(), x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy)
}

pub(crate) unsafe fn caxpy_v2_64(
    handle: &Handle,
    n: i64,
    alpha: *const cuComplex,
    x: *const cuComplex,
    incx: i64,
    y: *mut cuComplex,
    incy: i64,
) -> rocblas_status {
    rocblas()?.rocblas_caxpy_64(handle.rocm_handle, n, alpha.cast::<rocblas_float_complex>(), x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy)
}

pub(crate) unsafe fn izamax_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const cuDoubleComplex,
    incx: ::core::ffi::c_int,
    result: *mut ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_izamax(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, result)
}

pub(crate) unsafe fn izamax_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuDoubleComplex,
    incx: i64,
    result: *mut i64,
) -> rocblas_status {
    rocblas()?.rocblas_izamax_64(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, result)
}

pub(crate) unsafe fn izamin_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const cuDoubleComplex,
    incx: ::core::ffi::c_int,
    result: *mut ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_izamin(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, result)
}

pub(crate) unsafe fn izamin_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuDoubleComplex,
    incx: i64,
    result: *mut i64,
) -> rocblas_status {
    rocblas()?.rocblas_izamin_64(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, result)
}

pub(crate) unsafe fn zscal_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    alpha: *const cuDoubleComplex,
    x: *mut cuDoubleComplex,
    incx: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_zscal(handle.rocm_handle, n, alpha.cast::<rocblas_double_complex>(), x.cast::<rocblas_double_complex>(), incx)
}

pub(crate) unsafe fn zscal_v2_64(
    handle: &Handle,
    n: i64,
    alpha: *const cuDoubleComplex,
    x: *mut cuDoubleComplex,
    incx: i64,
) -> rocblas_status {
    rocblas()?.rocblas_zscal_64(handle.rocm_handle, n, alpha.cast::<rocblas_double_complex>(), x.cast::<rocblas_double_complex>(), incx)
}

pub(crate) unsafe fn zswap_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *mut cuDoubleComplex,
    incx: ::core::ffi::c_int,
    y: *mut cuDoubleComplex,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_zswap(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy)
}

pub(crate) unsafe fn zswap_v2_64(
    handle: &Handle,
    n: i64,
    x: *mut cuDoubleComplex,
    incx: i64,
    y: *mut cuDoubleComplex,
    incy: i64,
) -> rocblas_status {
    rocblas()?.rocblas_zswap_64(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy)
}

pub(crate) unsafe fn zcopy_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const cuDoubleComplex,
    incx: ::core::ffi::c_int,
    y: *mut cuDoubleComplex,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_zcopy(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy)
}

pub(crate) unsafe fn zcopy_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuDoubleComplex,
    incx: i64,
    y: *mut cuDoubleComplex,
    incy: i64,
) -> rocblas_status {
    rocblas()?.rocblas_zcopy_64(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy)
}

pub(crate) unsafe fn zaxpy_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    alpha: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: ::core::ffi::c_int,
    y: *mut cuDoubleComplex,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_zaxpy(handle.rocm_handle, n, alpha.cast::<rocblas_double_complex>(), x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy)
}

pub(crate) unsafe fn zaxpy_v2_64(
    handle: &Handle,
    n: i64,
    alpha: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: i64,
    y: *mut cuDoubleComplex,
    incy: i64,
) -> rocblas_status {
    rocblas()?.rocblas_zaxpy_64(handle.rocm_handle, n, alpha.cast::<rocblas_double_complex>(), x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy)
}

pub(crate) unsafe fn sger_v2(
    handle: &Handle,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    alpha: *const f32,
    x: *const f32,
    incx: ::core::ffi::c_int,
    y: *const f32,
    incy: ::core::ffi::c_int,
    a: *mut f32,
    lda: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_sger(handle.rocm_handle, m, n, alpha.cast::<f32>(), x.cast::<f32>(), incx, y.cast::<f32>(), incy, a.cast::<f32>(), lda)
}

pub(crate) unsafe fn sger_v2_64(
    handle: &Handle,
    m: i64,
    n: i64,
    alpha: *const f32,
    x: *const f32,
    incx: i64,
    y: *const f32,
    incy: i64,
    a: *mut f32,
    lda: i64,
) -> rocblas_status {
    rocblas()?.rocblas_sger_64(handle.rocm_handle, m, n, alpha.cast::<f32>(), x.cast::<f32>(), incx, y.cast::<f32>(), incy, a.cast::<f32>(), lda)
}

pub(crate) unsafe fn snrm2_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f32,
    incx: ::core::ffi::c_int,
    result: *mut f32,
) -> rocblas_status {
    rocblas()?.rocblas_snrm2(handle.rocm_handle, n, x.cast::<f32>(), incx, result)
}

pub(crate) unsafe fn snrm2_v2_64(
    handle: &Handle,
    n: i64,
    x: *const f32,
    incx: i64,
    result: *mut f32,
) -> rocblas_status {
    rocblas()?.rocblas_snrm2_64(handle.rocm_handle, n, x.cast::<f32>(), incx, result)
}

pub(crate) unsafe fn sasum_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f32,
    incx: ::core::ffi::c_int,
    result: *mut f32,
) -> rocblas_status {
    rocblas()?.rocblas_sasum(handle.rocm_handle, n, x.cast::<f32>(), incx, result)
}

pub(crate) unsafe fn sasum_v2_64(
    handle: &Handle,
    n: i64,
    x: *const f32,
    incx: i64,
    result: *mut f32,
) -> rocblas_status {
    rocblas()?.rocblas_sasum_64(handle.rocm_handle, n, x.cast::<f32>(), incx, result)
}

pub(crate) unsafe fn sdot_v2_64(
    handle: &Handle,
    n: i64,
    x: *const f32,
    incx: i64,
    y: *const f32,
    incy: i64,
    result: *mut f32,
) -> rocblas_status {
    rocblas()?.rocblas_sdot_64(handle.rocm_handle, n, x.cast::<f32>(), incx, y.cast::<f32>(), incy, result.cast::<f32>())
}

pub(crate) unsafe fn dger_v2(
    handle: &Handle,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    alpha: *const f64,
    x: *const f64,
    incx: ::core::ffi::c_int,
    y: *const f64,
    incy: ::core::ffi::c_int,
    a: *mut f64,
    lda: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_dger(handle.rocm_handle, m, n, alpha.cast::<f64>(), x.cast::<f64>(), incx, y.cast::<f64>(), incy, a.cast::<f64>(), lda)
}

pub(crate) unsafe fn dger_v2_64(
    handle: &Handle,
    m: i64,
    n: i64,
    alpha: *const f64,
    x: *const f64,
    incx: i64,
    y: *const f64,
    incy: i64,
    a: *mut f64,
    lda: i64,
) -> rocblas_status {
    rocblas()?.rocblas_dger_64(handle.rocm_handle, m, n, alpha.cast::<f64>(), x.cast::<f64>(), incx, y.cast::<f64>(), incy, a.cast::<f64>(), lda)
}

pub(crate) unsafe fn dnrm2_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f64,
    incx: ::core::ffi::c_int,
    result: *mut f64,
) -> rocblas_status {
    rocblas()?.rocblas_dnrm2(handle.rocm_handle, n, x.cast::<f64>(), incx, result)
}

pub(crate) unsafe fn dnrm2_v2_64(
    handle: &Handle,
    n: i64,
    x: *const f64,
    incx: i64,
    result: *mut f64,
) -> rocblas_status {
    rocblas()?.rocblas_dnrm2_64(handle.rocm_handle, n, x.cast::<f64>(), incx, result)
}

pub(crate) unsafe fn dasum_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f64,
    incx: ::core::ffi::c_int,
    result: *mut f64,
) -> rocblas_status {
    rocblas()?.rocblas_dasum(handle.rocm_handle, n, x.cast::<f64>(), incx, result)
}

pub(crate) unsafe fn dasum_v2_64(
    handle: &Handle,
    n: i64,
    x: *const f64,
    incx: i64,
    result: *mut f64,
) -> rocblas_status {
    rocblas()?.rocblas_dasum_64(handle.rocm_handle, n, x.cast::<f64>(), incx, result)
}

pub(crate) unsafe fn ddot_v2_64(
    handle: &Handle,
    n: i64,
    x: *const f64,
    incx: i64,
    y: *const f64,
    incy: i64,
    result: *mut f64,
) -> rocblas_status {
    rocblas()?.rocblas_ddot_64(handle.rocm_handle, n, x.cast::<f64>(), incx, y.cast::<f64>(), incy, result.cast::<f64>())
}

pub(crate) unsafe fn cgeru_v2(
    handle: &Handle,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    alpha: *const cuComplex,
    x: *const cuComplex,
    incx: ::core::ffi::c_int,
    y: *const cuComplex,
    incy: ::core::ffi::c_int,
    a: *mut cuComplex,
    lda: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_cgeru(handle.rocm_handle, m, n, alpha.cast::<rocblas_float_complex>(), x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy, a.cast::<rocblas_float_complex>(), lda)
}

pub(crate) unsafe fn cgeru_v2_64(
    handle: &Handle,
    m: i64,
    n: i64,
    alpha: *const cuComplex,
    x: *const cuComplex,
    incx: i64,
    y: *const cuComplex,
    incy: i64,
    a: *mut cuComplex,
    lda: i64,
) -> rocblas_status {
    rocblas()?.rocblas_cgeru_64(handle.rocm_handle, m, n, alpha.cast::<rocblas_float_complex>(), x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy, a.cast::<rocblas_float_complex>(), lda)
}

pub(crate) unsafe fn cdotu_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuComplex,
    incx: i64,
    y: *const cuComplex,
    incy: i64,
    result: *mut cuComplex,
) -> rocblas_status {
    rocblas()?.rocblas_cdotu_64(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy, result.cast::<rocblas_float_complex>())
}

pub(crate) unsafe fn cgerc_v2(
    handle: &Handle,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    alpha: *const cuComplex,
    x: *const cuComplex,
    incx: ::core::ffi::c_int,
    y: *const cuComplex,
    incy: ::core::ffi::c_int,
    a: *mut cuComplex,
    lda: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_cgerc(handle.rocm_handle, m, n, alpha.cast::<rocblas_float_complex>(), x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy, a.cast::<rocblas_float_complex>(), lda)
}

pub(crate) unsafe fn cgerc_v2_64(
    handle: &Handle,
    m: i64,
    n: i64,
    alpha: *const cuComplex,
    x: *const cuComplex,
    incx: i64,
    y: *const cuComplex,
    incy: i64,
    a: *mut cuComplex,
    lda: i64,
) -> rocblas_status {
    rocblas()?.rocblas_cgerc_64(handle.rocm_handle, m, n, alpha.cast::<rocblas_float_complex>(), x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy, a.cast::<rocblas_float_complex>(), lda)
}

pub(crate) unsafe fn cdotc_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const cuComplex,
    incx: ::core::ffi::c_int,
    y: *const cuComplex,
    incy: ::core::ffi::c_int,
    result: *mut cuComplex,
) -> rocblas_status {
    rocblas()?.rocblas_cdotc(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy, result.cast::<rocblas_float_complex>())
}

pub(crate) unsafe fn cdotc_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuComplex,
    incx: i64,
    y: *const cuComplex,
    incy: i64,
    result: *mut cuComplex,
) -> rocblas_status {
    rocblas()?.rocblas_cdotc_64(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy, result.cast::<rocblas_float_complex>())
}

pub(crate) unsafe fn scnrm2_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const cuComplex,
    incx: ::core::ffi::c_int,
    result: *mut f32,
) -> rocblas_status {
    rocblas()?.rocblas_scnrm2(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, result)
}

pub(crate) unsafe fn scnrm2_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuComplex,
    incx: i64,
    result: *mut f32,
) -> rocblas_status {
    rocblas()?.rocblas_scnrm2_64(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, result)
}

pub(crate) unsafe fn scasum_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const cuComplex,
    incx: ::core::ffi::c_int,
    result: *mut f32,
) -> rocblas_status {
    rocblas()?.rocblas_scasum(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, result)
}

pub(crate) unsafe fn scasum_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuComplex,
    incx: i64,
    result: *mut f32,
) -> rocblas_status {
    rocblas()?.rocblas_scasum_64(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, result)
}

pub(crate) unsafe fn csscal_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    alpha: *const f32,
    x: *mut cuComplex,
    incx: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_csscal(handle.rocm_handle, n, alpha, x.cast::<rocblas_float_complex>(), incx)
}

pub(crate) unsafe fn csscal_v2_64(
    handle: &Handle,
    n: i64,
    alpha: *const f32,
    x: *mut cuComplex,
    incx: i64,
) -> rocblas_status {
    rocblas()?.rocblas_csscal_64(handle.rocm_handle, n, alpha, x.cast::<rocblas_float_complex>(), incx)
}

pub(crate) unsafe fn zgeru_v2(
    handle: &Handle,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    alpha: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: ::core::ffi::c_int,
    y: *const cuDoubleComplex,
    incy: ::core::ffi::c_int,
    a: *mut cuDoubleComplex,
    lda: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_zgeru(handle.rocm_handle, m, n, alpha.cast::<rocblas_double_complex>(), x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy, a.cast::<rocblas_double_complex>(), lda)
}

pub(crate) unsafe fn zgeru_v2_64(
    handle: &Handle,
    m: i64,
    n: i64,
    alpha: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: i64,
    y: *const cuDoubleComplex,
    incy: i64,
    a: *mut cuDoubleComplex,
    lda: i64,
) -> rocblas_status {
    rocblas()?.rocblas_zgeru_64(handle.rocm_handle, m, n, alpha.cast::<rocblas_double_complex>(), x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy, a.cast::<rocblas_double_complex>(), lda)
}

pub(crate) unsafe fn zdotu_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuDoubleComplex,
    incx: i64,
    y: *const cuDoubleComplex,
    incy: i64,
    result: *mut cuDoubleComplex,
) -> rocblas_status {
    rocblas()?.rocblas_zdotu_64(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy, result.cast::<rocblas_double_complex>())
}

pub(crate) unsafe fn zgerc_v2(
    handle: &Handle,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    alpha: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: ::core::ffi::c_int,
    y: *const cuDoubleComplex,
    incy: ::core::ffi::c_int,
    a: *mut cuDoubleComplex,
    lda: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_zgerc(handle.rocm_handle, m, n, alpha.cast::<rocblas_double_complex>(), x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy, a.cast::<rocblas_double_complex>(), lda)
}

pub(crate) unsafe fn zgerc_v2_64(
    handle: &Handle,
    m: i64,
    n: i64,
    alpha: *const cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: i64,
    y: *const cuDoubleComplex,
    incy: i64,
    a: *mut cuDoubleComplex,
    lda: i64,
) -> rocblas_status {
    rocblas()?.rocblas_zgerc_64(handle.rocm_handle, m, n, alpha.cast::<rocblas_double_complex>(), x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy, a.cast::<rocblas_double_complex>(), lda)
}

pub(crate) unsafe fn zdotc_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const cuDoubleComplex,
    incx: ::core::ffi::c_int,
    y: *const cuDoubleComplex,
    incy: ::core::ffi::c_int,
    result: *mut cuDoubleComplex,
) -> rocblas_status {
    rocblas()?.rocblas_zdotc(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy, result.cast::<rocblas_double_complex>())
}

pub(crate) unsafe fn zdotc_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuDoubleComplex,
    incx: i64,
    y: *const cuDoubleComplex,
    incy: i64,
    result: *mut cuDoubleComplex,
) -> rocblas_status {
    rocblas()?.rocblas_zdotc_64(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy, result.cast::<rocblas_double_complex>())
}

pub(crate) unsafe fn dznrm2_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const cuDoubleComplex,
    incx: ::core::ffi::c_int,
    result: *mut f64,
) -> rocblas_status {
    rocblas()?.rocblas_dznrm2(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, result)
}

pub(crate) unsafe fn dznrm2_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuDoubleComplex,
    incx: i64,
    result: *mut f64,
) -> rocblas_status {
    rocblas()?.rocblas_dznrm2_64(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, result)
}

pub(crate) unsafe fn dzasum_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const cuDoubleComplex,
    incx: ::core::ffi::c_int,
    result: *mut f64,
) -> rocblas_status {
    rocblas()?.rocblas_dzasum(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, result)
}

pub(crate) unsafe fn dzasum_v2_64(
    handle: &Handle,
    n: i64,
    x: *const cuDoubleComplex,
    incx: i64,
    result: *mut f64,
) -> rocblas_status {
    rocblas()?.rocblas_dzasum_64(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, result)
}

pub(crate) unsafe fn zdscal_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    alpha: *const f64,
    x: *mut cuDoubleComplex,
    incx: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_zdscal(handle.rocm_handle, n, alpha, x.cast::<rocblas_double_complex>(), incx)
}

pub(crate) unsafe fn zdscal_v2_64(
    handle: &Handle,
    n: i64,
    alpha: *const f64,
    x: *mut cuDoubleComplex,
    incx: i64,
) -> rocblas_status {
    rocblas()?.rocblas_zdscal_64(handle.rocm_handle, n, alpha, x.cast::<rocblas_double_complex>(), incx)
}

pub(crate) unsafe fn srot_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *mut f32,
    incx: ::core::ffi::c_int,
    y: *mut f32,
    incy: ::core::ffi::c_int,
    c: *const f32,
    s: *const f32,
) -> rocblas_status {
    rocblas()?.rocblas_srot(handle.rocm_handle, n, x.cast::<f32>(), incx, y.cast::<f32>(), incy, c, s)
}

pub(crate) unsafe fn srot_v2_64(
    handle: &Handle,
    n: i64,
    x: *mut f32,
    incx: i64,
    y: *mut f32,
    incy: i64,
    c: *const f32,
    s: *const f32,
) -> rocblas_status {
    rocblas()?.rocblas_srot_64(handle.rocm_handle, n, x.cast::<f32>(), incx, y.cast::<f32>(), incy, c, s)
}

pub(crate) unsafe fn srotg_v2(
    handle: &Handle,
    a: *mut f32,
    b: *mut f32,
    c: *mut f32,
    s: *mut f32,
) -> rocblas_status {
    rocblas()?.rocblas_srotg(handle.rocm_handle, a.cast::<f32>(), b.cast::<f32>(), c, s.cast::<f32>())
}

pub(crate) unsafe fn srotm_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *mut f32,
    incx: ::core::ffi::c_int,
    y: *mut f32,
    incy: ::core::ffi::c_int,
    param: *const f32,
) -> rocblas_status {
    rocblas()?.rocblas_srotm(handle.rocm_handle, n, x.cast::<f32>(), incx, y.cast::<f32>(), incy, param)
}

pub(crate) unsafe fn srotm_v2_64(
    handle: &Handle,
    n: i64,
    x: *mut f32,
    incx: i64,
    y: *mut f32,
    incy: i64,
    param: *const f32,
) -> rocblas_status {
    rocblas()?.rocblas_srotm_64(handle.rocm_handle, n, x.cast::<f32>(), incx, y.cast::<f32>(), incy, param)
}

pub(crate) unsafe fn srotmg_v2(
    handle: &Handle,
    d1: *mut f32,
    d2: *mut f32,
    x1: *mut f32,
    y1: *const f32,
    param: *mut f32,
) -> rocblas_status {
    rocblas()?.rocblas_srotmg(handle.rocm_handle, d1.cast::<f32>(), d2.cast::<f32>(), x1.cast::<f32>(), y1.cast::<f32>(), param.cast::<f32>())
}

pub(crate) unsafe fn drot_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *mut f64,
    incx: ::core::ffi::c_int,
    y: *mut f64,
    incy: ::core::ffi::c_int,
    c: *const f64,
    s: *const f64,
) -> rocblas_status {
    rocblas()?.rocblas_drot(handle.rocm_handle, n, x.cast::<f64>(), incx, y.cast::<f64>(), incy, c, s)
}

pub(crate) unsafe fn drot_v2_64(
    handle: &Handle,
    n: i64,
    x: *mut f64,
    incx: i64,
    y: *mut f64,
    incy: i64,
    c: *const f64,
    s: *const f64,
) -> rocblas_status {
    rocblas()?.rocblas_drot_64(handle.rocm_handle, n, x.cast::<f64>(), incx, y.cast::<f64>(), incy, c, s)
}

pub(crate) unsafe fn drotg_v2(
    handle: &Handle,
    a: *mut f64,
    b: *mut f64,
    c: *mut f64,
    s: *mut f64,
) -> rocblas_status {
    rocblas()?.rocblas_drotg(handle.rocm_handle, a.cast::<f64>(), b.cast::<f64>(), c, s.cast::<f64>())
}

pub(crate) unsafe fn drotm_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *mut f64,
    incx: ::core::ffi::c_int,
    y: *mut f64,
    incy: ::core::ffi::c_int,
    param: *const f64,
) -> rocblas_status {
    rocblas()?.rocblas_drotm(handle.rocm_handle, n, x.cast::<f64>(), incx, y.cast::<f64>(), incy, param)
}

pub(crate) unsafe fn drotm_v2_64(
    handle: &Handle,
    n: i64,
    x: *mut f64,
    incx: i64,
    y: *mut f64,
    incy: i64,
    param: *const f64,
) -> rocblas_status {
    rocblas()?.rocblas_drotm_64(handle.rocm_handle, n, x.cast::<f64>(), incx, y.cast::<f64>(), incy, param)
}

pub(crate) unsafe fn drotmg_v2(
    handle: &Handle,
    d1: *mut f64,
    d2: *mut f64,
    x1: *mut f64,
    y1: *const f64,
    param: *mut f64,
) -> rocblas_status {
    rocblas()?.rocblas_drotmg(handle.rocm_handle, d1.cast::<f64>(), d2.cast::<f64>(), x1.cast::<f64>(), y1.cast::<f64>(), param.cast::<f64>())
}

pub(crate) unsafe fn crot_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *mut cuComplex,
    incx: ::core::ffi::c_int,
    y: *mut cuComplex,
    incy: ::core::ffi::c_int,
    c: *const f32,
    s: *const cuComplex,
) -> rocblas_status {
    rocblas()?.rocblas_crot(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy, c, s.cast::<rocblas_float_complex>())
}

pub(crate) unsafe fn crot_v2_64(
    handle: &Handle,
    n: i64,
    x: *mut cuComplex,
    incx: i64,
    y: *mut cuComplex,
    incy: i64,
    c: *const f32,
    s: *const cuComplex,
) -> rocblas_status {
    rocblas()?.rocblas_crot_64(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy, c, s.cast::<rocblas_float_complex>())
}

pub(crate) unsafe fn csrot_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *mut cuComplex,
    incx: ::core::ffi::c_int,
    y: *mut cuComplex,
    incy: ::core::ffi::c_int,
    c: *const f32,
    s: *const f32,
) -> rocblas_status {
    rocblas()?.rocblas_csrot(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy, c, s)
}

pub(crate) unsafe fn csrot_v2_64(
    handle: &Handle,
    n: i64,
    x: *mut cuComplex,
    incx: i64,
    y: *mut cuComplex,
    incy: i64,
    c: *const f32,
    s: *const f32,
) -> rocblas_status {
    rocblas()?.rocblas_csrot_64(handle.rocm_handle, n, x.cast::<rocblas_float_complex>(), incx, y.cast::<rocblas_float_complex>(), incy, c, s)
}

pub(crate) unsafe fn crotg_v2(
    handle: &Handle,
    a: *mut cuComplex,
    b: *mut cuComplex,
    c: *mut f32,
    s: *mut cuComplex,
) -> rocblas_status {
    rocblas()?.rocblas_crotg(handle.rocm_handle, a.cast::<rocblas_float_complex>(), b.cast::<rocblas_float_complex>(), c, s.cast::<rocblas_float_complex>())
}

pub(crate) unsafe fn zrot_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *mut cuDoubleComplex,
    incx: ::core::ffi::c_int,
    y: *mut cuDoubleComplex,
    incy: ::core::ffi::c_int,
    c: *const f64,
    s: *const cuDoubleComplex,
) -> rocblas_status {
    rocblas()?.rocblas_zrot(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy, c, s.cast::<rocblas_double_complex>())
}

pub(crate) unsafe fn zrot_v2_64(
    handle: &Handle,
    n: i64,
    x: *mut cuDoubleComplex,
    incx: i64,
    y: *mut cuDoubleComplex,
    incy: i64,
    c: *const f64,
    s: *const cuDoubleComplex,
) -> rocblas_status {
    rocblas()?.rocblas_zrot_64(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy, c, s.cast::<rocblas_double_complex>())
}

pub(crate) unsafe fn zdrot_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *mut cuDoubleComplex,
    incx: ::core::ffi::c_int,
    y: *mut cuDoubleComplex,
    incy: ::core::ffi::c_int,
    c: *const f64,
    s: *const f64,
) -> rocblas_status {
    rocblas()?.rocblas_zdrot(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy, c, s)
}

pub(crate) unsafe fn zdrot_v2_64(
    handle: &Handle,
    n: i64,
    x: *mut cuDoubleComplex,
    incx: i64,
    y: *mut cuDoubleComplex,
    incy: i64,
    c: *const f64,
    s: *const f64,
) -> rocblas_status {
    rocblas()?.rocblas_zdrot_64(handle.rocm_handle, n, x.cast::<rocblas_double_complex>(), incx, y.cast::<rocblas_double_complex>(), incy, c, s)
}

pub(crate) unsafe fn zrotg_v2(
    handle: &Handle,
    a: *mut cuDoubleComplex,
    b: *mut cuDoubleComplex,
    c: *mut f64,
    s: *mut cuDoubleComplex,
) -> rocblas_status {
    rocblas()?.rocblas_zrotg(handle.rocm_handle, a.cast::<rocblas_double_complex>(), b.cast::<rocblas_double_complex>(), c, s.cast::<rocblas_double_complex>())
}

// 64-bit integer (ILP64) variants, used e.g. by cuSOLVER

macro_rules! gemm_64 {
    ($($name:ident => $roc:ident: $t:ty as $rt:ty),*) => {
        $(
            pub(crate) unsafe fn $name(
                handle: &Handle,
                transa: rocblas_operation,
                transb: rocblas_operation,
                m: i64,
                n: i64,
                k: i64,
                alpha: *const $t,
                a: *const $t,
                lda: i64,
                b: *const $t,
                ldb: i64,
                beta: *const $t,
                c: *mut $t,
                ldc: i64,
            ) -> rocblas_status {
                rocblas()?.$roc(
                    handle.rocm_handle,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    alpha.cast::<$rt>(),
                    a.cast::<$rt>(),
                    lda,
                    b.cast::<$rt>(),
                    ldb,
                    beta.cast::<$rt>(),
                    c.cast::<$rt>(),
                    ldc,
                )
            }
        )*
    };
}

gemm_64!(
    sgemm_v2_64 => rocblas_sgemm_64: f32 as f32,
    dgemm_v2_64 => rocblas_dgemm_64: f64 as f64,
    cgemm_v2_64 => rocblas_cgemm_64: cuComplex as rocblas_float_complex,
    zgemm_v2_64 => rocblas_zgemm_64: cuDoubleComplex as rocblas_double_complex
);

macro_rules! gemv_64 {
    ($($name:ident => $roc:ident: $t:ty as $rt:ty),*) => {
        $(
            pub(crate) unsafe fn $name(
                handle: &Handle,
                trans: rocblas_operation,
                m: i64,
                n: i64,
                alpha: *const $t,
                a: *const $t,
                lda: i64,
                x: *const $t,
                incx: i64,
                beta: *const $t,
                y: *mut $t,
                incy: i64,
            ) -> rocblas_status {
                rocblas()?.$roc(
                    handle.rocm_handle,
                    trans,
                    m,
                    n,
                    alpha.cast::<$rt>(),
                    a.cast::<$rt>(),
                    lda,
                    x.cast::<$rt>(),
                    incx,
                    beta.cast::<$rt>(),
                    y.cast::<$rt>(),
                    incy,
                )
            }
        )*
    };
}

gemv_64!(
    sgemv_v2_64 => rocblas_sgemv_64: f32 as f32,
    dgemv_v2_64 => rocblas_dgemv_64: f64 as f64,
    cgemv_v2_64 => rocblas_cgemv_64: cuComplex as rocblas_float_complex,
    zgemv_v2_64 => rocblas_zgemv_64: cuDoubleComplex as rocblas_double_complex
);

pub(crate) unsafe fn sgeam_64(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: i64,
    n: i64,
    alpha: *const f32,
    a: *const f32,
    lda: i64,
    beta: *const f32,
    b: *const f32,
    ldb: i64,
    c: *mut f32,
    ldc: i64,
) -> rocblas_status {
    let zero: f32 = std::mem::zeroed();
    let beta = geam_beta(handle, beta, &zero)?;
    rocblas()?.rocblas_sgeam_64(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        alpha.cast::<f32>(),
        a.cast::<f32>(),
        lda,
        beta.cast::<f32>(),
        b.cast::<f32>(),
        ldb,
        c.cast::<f32>(),
        ldc,
    )
}

pub(crate) unsafe fn dgeam_64(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: i64,
    n: i64,
    alpha: *const f64,
    a: *const f64,
    lda: i64,
    beta: *const f64,
    b: *const f64,
    ldb: i64,
    c: *mut f64,
    ldc: i64,
) -> rocblas_status {
    let zero: f64 = std::mem::zeroed();
    let beta = geam_beta(handle, beta, &zero)?;
    rocblas()?.rocblas_dgeam_64(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        alpha.cast::<f64>(),
        a.cast::<f64>(),
        lda,
        beta.cast::<f64>(),
        b.cast::<f64>(),
        ldb,
        c.cast::<f64>(),
        ldc,
    )
}

pub(crate) unsafe fn cgeam_64(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: i64,
    n: i64,
    alpha: *const cuComplex,
    a: *const cuComplex,
    lda: i64,
    beta: *const cuComplex,
    b: *const cuComplex,
    ldb: i64,
    c: *mut cuComplex,
    ldc: i64,
) -> rocblas_status {
    let zero: cuComplex = std::mem::zeroed();
    let beta = geam_beta(handle, beta, &zero)?;
    rocblas()?.rocblas_cgeam_64(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        alpha.cast::<rocblas_float_complex>(),
        a.cast::<rocblas_float_complex>(),
        lda,
        beta.cast::<rocblas_float_complex>(),
        b.cast::<rocblas_float_complex>(),
        ldb,
        c.cast::<rocblas_float_complex>(),
        ldc,
    )
}

pub(crate) unsafe fn zgeam_64(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: i64,
    n: i64,
    alpha: *const cuDoubleComplex,
    a: *const cuDoubleComplex,
    lda: i64,
    beta: *const cuDoubleComplex,
    b: *const cuDoubleComplex,
    ldb: i64,
    c: *mut cuDoubleComplex,
    ldc: i64,
) -> rocblas_status {
    let zero: cuDoubleComplex = std::mem::zeroed();
    let beta = geam_beta(handle, beta, &zero)?;
    rocblas()?.rocblas_zgeam_64(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        alpha.cast::<rocblas_double_complex>(),
        a.cast::<rocblas_double_complex>(),
        lda,
        beta.cast::<rocblas_double_complex>(),
        b.cast::<rocblas_double_complex>(),
        ldb,
        c.cast::<rocblas_double_complex>(),
        ldc,
    )
}

macro_rules! trsm_64 {
    ($($name:ident => $roc:ident: $t:ty as $rt:ty),*) => {
        $(
            pub(crate) unsafe fn $name(
                handle: &Handle,
                side: rocblas_side,
                uplo: rocblas_fill,
                trans: rocblas_operation,
                diag: rocblas_diagonal,
                m: i64,
                n: i64,
                alpha: *const $t,
                a: *const $t,
                lda: i64,
                b: *mut $t,
                ldb: i64,
            ) -> rocblas_status {
                rocblas()?.$roc(
                    handle.rocm_handle,
                    side,
                    uplo,
                    trans,
                    diag,
                    m,
                    n,
                    alpha.cast::<$rt>(),
                    a.cast::<$rt>(),
                    lda,
                    b.cast::<$rt>(),
                    ldb,
                )
            }
        )*
    };
}

trsm_64!(
    strsm_v2_64 => rocblas_strsm_64: f32 as f32,
    dtrsm_v2_64 => rocblas_dtrsm_64: f64 as f64,
    ctrsm_v2_64 => rocblas_ctrsm_64: cuComplex as rocblas_float_complex,
    ztrsm_v2_64 => rocblas_ztrsm_64: cuDoubleComplex as rocblas_double_complex
);

#[cfg(test)]
mod level12_tests {
    use cuda_types::cublas::*;
    use zluda_common::test::Runtime;

    struct Fixture {
        rt: Runtime,
        _ctx: *mut ::core::ffi::c_void,
        handle: cublasHandle_t,
    }

    impl Fixture {
        fn new() -> Self {
            let rt = Runtime::load_hip();
            let _ctx = rt.init_context();
            let mut handle = unsafe { std::mem::zeroed() };
            assert_eq!(unsafe { crate::cublasCreate_v2(&mut handle) }, cublasStatus_t::SUCCESS);
            Self { rt, _ctx, handle }
        }
        fn upload<T: Copy>(&self, host: &[T]) -> *mut T {
            let ptr = self.rt.allocate(std::mem::size_of_val(host).max(1));
            self.rt.copy_to_device(ptr, host);
            ptr.cast()
        }
        fn download<T: Copy>(&self, ptr: *mut T, len: usize) -> Vec<T> {
            let mut host = vec![unsafe { std::mem::zeroed::<T>() }; len];
            self.rt.copy_to_host(&mut host, ptr.cast());
            host
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = unsafe { crate::cublasDestroy_v2(self.handle) };
        }
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= 1e-9 * e.abs().max(1.0), "index {i}: {a} != {e}");
        }
    }

    // Column-major m x n matrix element
    fn at(a: &[f64], ld: usize, row: usize, col: usize) -> f64 {
        a[row + col * ld]
    }

    #[test]
    fn daxpy() {
        let f = Fixture::new();
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [10.0, 20.0, 30.0, 40.0];
        let (dx, dy) = (f.upload(&x), f.upload(&y));
        let alpha = 0.5f64;
        assert_eq!(
            unsafe { crate::cublasDaxpy_v2(f.handle, 4, &alpha, dx, 1, dy, 1) },
            cublasStatus_t::SUCCESS
        );
        assert_close(&f.download(dy, 4), &[10.5, 21.0, 31.5, 42.0]);
    }

    #[test]
    fn saxpy() {
        let f = Fixture::new();
        let (dx, dy) = (f.upload(&[1.0f32, -2.0]), f.upload(&[1.0f32, 1.0]));
        let alpha = 2.0f32;
        assert_eq!(
            unsafe { crate::cublasSaxpy_v2(f.handle, 2, &alpha, dx, 1, dy, 1) },
            cublasStatus_t::SUCCESS
        );
        assert_eq!(f.download(dy, 2), vec![3.0f32, -3.0]);
    }

    // C = alpha * op(A) + beta * op(B), with op(A) = A^T: this is how row-major data is
    // turned into column-major data (and back)
    #[test]
    fn dgeam_transpose() {
        let f = Fixture::new();
        let (m, n) = (3usize, 2usize);
        // A is n x m (column-major, lda = n), op(A) = A^T is m x n
        let a: Vec<f64> = (0..m * n).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..m * n).map(|i| 100.0 + i as f64).collect();
        let (da, db, dc) = (f.upload(&a), f.upload(&b), f.upload(&vec![0.0f64; m * n]));
        let (alpha, beta) = (2.0f64, 1.0f64);
        assert_eq!(
            unsafe {
                crate::cublasDgeam(
                    f.handle,
                    cublasOperation_t::CUBLAS_OP_T,
                    cublasOperation_t::CUBLAS_OP_N,
                    m as i32,
                    n as i32,
                    &alpha,
                    da,
                    n as i32,
                    &beta,
                    db,
                    m as i32,
                    dc,
                    m as i32,
                )
            },
            cublasStatus_t::SUCCESS
        );
        let c = f.download(dc, m * n);
        let mut expected = vec![0.0; m * n];
        for row in 0..m {
            for col in 0..n {
                expected[row + col * m] = alpha * at(&a, n, col, row) + beta * at(&b, m, row, col);
            }
        }
        assert_close(&c, &expected);
    }

    #[test]
    fn zgeam_conjugate_transpose() {
        let f = Fixture::new();
        let a = [
            cuDoubleComplex { x: 1.0, y: 2.0 },
            cuDoubleComplex { x: 3.0, y: 4.0 },
            cuDoubleComplex { x: 5.0, y: 6.0 },
            cuDoubleComplex { x: 7.0, y: 8.0 },
        ];
        let zero = cuDoubleComplex { x: 0.0, y: 0.0 };
        let one = cuDoubleComplex { x: 1.0, y: 0.0 };
        let (da, dc) = (f.upload(&a), f.upload(&[zero; 4]));
        assert_eq!(
            unsafe {
                crate::cublasZgeam(
                    f.handle,
                    cublasOperation_t::CUBLAS_OP_C,
                    cublasOperation_t::CUBLAS_OP_N,
                    2,
                    2,
                    &one,
                    da,
                    2,
                    &zero,
                    dc,
                    2,
                    dc,
                    2,
                )
            },
            cublasStatus_t::SUCCESS
        );
        let c = f.download(dc, 4);
        // conj(A^T): c[r + 2c] = conj(a[c + 2r])
        let expected = [(1.0, -2.0), (5.0, -6.0), (3.0, -4.0), (7.0, -8.0)];
        for (z, (x, y)) in c.iter().zip(expected) {
            assert_eq!((z.x, z.y), (x, y));
        }
    }

    #[test]
    fn dgemv() {
        let f = Fixture::new();
        // A = [[1, 2, 3], [4, 5, 6]] column-major, lda = 2
        let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = [1.0, 0.5, -1.0];
        let y = [1.0, 1.0];
        let (da, dx, dy) = (f.upload(&a), f.upload(&x), f.upload(&y));
        let (alpha, beta) = (1.0f64, 2.0f64);
        assert_eq!(
            unsafe {
                crate::cublasDgemv_v2(
                    f.handle,
                    cublasOperation_t::CUBLAS_OP_N,
                    2,
                    3,
                    &alpha,
                    da,
                    2,
                    dx,
                    1,
                    &beta,
                    dy,
                    1,
                )
            },
            cublasStatus_t::SUCCESS
        );
        assert_close(&f.download(dy, 2), &[1.0 + 1.0 - 3.0 + 2.0, 4.0 + 2.5 - 6.0 + 2.0]);
    }

    #[test]
    fn cgemv_and_cdotu() {
        let f = Fixture::new();
        let i = |x: f32, y: f32| cuComplex { x, y };
        // A = [[1+i, 2], [0, i]] column-major
        let a = [i(1.0, 1.0), i(0.0, 0.0), i(2.0, 0.0), i(0.0, 1.0)];
        let x = [i(1.0, 0.0), i(0.0, 1.0)];
        let (da, dx, dy) = (f.upload(&a), f.upload(&x), f.upload(&[i(0.0, 0.0); 2]));
        let (one, zero) = (i(1.0, 0.0), i(0.0, 0.0));
        assert_eq!(
            unsafe {
                crate::cublasCgemv_v2(
                    f.handle,
                    cublasOperation_t::CUBLAS_OP_N,
                    2,
                    2,
                    &one,
                    da,
                    2,
                    dx,
                    1,
                    &zero,
                    dy,
                    1,
                )
            },
            cublasStatus_t::SUCCESS
        );
        // y = [(1+i)*1 + 2*i, i*i] = [1+3i, -1]
        let y = f.download(dy, 2);
        assert_eq!((y[0].x, y[0].y, y[1].x, y[1].y), (1.0, 3.0, -1.0, 0.0));
        // dotu (no conjugation): x . x = 1 + i*i = 0
        let mut result = i(5.0, 5.0);
        assert_eq!(
            unsafe { crate::cublasCdotu_v2(f.handle, 2, dx, 1, dx, 1, &mut result) },
            cublasStatus_t::SUCCESS
        );
        assert_eq!((result.x, result.y), (0.0, 0.0));
    }

    #[test]
    fn zgemm() {
        let f = Fixture::new();
        let z = |x: f64, y: f64| cuDoubleComplex { x, y };
        // A = [[i, 0], [0, 2]], B = [[1, 1], [i, 0]] (column-major)
        let a = [z(0.0, 1.0), z(0.0, 0.0), z(0.0, 0.0), z(2.0, 0.0)];
        let b = [z(1.0, 0.0), z(0.0, 1.0), z(1.0, 0.0), z(0.0, 0.0)];
        let (da, db, dc) = (f.upload(&a), f.upload(&b), f.upload(&[z(0.0, 0.0); 4]));
        let (one, zero) = (z(1.0, 0.0), z(0.0, 0.0));
        assert_eq!(
            unsafe {
                crate::cublasZgemm_v2(
                    f.handle,
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    2,
                    2,
                    2,
                    &one,
                    da,
                    2,
                    db,
                    2,
                    &zero,
                    dc,
                    2,
                )
            },
            cublasStatus_t::SUCCESS
        );
        // A.B = [[i, i], [2i, 0]]
        let c = f.download(dc, 4);
        let expected = [(0.0, 1.0), (0.0, 2.0), (0.0, 1.0), (0.0, 0.0)];
        for (v, (x, y)) in c.iter().zip(expected) {
            assert_eq!((v.x, v.y), (x, y));
        }
    }

    #[test]
    fn dgemm_64() {
        let f = Fixture::new();
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] (column-major)
        let (da, db, dc) = (f.upload(&[1.0, 3.0, 2.0, 4.0]), f.upload(&[5.0, 7.0, 6.0, 8.0]), f.upload(&[1.0f64; 4]));
        let (alpha, beta) = (1.0f64, 1.0f64);
        assert_eq!(
            unsafe {
                crate::cublasDgemm_v2_64(
                    f.handle,
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    2,
                    2,
                    2,
                    &alpha,
                    da,
                    2,
                    db,
                    2,
                    &beta,
                    dc,
                    2,
                )
            },
            cublasStatus_t::SUCCESS
        );
        // A.B + 1 = [[20, 23], [44, 51]]
        assert_close(&f.download(dc, 4), &[20.0, 44.0, 23.0, 51.0]);
    }

    #[test]
    fn zgemv_64() {
        let f = Fixture::new();
        let z = |x: f64, y: f64| cuDoubleComplex { x, y };
        // A = [[1, i], [0, 2]] column-major, x = [1, 1]
        let a = [z(1.0, 0.0), z(0.0, 0.0), z(0.0, 1.0), z(2.0, 0.0)];
        let (da, dx, dy) = (f.upload(&a), f.upload(&[z(1.0, 0.0); 2]), f.upload(&[z(0.0, 0.0); 2]));
        let (one, zero) = (z(1.0, 0.0), z(0.0, 0.0));
        assert_eq!(
            unsafe {
                crate::cublasZgemv_v2_64(f.handle, cublasOperation_t::CUBLAS_OP_N, 2, 2, &one, da, 2, dx, 1, &zero, dy, 1)
            },
            cublasStatus_t::SUCCESS
        );
        let y = f.download(dy, 2);
        assert_eq!([(y[0].x, y[0].y), (y[1].x, y[1].y)], [(1.0, 1.0), (2.0, 0.0)]);
    }

    #[test]
    fn dtrsm_64() {
        let f = Fixture::new();
        // Solve L.X = B with L = [[2, 0], [1, 4]] (lower), B = [2, 9]
        let (dl, db) = (f.upload(&[2.0, 1.0, 0.0, 4.0]), f.upload(&[2.0f64, 9.0]));
        let alpha = 1.0f64;
        assert_eq!(
            unsafe {
                crate::cublasDtrsm_v2_64(
                    f.handle,
                    cublasSideMode_t::CUBLAS_SIDE_LEFT,
                    cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
                    2,
                    1,
                    &alpha,
                    dl,
                    2,
                    db,
                    2,
                )
            },
            cublasStatus_t::SUCCESS
        );
        assert_close(&f.download(db, 2), &[1.0, 2.0]);
    }

    #[test]
    fn idamax_and_dscal() {
        let f = Fixture::new();
        let dx = f.upload(&[1.0f64, -7.0, 3.0]);
        let mut index = 0i32;
        assert_eq!(unsafe { crate::cublasIdamax_v2(f.handle, 3, dx, 1, &mut index) }, cublasStatus_t::SUCCESS);
        // 1-based, like BLAS
        assert_eq!(index, 2);
        let alpha = -0.5f64;
        assert_eq!(unsafe { crate::cublasDscal_v2_64(f.handle, 3, &alpha, dx, 1) }, cublasStatus_t::SUCCESS);
        assert_close(&f.download(dx, 3), &[-0.5, 3.5, -1.5]);
    }

    #[test]
    fn dger_64_and_dswap() {
        let f = Fixture::new();
        let (dx, dy, da) = (f.upload(&[1.0f64, 2.0]), f.upload(&[3.0f64, 4.0]), f.upload(&[1.0f64; 4]));
        let alpha = 1.0f64;
        assert_eq!(unsafe { crate::cublasDger_v2_64(f.handle, 2, 2, &alpha, dx, 1, dy, 1, da, 2) }, cublasStatus_t::SUCCESS);
        // A + x.y^T, column-major
        assert_close(&f.download(da, 4), &[4.0, 7.0, 5.0, 9.0]);
        assert_eq!(unsafe { crate::cublasDswap_v2(f.handle, 2, dx, 1, dy, 1) }, cublasStatus_t::SUCCESS);
        assert_close(&f.download(dx, 2), &[3.0, 4.0]);
        assert_close(&f.download(dy, 2), &[1.0, 2.0]);
    }

    #[test]
    fn dznrm2_and_zgerc() {
        let f = Fixture::new();
        let z = |x: f64, y: f64| cuDoubleComplex { x, y };
        let dx = f.upload(&[z(3.0, 4.0), z(0.0, 0.0)]);
        let mut norm = 0.0f64;
        assert_eq!(unsafe { crate::cublasDznrm2_v2(f.handle, 2, dx, 1, &mut norm) }, cublasStatus_t::SUCCESS);
        assert!((norm - 5.0).abs() < 1e-12);
        // A = 0 + x.conj(y)^T with x = [3+4i, 0], y = [i]
        let (dy, da) = (f.upload(&[z(0.0, 1.0)]), f.upload(&[z(0.0, 0.0); 2]));
        let one = z(1.0, 0.0);
        assert_eq!(unsafe { crate::cublasZgerc_v2(f.handle, 2, 1, &one, dx, 1, dy, 1, da, 2) }, cublasStatus_t::SUCCESS);
        let a = f.download(da, 2);
        // (3+4i) * conj(i) = (3+4i) * -i = 4-3i
        assert_eq!([(a[0].x, a[0].y), (a[1].x, a[1].y)], [(4.0, -3.0), (0.0, 0.0)]);
    }

    #[test]
    fn dgeam_64_transpose() {
        let f = Fixture::new();
        // C = A^T + 2 B with A = [[1, 2], [3, 4]], B = I (column-major)
        let (da, db, dc) = (f.upload(&[1.0f64, 3.0, 2.0, 4.0]), f.upload(&[1.0f64, 0.0, 0.0, 1.0]), f.upload(&[0.0f64; 4]));
        let (alpha, beta) = (1.0f64, 2.0f64);
        assert_eq!(
            unsafe {
                crate::cublasDgeam_64(
                    f.handle,
                    cublasOperation_t::CUBLAS_OP_T,
                    cublasOperation_t::CUBLAS_OP_N,
                    2,
                    2,
                    &alpha,
                    da,
                    2,
                    &beta,
                    db,
                    2,
                    dc,
                    2,
                )
            },
            cublasStatus_t::SUCCESS
        );
        // A^T = [[1, 3], [2, 4]] -> column-major [1, 2, 3, 4]
        assert_close(&f.download(dc, 4), &[3.0, 2.0, 3.0, 6.0]);
    }

    #[test]
    fn dgeam_null_beta_transposes() {
        let f = Fixture::new();
        let (da, dc) = (f.upload(&[1.0f64, 3.0, 2.0, 4.0]), f.upload(&[0.0f64; 4]));
        let alpha = 1.0f64;
        assert_eq!(
            unsafe {
                crate::cublasDgeam_64(
                    f.handle,
                    cublasOperation_t::CUBLAS_OP_T,
                    cublasOperation_t::CUBLAS_OP_T,
                    2,
                    2,
                    &alpha,
                    da,
                    2,
                    std::ptr::null(),
                    da,
                    2,
                    dc,
                    2,
                )
            },
            cublasStatus_t::SUCCESS
        );
        assert_close(&f.download(dc, 4), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn drot_64_and_drotg() {
        let f = Fixture::new();
        let (mut a, mut b, mut c, mut s) = (3.0f64, 4.0f64, 0.0f64, 0.0f64);
        assert_eq!(unsafe { crate::cublasDrotg_v2(f.handle, &mut a, &mut b, &mut c, &mut s) }, cublasStatus_t::SUCCESS);
        assert!((a - 5.0).abs() < 1e-12 && (c - 0.6).abs() < 1e-12 && (s - 0.8).abs() < 1e-12);
        // x' = c x + s y, y' = c y - s x
        let (dx, dy) = (f.upload(&[3.0f64, 1.0]), f.upload(&[4.0f64, 0.0]));
        assert_eq!(unsafe { crate::cublasDrot_v2_64(f.handle, 2, dx, 1, dy, 1, &c, &s) }, cublasStatus_t::SUCCESS);
        assert_close(&f.download(dx, 2), &[5.0, 0.6]);
        assert_close(&f.download(dy, 2), &[0.0, -0.8]);
    }
}
