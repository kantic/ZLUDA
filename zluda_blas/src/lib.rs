use cuda_types::cublas::cublasError_t;

mod r#impl;

macro_rules! unimplemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            #[allow(unused_variables)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                crate::r#impl::unimplemented()
            }
        )*
    };
}

macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_macros::cublas_normalize_fn!( crate::r#impl::$fn_name ) ($(zluda_common::FromCuda::<_, cublasError_t>::from_cuda(&$arg_id)?),*)?;
                Ok(())
            }
        )*
    };
}

macro_rules! implemented_and_always_succeeds {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_macros::cublas_normalize_fn!( crate::r#impl::$fn_name ) ( $( $arg_id ),* )
            }
        )*
    };
}

cuda_macros::cublas_function_declarations!(
    unimplemented,
    implemented
        <= [
            cublasCreate_v2,
            cublasDdot_v2,
            cublasDestroy_v2,
            cublasDgemmStridedBatched,
            cublasDgemm_v2,
            cublasDsyrk_v2,
            cublasDtrmm_v2,
            cublasDtrsm_v2,
            cublasGemmBatchedEx,
            cublasGemmEx,
            cublasGemmStridedBatchedEx,
            cublasGetAtomicsMode,
            cublasGetMathMode,
            cublasGetPointerMode_v2,
            cublasGetProperty,
            cublasGetStream_v2,
            cublasGetVector,
            cublasGetVersion_v2,
            cublasHgemm,
            cublasHgemmStridedBatched,
            cublasSdot_v2,
            cublasSetAtomicsMode,
            cublasSetMathMode,
            cublasSetPointerMode_v2,
            cublasSetStream_v2,
            cublasSetVector,
            cublasSetWorkspace_v2,
            cublasSgemmEx,
            cublasSgemmStridedBatched,
            cublasSgemm_v2,
            cublasSsyrk_v2,
            cublasStrmm_v2,
            cublasStrsm_v2,
        ],
    implemented_and_always_succeeds
        <= [
            cublasGetStatusName,
            cublasGetStatusString,
            cublasXerbla,
            cublasGetCudartVersion
        ]
);

macro_rules! noop {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {};
}

#[cfg(windows)]
mod os_macro {
    macro_rules! vtable_impl {
        ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
            use rocblas_sys::*;
            struct RocblasVtable {
                _lib: libloading::os::windows::Library,
                $($fn_name: unsafe extern "C" fn($($arg_id: $arg_type),*) -> $ret_type,)*
            }

            impl RocblasVtable {
                pub unsafe fn new() -> Result<Self, rocblas_error> {
                    let hmodule = zluda_windows::try_load_from_self_or_hip_with_message(&["rocblas.dll"]).ok_or(rocblas_error::internal_error)?;
                    let lib = libloading::os::windows::Library::from_raw(hmodule.0 as _);
                    $(
                        let $fn_name = *lib.get::<unsafe extern "C" fn($($arg_id: $arg_type),*) -> $ret_type>(concat!(stringify!($fn_name), "\0").as_bytes()).map_err(|_| rocblas_error::internal_error)?;
                    )*
                    Ok(Self {
                        _lib: lib,
                        $($fn_name,)*
                    })
                }

                $(
                    pub unsafe fn $fn_name(&self, $($arg_id: $arg_type),*) -> $ret_type {
                        (self.$fn_name)($($arg_id),*)
                    }
                )*
            }
        };
    }
    pub(crate) use vtable_impl;
}

#[cfg(not(windows))]
mod os_macro {
    macro_rules! vtable_impl {
        ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
            use rocblas_sys::*;

            struct RocblasVtable {}

            impl RocblasVtable {
                pub unsafe fn new() -> Result<Self, rocblas_error> {
                    Ok(Self {})
                }
            }

            impl RocblasVtable {
                $(
                    pub unsafe fn $fn_name(&self, $($arg_id: $arg_type),*) -> $ret_type {
                        (rocblas_sys::$fn_name)($($arg_id),*)
                    }
                )*
            }
        };
    }
    pub(crate) use vtable_impl;
}

cuda_macros::rocblas_function_declarations!(
    noop,
    os_macro::vtable_impl
        <= [
            rocblas_create_handle,
            rocblas_ddot,
            rocblas_destroy_handle,
            rocblas_dgemm,
            rocblas_dgemm_strided_batched,
            rocblas_dsyrk,
            rocblas_dtrmm,
            rocblas_dtrsm,
            rocblas_gemm_batched_ex,
            rocblas_gemm_ex,
            rocblas_gemm_strided_batched_ex,
            rocblas_get_atomics_mode,
            rocblas_get_math_mode,
            rocblas_get_pointer_mode,
            rocblas_get_stream,
            rocblas_get_vector,
            rocblas_hgemm,
            rocblas_hgemm_strided_batched,
            rocblas_sdot,
            rocblas_set_atomics_mode,
            rocblas_set_math_mode,
            rocblas_set_pointer_mode,
            rocblas_set_stream,
            rocblas_set_vector,
            rocblas_set_workspace,
            rocblas_sgemm,
            rocblas_sgemm_strided_batched,
            rocblas_ssyrk,
            rocblas_strmm,
            rocblas_strsm
        ]
);

#[cfg(test)]
mod tests;
