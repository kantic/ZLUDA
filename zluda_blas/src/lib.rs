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
            cublasCaxpy_v2,
            cublasCaxpy_v2_64,
            cublasCcopy_v2,
            cublasCcopy_v2_64,
            cublasCdotc_v2,
            cublasCdotc_v2_64,
            cublasCdotu_v2,
            cublasCdotu_v2_64,
            cublasCgeam,
            cublasCgeam_64,
            cublasCgemm_v2,
            cublasCgemm_v2_64,
            cublasCgemv_v2,
            cublasCgemv_v2_64,
            cublasCgerc_v2,
            cublasCgerc_v2_64,
            cublasCgeru_v2,
            cublasCgeru_v2_64,
            cublasCreate_v2,
            cublasCrot_v2,
            cublasCrot_v2_64,
            cublasCrotg_v2,
            cublasCscal_v2,
            cublasCscal_v2_64,
            cublasCsrot_v2,
            cublasCsrot_v2_64,
            cublasCsscal_v2,
            cublasCsscal_v2_64,
            cublasCswap_v2,
            cublasCswap_v2_64,
            cublasCtrmm_v2,
            cublasCtrmm_v2_64,
            cublasCtrsm_v2_64,
            cublasDasum_v2,
            cublasDasum_v2_64,
            cublasDaxpy_v2,
            cublasDaxpy_v2_64,
            cublasDcopy_v2,
            cublasDcopy_v2_64,
            cublasDdot_v2,
            cublasDdot_v2_64,
            cublasDestroy_v2,
            cublasDgeam,
            cublasDgeam_64,
            cublasDgemmStridedBatched,
            cublasDgemm_v2,
            cublasDgemm_v2_64,
            cublasDgemv_v2,
            cublasDgemv_v2_64,
            cublasDger_v2,
            cublasDger_v2_64,
            cublasDnrm2_v2,
            cublasDnrm2_v2_64,
            cublasDrot_v2,
            cublasDrot_v2_64,
            cublasDrotg_v2,
            cublasDrotm_v2,
            cublasDrotm_v2_64,
            cublasDrotmg_v2,
            cublasDscal_v2,
            cublasDscal_v2_64,
            cublasDswap_v2,
            cublasDswap_v2_64,
            cublasDsyrk_v2,
            cublasDtrmm_v2,
            cublasDtrmm_v2_64,
            cublasDtrsm_v2,
            cublasDtrsm_v2_64,
            cublasDzasum_v2,
            cublasDzasum_v2_64,
            cublasDznrm2_v2,
            cublasDznrm2_v2_64,
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
            cublasIcamax_v2,
            cublasIcamax_v2_64,
            cublasIcamin_v2,
            cublasIcamin_v2_64,
            cublasIdamax_v2,
            cublasIdamax_v2_64,
            cublasIdamin_v2,
            cublasIdamin_v2_64,
            cublasIsamax_v2,
            cublasIsamax_v2_64,
            cublasIsamin_v2,
            cublasIsamin_v2_64,
            cublasIzamax_v2,
            cublasIzamax_v2_64,
            cublasIzamin_v2,
            cublasIzamin_v2_64,
            cublasSasum_v2,
            cublasSasum_v2_64,
            cublasSaxpy_v2,
            cublasSaxpy_v2_64,
            cublasScasum_v2,
            cublasScasum_v2_64,
            cublasScnrm2_v2,
            cublasScnrm2_v2_64,
            cublasScopy_v2,
            cublasScopy_v2_64,
            cublasSdot_v2,
            cublasSdot_v2_64,
            cublasSetAtomicsMode,
            cublasSetMathMode,
            cublasSetPointerMode_v2,
            cublasSetStream_v2,
            cublasSetVector,
            cublasSetWorkspace_v2,
            cublasSgeam,
            cublasSgeam_64,
            cublasSgemmEx,
            cublasSgemmStridedBatched,
            cublasSgemm_v2,
            cublasSgemm_v2_64,
            cublasSgemv_v2,
            cublasSgemv_v2_64,
            cublasSger_v2,
            cublasSger_v2_64,
            cublasSnrm2_v2,
            cublasSnrm2_v2_64,
            cublasSrot_v2,
            cublasSrot_v2_64,
            cublasSrotg_v2,
            cublasSrotm_v2,
            cublasSrotm_v2_64,
            cublasSrotmg_v2,
            cublasSscal_v2,
            cublasSscal_v2_64,
            cublasSswap_v2,
            cublasSswap_v2_64,
            cublasSsyrk_v2,
            cublasStrmm_v2,
            cublasStrmm_v2_64,
            cublasStrsm_v2,
            cublasStrsm_v2_64,
            cublasZaxpy_v2,
            cublasZaxpy_v2_64,
            cublasZcopy_v2,
            cublasZcopy_v2_64,
            cublasZdotc_v2,
            cublasZdotc_v2_64,
            cublasZdotu_v2,
            cublasZdotu_v2_64,
            cublasZdrot_v2,
            cublasZdrot_v2_64,
            cublasZdscal_v2,
            cublasZdscal_v2_64,
            cublasZgeam,
            cublasZgeam_64,
            cublasZgemm_v2,
            cublasZgemm_v2_64,
            cublasZgemv_v2,
            cublasZgemv_v2_64,
            cublasZgerc_v2,
            cublasZgerc_v2_64,
            cublasZgeru_v2,
            cublasZgeru_v2_64,
            cublasZrot_v2,
            cublasZrot_v2_64,
            cublasZrotg_v2,
            cublasZscal_v2,
            cublasZscal_v2_64,
            cublasZswap_v2,
            cublasZswap_v2_64,
            cublasZtrmm_v2,
            cublasZtrmm_v2_64,
            cublasZtrsm_v2_64,
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
            rocblas_caxpy,
            rocblas_caxpy_64,
            rocblas_ccopy,
            rocblas_ccopy_64,
            rocblas_cdotc,
            rocblas_cdotc_64,
            rocblas_cdotu,
            rocblas_cdotu_64,
            rocblas_cgeam,
            rocblas_cgeam_64,
            rocblas_cgemm,
            rocblas_cgemm_64,
            rocblas_cgemv,
            rocblas_cgemv_64,
            rocblas_cgerc,
            rocblas_cgerc_64,
            rocblas_cgeru,
            rocblas_cgeru_64,
            rocblas_create_handle,
            rocblas_crot,
            rocblas_crot_64,
            rocblas_crotg,
            rocblas_cscal,
            rocblas_cscal_64,
            rocblas_csrot,
            rocblas_csrot_64,
            rocblas_csscal,
            rocblas_csscal_64,
            rocblas_cswap,
            rocblas_cswap_64,
            rocblas_ctrmm,
            rocblas_ctrmm_64,
            rocblas_ctrsm_64,
            rocblas_dasum,
            rocblas_dasum_64,
            rocblas_daxpy,
            rocblas_daxpy_64,
            rocblas_dcopy,
            rocblas_dcopy_64,
            rocblas_ddot,
            rocblas_ddot_64,
            rocblas_destroy_handle,
            rocblas_dgeam,
            rocblas_dgeam_64,
            rocblas_dgemm,
            rocblas_dgemm_64,
            rocblas_dgemm_strided_batched,
            rocblas_dgemv,
            rocblas_dgemv_64,
            rocblas_dger,
            rocblas_dger_64,
            rocblas_dnrm2,
            rocblas_dnrm2_64,
            rocblas_drot,
            rocblas_drot_64,
            rocblas_drotg,
            rocblas_drotm,
            rocblas_drotm_64,
            rocblas_drotmg,
            rocblas_dscal,
            rocblas_dscal_64,
            rocblas_dswap,
            rocblas_dswap_64,
            rocblas_dsyrk,
            rocblas_dtrmm,
            rocblas_dtrmm_64,
            rocblas_dtrsm,
            rocblas_dtrsm_64,
            rocblas_dzasum,
            rocblas_dzasum_64,
            rocblas_dznrm2,
            rocblas_dznrm2_64,
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
            rocblas_icamax,
            rocblas_icamax_64,
            rocblas_icamin,
            rocblas_icamin_64,
            rocblas_idamax,
            rocblas_idamax_64,
            rocblas_idamin,
            rocblas_idamin_64,
            rocblas_isamax,
            rocblas_isamax_64,
            rocblas_isamin,
            rocblas_isamin_64,
            rocblas_izamax,
            rocblas_izamax_64,
            rocblas_izamin,
            rocblas_izamin_64,
            rocblas_sasum,
            rocblas_sasum_64,
            rocblas_saxpy,
            rocblas_saxpy_64,
            rocblas_scasum,
            rocblas_scasum_64,
            rocblas_scnrm2,
            rocblas_scnrm2_64,
            rocblas_scopy,
            rocblas_scopy_64,
            rocblas_sdot,
            rocblas_sdot_64,
            rocblas_set_atomics_mode,
            rocblas_set_math_mode,
            rocblas_set_pointer_mode,
            rocblas_set_stream,
            rocblas_set_vector,
            rocblas_set_workspace,
            rocblas_sgeam,
            rocblas_sgeam_64,
            rocblas_sgemm,
            rocblas_sgemm_64,
            rocblas_sgemm_strided_batched,
            rocblas_sgemv,
            rocblas_sgemv_64,
            rocblas_sger,
            rocblas_sger_64,
            rocblas_snrm2,
            rocblas_snrm2_64,
            rocblas_srot,
            rocblas_srot_64,
            rocblas_srotg,
            rocblas_srotm,
            rocblas_srotm_64,
            rocblas_srotmg,
            rocblas_sscal,
            rocblas_sscal_64,
            rocblas_sswap,
            rocblas_sswap_64,
            rocblas_ssyrk,
            rocblas_strmm,
            rocblas_strmm_64,
            rocblas_strsm,
            rocblas_strsm_64,
            rocblas_zaxpy,
            rocblas_zaxpy_64,
            rocblas_zcopy,
            rocblas_zcopy_64,
            rocblas_zdotc,
            rocblas_zdotc_64,
            rocblas_zdotu,
            rocblas_zdotu_64,
            rocblas_zdrot,
            rocblas_zdrot_64,
            rocblas_zdscal,
            rocblas_zdscal_64,
            rocblas_zgeam,
            rocblas_zgeam_64,
            rocblas_zgemm,
            rocblas_zgemm_64,
            rocblas_zgemv,
            rocblas_zgemv_64,
            rocblas_zgerc,
            rocblas_zgerc_64,
            rocblas_zgeru,
            rocblas_zgeru_64,
            rocblas_zrot,
            rocblas_zrot_64,
            rocblas_zrotg,
            rocblas_zscal,
            rocblas_zscal_64,
            rocblas_zswap,
            rocblas_zswap_64,
            rocblas_ztrmm,
            rocblas_ztrmm_64,
            rocblas_ztrsm_64,
        ]
);

#[cfg(test)]
mod tests;
