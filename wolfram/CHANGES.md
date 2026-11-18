# Changes compared to upstream ZLUDA

This fork is based on upstream ZLUDA commit `eb89ae3` ("Add an executable picker to the Windows
launcher (#676)"). All changes were made to run the GPU functionality of the Wolfram Language;
see [README.md](README.md) for how to use it. Most of them are general fixes and additions that
other applications benefit from too.

| Commit | Summary |
|---|---|
| `24dafa6` | Neural network training (`NetTrain`) on the GPU |
| `af95699` | Convolutional networks: cuDNN backend API, batch normalization, pooling, ... |
| `9aa88b2` | Recurrent networks: cuDNN RNN API |
| `32f816d` | GPU computing: GPUArray, CUDALink, `FunctionCompile`; comgr shim |
| `066f97b` | cuBLAS 64-bit trmm; `cublasSetWorkspace` fix (large linear solves and SVDs) |
| `188d1a8` | Complex cuBLAS trmm (large complex SVDs) |

## New crates

### `zluda_rtc` (`libnvrtc.so`)

An NVRTC shim. NVRTC compiles CUDA C++ to PTX or to NVIDIA machine code (CUBIN), but ZLUDA can
only use PTX. The shim forwards everything to the real NVIDIA NVRTC (`ZLUDA_NVRTC_LIB`, or the
usual CUDA installation paths), except that:

* real architectures (`sm_XX`) are replaced by virtual ones (`compute_XX`), clamped to an
  architecture the NVRTC supports, so NVRTC always produces PTX
* `nvrtcGetCUBINSize`/`nvrtcGetCUBIN` return the PTX. ZLUDA's `cuModuleLoadData` accepts PTX, so
  applications that compile to CUBIN and load it with the driver API (MXNet, PyTorch) work
  unmodified

### `zluda_comgr` (`libzluda_comgr.so`, soname `libamd_comgr.so.3`)

Loads the ROCm code object manager (comgr) into its own link-map namespace with `dlmopen` and
forwards every comgr function to it. comgr is linked against ROCm's LLVM, whose symbols
interpose with those of any other LLVM in the process; GNU unique symbols are even shared across
symbol versions and `RTLD_DEEPBIND`. In the Wolfram Language, which has its own LLVM in its
compiler, `FunctionCompile` failed with "Cannot verify module" once HIP was loaded, and HIP hung
if the compiler was used first.

The shim must be loaded before HIP. The forwarding functions also copy the main `environ` into
the namespace's libc, whose own copy becomes stale after `setenv` and crashed `getenv` in comgr.
The real comgr is found via `ZLUDA_COMGR_LIB`, `LD_LIBRARY_PATH` or `$ROCM_PATH/lib`.

## Driver (`zluda`)

* Implemented `cuMemcpy2DAsync_v2`.
* Implemented the legacy launch API that CUDALink uses: `cuParamSetSize`, `cuParamSeti`,
  `cuParamSetf`, `cuParamSetv`, `cuFuncSetBlockShape`, `cuFuncSetSharedSize`, `cuLaunch`,
  `cuLaunchGrid`, `cuLaunchGridAsync`.
* `cuCtxSetCurrent` replaces the top of the context stack instead of pushing onto it.
* The per-thread context stack has no destructor, so CUDA calls from static destructors at
  process exit work. glibc runs thread-local destructors of the main thread before `atexit`
  handlers and C++ static destructors, and MXNet frees its CUDA resources in those.
* `cuCtxDestroy` on a primary context returns `CUDA_ERROR_INVALID_CONTEXT` instead of freeing
  it. CUDALink destroys the context it finds current, which crashed with
  `free(): invalid size`.
* The driver and the trace driver have the soname `libcuda.so.1`. Wolfram's MXNet has a
  `DT_RPATH` containing a stub `libcuda.so.1`, which takes precedence over `LD_LIBRARY_PATH`
  unless a library with that soname is already loaded.
* PTX images without a terminating NUL: junk after the PTX text is dropped (`zluda_common`).

## PTX compiler (`ptx_parser`)

* `prefetch` with a memory operand failed with `TranslateError::MismatchedType`. Fixed, with a
  new test (`ptx/src/test/spirv_run/prefetch.ptx`).

## cuBLAS (`zluda_blas`)

Newly implemented on rocBLAS, most of them also as the 64-bit `_64` variants, which cuSOLVER
uses:

* **Handle**: `cublasGetAtomicsMode`/`cublasSetAtomicsMode`, `cublasGetPointerMode_v2`,
  `cublasGetStream_v2`, `cublasGetVersion_v2`, `cublasGetProperty`.
* **Level 1** (S/D/C/Z and mixed variants such as `Scasum`, `Dznrm2`, `Csscal`, `Zdrot`;
  32-bit and `_64`): `I?amax`, `I?amin`, `asum`, `axpy`, `copy`, `dot` (S/D), `dotu`/`dotc`
  (C/Z), `nrm2`, `rot`, `scal`, `swap`, `rotm` (S/D); 32-bit only: `rotg`, `rotmg` (S/D).
* **Level 2** (32-bit and `_64`): `gemv` (S/D/C/Z), `ger` (S/D), `geru`/`gerc` (C/Z).
* **Level 3**: `gemm` (D/C/Z, and `_64` for S/D/C/Z), `SgemmEx`, `DgemmStridedBatched`,
  `HgemmStridedBatched`, `syrk` (S/D), `trmm` (S/D/C/Z, 32-bit and `_64`), `trsm` (S/D 32-bit,
  S/D/C/Z `_64`).
* **Extensions**: `geam` (S/D/C/Z, 32-bit and `_64`). Accepts a NULL `beta` like cuBLAS does.

`cublasSetWorkspace_v2` no longer passes the workspace to `rocblas_set_workspace`. cuBLAS uses
the user workspace only as scratch space, but `rocblas_set_workspace` turns off rocBLAS-managed
memory, so every operation that needs more than the workspace failed with
`rocblas_status_memory_error`. cuSOLVER sets a 4 MiB workspace and rocBLAS trsm copies the whole
right-hand side into it, so e.g. LU solves failed from n ≈ 724.

## cuDNN (`zluda_dnn`, used by the cuDNN 8 and 9 libraries)

Newly implemented on MIOpen:

* **Backend (graph) API**, used by MXNet for convolutions: `cudnnBackendCreateDescriptor`,
  `cudnnBackendDestroyDescriptor`, `cudnnBackendInitialize`, `cudnnBackendSetAttribute`,
  `cudnnBackendGetAttribute`, `cudnnBackendFinalize`, `cudnnBackendExecute`. Supports graphs with
  a single convolution operation (forward, backward data, backward filter). Each graph has one
  engine without knobs, which runs the fastest MIOpen solution (`zluda_dnn/src/impl/backend.rs`).
* **Batch normalization**: `cudnnDeriveBNTensorDescriptor`,
  `cudnnBatchNormalizationForwardTraining(Ex)`, `cudnnBatchNormalizationForwardInference`,
  `cudnnBatchNormalizationBackward(Ex)` and the `Ex` workspace/reserve size functions.
* **Activation**: `cudnnCreate/Set/DestroyActivationDescriptor`, `cudnnActivationForward`,
  `cudnnActivationBackward`.
* **Pooling**: `cudnnCreate/DestroyPoolingDescriptor`, `cudnnSetPooling2dDescriptor`,
  `cudnnSetPoolingNdDescriptor`, `cudnnPoolingForward`, `cudnnPoolingBackward`. cuDNN's pooling
  backward has no workspace argument, but MIOpen needs the index workspace of the forward pass,
  so the forward pass is run again into a temporary buffer.
* **Softmax**: `cudnnSoftmaxForward`, `cudnnSoftmaxBackward`.
* **RNN** (legacy API, on MIOpen's sequence tensor RNN API, `zluda_dnn/src/impl/rnn.rs`):
  `cudnnCreate/DestroyRNNDescriptor`, `cudnnSetRNNDescriptor_v6`, `cudnnSetRNNMatrixMathType`,
  `cudnnSetRNNPaddingMode`, `cudnnSetRNNProjectionLayers`, `cudnnRNNSetClip`,
  `cudnnCreate/Set/DestroyRNNDataDescriptor`, `cudnnGetRNNParamsSize`,
  `cudnnGetRNNWorkspaceSize`, `cudnnGetRNNTrainingReserveSize`, `cudnnRNNForwardTrainingEx`,
  `cudnnRNNForwardInferenceEx`, `cudnnRNNBackwardDataEx`, `cudnnRNNBackwardWeightsEx`. LSTM, GRU
  and basic RNNs, unidirectional and bidirectional, stacked. Weights are converted between the
  cuDNN and MIOpen layouts (different gate order) in a workspace prefix, weight gradients are
  accumulated like cuDNN does, and the padding of output sequences is zeroed.
* Dropout descriptors without RNG states are accepted for a dropout of 0 (used by MXNet), and
  NCHW filter descriptors with other than 4 or 5 dimensions (used for RNN weights).

`cudnnAddTensor` and `cudnnConvolutionBackwardBias` intentionally stay unsupported: MXNet treats
`NOT_SUPPORTED` as the signal to use its own bias kernels.

## Tests

New unit tests cover:

* the new cuBLAS functions and the workspace fix (`zluda_blas/src/impl.rs`)
* the cuDNN backend API, including GPU execution of all three convolution directions against a
  CPU reference (`zluda_dnn/src/impl/backend.rs`)
* the RNN functions, against a CPU reference for the forward pass and numerical gradients for
  the backward pass, for LSTM, GRU and basic RNNs, stacked, bidirectional and padded
  (`zluda_dnn/src/impl/rnn.rs`)
* the comgr shim (`zluda_comgr`), the legacy launch API, the context changes, PTX image trimming
  and the `prefetch` fix

## Wolfram-specific files

These files aren't part of ZLUDA itself:

* `wolfram/ZLUDAInit.wl`: Wolfram Language integration, see [README.md](README.md)
* `wolfram/BuildGPUArrayLibraries.wls`: rebuilds the GPUArray libraries with PTX
* `wolfram/README.md`, `wolfram/CHANGES.md`: this documentation
* `AGENTS.md`: build and run instructions for coding agents working on this fork
