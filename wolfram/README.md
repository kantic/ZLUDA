# ZLUDA for the Wolfram Language

This fork of [ZLUDA](https://github.com/vosen/ZLUDA) makes the GPU functionality of the Wolfram
Language run on AMD GPUs: training and evaluating neural networks with `TargetDevice -> "GPU"`,
and GPU computing (`GPUArray`, CUDALink, `FunctionCompile` with GPU types).

This directory contains everything Wolfram-specific: the integration package `ZLUDAInit.wl`, the
build script `BuildGPUArrayLibraries.wls`, this guide, and the list of changes to ZLUDA in
[CHANGES.md](CHANGES.md).

Tested with Wolfram Language 15.0 on Linux, ROCm 10.1.0 and an AMD Radeon 890M (gfx1150).
Other platforms haven't been tested.

## How it works

ZLUDA provides `libcuda.so.1` and replacements for the CUDA libraries (cuBLAS, cuDNN, cuFFT,
cuSPARSE) on top of HIP, rocBLAS, MIOpen, etc. The Wolfram Language doesn't find these by itself:

* the neural network runtime (MXNet) and CUDALink load the CUDA libraries by absolute path from
  the `WolframGPUSDK` paclet. These libraries contain NVIDIA machine code, which ZLUDA can't run
* the GPUArray libraries shipped with the Wolfram Language contain NVIDIA machine code only
* CUDA code compiled by the Wolfram Language targets an NVIDIA architecture by default

`ZLUDAInit.wl` redirects the library loading to ZLUDA, uses rebuilt GPUArray libraries, and makes
the Wolfram Language compile CUDA code to PTX, which ZLUDA compiles for the AMD GPU. Nothing in
the Wolfram installation is modified.

## Requirements

* Linux with an AMD GPU supported by ROCm
* ROCm with HIP, rocBLAS, hipBLASLt, MIOpen, hipFFT/rocFFT and rocSPARSE
* the build dependencies of ZLUDA (see [Building](../docs/src/building.md))
* Wolfram Language 15.0 with the paclets `WolframGPUSDK` (NVIDIA CUDA libraries, NVRTC and nvcc)
  and, for neural networks, `NeuralNetworksResources`. Check with `PacletFind["WolframGPUSDK"]`
* a C compiler, to rebuild the GPUArray libraries

## Building

ROCm 10 no longer ships `librocm_smi64.so`, which ZLUDA links against. An empty stub library is
enough to link (ZLUDA's NVML library, which needs ROCm SMI, then doesn't work, but the Wolfram integration
doesn't use it):

```
mkdir -p ~/zluda-stubs
gcc -shared -o ~/zluda-stubs/librocm_smi64.so -x c /dev/null
```

Then build ZLUDA from the repository root:

```
LIBRARY_PATH=~/zluda-stubs cargo xtask --release
```

The libraries are in `target/release`. In the following, `<ZLUDA>` is the repository root and
`/opt/rocm/lib` is the directory with the ROCm libraries (`libamdhip64.so`, `librocblas.so`, ...).

## Setup

### 1. Start the kernel with the ZLUDA libraries

Put the ZLUDA and ROCm libraries on `LD_LIBRARY_PATH`, e.g.:

```
LD_LIBRARY_PATH=<ZLUDA>/target/release:/opt/rocm/lib wolframscript
```

Start other Wolfram programs (e.g. the notebook front end) the same way, or set the variable in
the environment they are started from.

### 2. Load ZLUDAInit.wl

`ZLUDAInit.wl` must be loaded before any GPU functionality is used. To load it in every session,
add this line to `~/.Wolfram/Kernel/init.m`:

```wl
Get["<ZLUDA>/wolfram/ZLUDAInit.wl"]
```

or put it at the top of a script. It finds the ZLUDA libraries in this order:

1. the environment variable `ZLUDA_LIB_DIR`
2. the first directory on `LD_LIBRARY_PATH` that contains `libcuda.so.1` (the build or the trace
   directory)
3. `../target/release` relative to `ZLUDAInit.wl`

### 3. Rebuild the GPUArray libraries (for GPUArray only)

Neural networks work without this step. For `GPUArray`, rebuild the GPUArray libraries once, and
again after every Wolfram Language update:

```
LD_LIBRARY_PATH=<ZLUDA>/target/release:/opt/rocm/lib \
    wolframscript -file <ZLUDA>/wolfram/BuildGPUArrayLibraries.wls
```

This compiles the libraries again from the Wolfram Language sources stored in them, with the CUDA
code compiled to PTX, and writes them to `<ZLUDA>/target/release/wolfram`. Each library takes a
few minutes. Single libraries can be rebuilt by passing their names, e.g.
`BuildGPUArrayLibraries.wls CUDAArrays_Random`. `ZLUDAInit.wl` loads the rebuilt libraries
instead of the original ones whenever they exist. Set `ZLUDAInit`$RedirectGPUArrayLibraries =
False` to use the original ones.

## Checking that it works

```wl
Needs["CUDALink`"];
CUDAQ[]
(* True *)
Lookup[CUDAInformation[][[1, 2]], "Name"]
(* "AMD Radeon 890M Graphics [ZLUDA]" *)

g = GPUArray[RandomReal[1, {1000, 1000}]];
Head[g . g]
(* GPUArray *)

net = NetChain[{LinearLayer[16], Ramp, LinearLayer[1]}];
data = Table[{x} -> {Sin[x]}, {x, 0., 6., 0.01}];
NetTrain[net, data, TargetDevice -> "GPU", MaxTrainingRounds -> 50]
```

The first use of each GPU kernel is slow, because ZLUDA compiles it for the GPU. The compiled
code is cached in `~/.cache/zluda/ComputeCache`, so later runs start faster.

## What works

* **Neural networks** (`NetTrain` and evaluation with `TargetDevice -> "GPU"`): linear layers,
  convolution, batch normalization, activations, pooling, softmax, and recurrent layers (LSTM,
  GRU, basic recurrent; bidirectional, stacked, `NetFoldOperator`, `NetUnfold`). Training matches
  CPU training to about 1e-6. Attention and transformer layers are untested.
* **GPU computing**: everything on the
  [GPU Computing with NVIDIA](https://reference.wolfram.com/language/guide/GPUComputing-NVIDIA.html)
  guide page: `GPUArray` arithmetic and elementwise math, linear algebra (`Dot`, `Transpose`,
  `LeastSquares`, `PseudoInverse`, `SingularValueDecomposition`, `CholeskyDecomposition`,
  `GPUArray`GPUArrayLinearSolve`), Fourier transforms, statistics, sorting, random numbers,
  visualization of GPU arrays, `FunctionCompile` with GPU types and `CUDAFunctionLoad`.

## Limitations

* **Silent CPU fallback.** Many GPUArray functions fall back to the CPU without a message when
  the GPU computation fails or isn't implemented. Check that the result is still a `GPUArray`
  (`Head[result]`). To see the error, call the internal function directly, e.g.
  `GPUArray`GPUArrayPseudoInverse[g]` instead of `PseudoInverse[g]`.
* These fall back to the CPU because of the Wolfram Language itself, not ZLUDA:
  * `Inverse[gpuArray]`: `GPUArray`GPUArrayInverse` isn't defined in Wolfram Language 15.0. To
    compute an inverse on the GPU, solve against the identity matrix (LU decomposition with
    cuSOLVER): `GPUArray`GPUArrayLinearSolve[g, GPUArray[N[Normal[IdentityMatrix[n]]]]]`.
    `PseudoInverse[g]` also works, but is much slower.
  * `LinearSolve[gpuMatrix, gpuVector]`: the right-hand side is converted to a CPU array before
    the GPU implementation is looked up. Use `GPUArray`GPUArrayLinearSolve` directly.
  * `PseudoInverse` of complex matrices (no complex implementation; complex
    `SingularValueDecomposition` runs on the GPU).
* `CUDADriverVersion[]` fails, because it reads the version of the NVIDIA kernel driver.
* `FunctionCompile` with GPUArray types needs the compiled component:
  `FunctionCompile[CompiledComponent["CUDAExtensions"], f]`.
* Device-side kernel launches (dynamic parallelism) aren't supported by ZLUDA. `ZLUDAInit.wl`
  links Wolfram-compiled CUDA code without the CUDA device runtime.
* On an integrated GPU, many operations are slower than on the CPU. The GPU pays off for large
  neural networks and for keeping data on the GPU across many operations.

## What ZLUDAInit.wl changes

All changes are made in the running kernel only:

* loads `libzluda_comgr.so` first, which keeps ROCm's LLVM from clashing with the LLVM of the
  Wolfram compiler (see [CHANGES.md](CHANGES.md))
* wraps `GeneralUtilities`GetNNLibPaths` (neural networks) and
  `CUDATools`FindCUDAToolkit`FindCUDALibraries` (CUDALink, GPUArray) to load ZLUDA's cuBLAS,
  cuBLASLt, cuDNN, cuFFT, cuSPARSE and NVRTC instead of the NVIDIA libraries. The NVIDIA cuDNN
  sub-libraries are dropped, because ZLUDA's cuDNN is a single library. The NVIDIA libraries
  that work on top of ZLUDA's driver (cudart, cuRAND, cuSOLVER, ...) are kept
* sets `ZLUDA_NVRTC_LIB` to the NVIDIA NVRTC, which ZLUDA's NVRTC shim forwards to, and
  `CUDA_LIBRARY_PATH` to ZLUDA's `libcuda.so.1`, unless they are already set
* wraps `CompiledLibrary`CompiledLibraryLoadFunctions` to load the rebuilt GPUArray libraries
* sets `CUDALink`$CUDAArchitecture = "compute_80"` (PTX only), makes `CUDAFunctionLoad` build
  PTX-only fatbinaries, and adds `-cudadevrt none` to `NVCC_APPEND_FLAGS`

## Troubleshooting

ZLUDA can log every CUDA call. Use the trace libraries in `target/release/trace` and tell them
where the ZLUDA libraries they wrap are:

```
R=<ZLUDA>/target/release
ZLUDA_CUDA_LIB=$R/libcuda.so ZLUDA_BLAS_LIB=$R/libcublas.so ZLUDA_BLASLT_LIB=$R/libcublaslt.so \
ZLUDA_DNN8_LIB=$R/libcudnn64_8.so ZLUDA_SPARSE_LIB=$R/libcusparse.so ZLUDA_FFT_LIB=$R/libcufft.so \
ZLUDA_LOG_DIR=~/zluda-logs LD_LIBRARY_PATH=$R/trace:/opt/rocm/lib wolframscript -file script.wls
```

`ZLUDAInit.wl` detects the trace directory on `LD_LIBRARY_PATH` by itself. The log is written to
`~/zluda-logs/WolframKernel/log.txt`, together with the GPU modules that were loaded. Calls that
failed are those that don't return `CUDA_SUCCESS`/`CUBLAS_STATUS_SUCCESS`/...; a call returning
`NOT_SUPPORTED` usually is a function that ZLUDA doesn't implement yet. A cuBLAS status that is a
plain number is a rocBLAS status passed through unchanged.

If kernel compilation fails with `CUDA_ERROR_UNKNOWN`, compile the saved PTX module with ZLUDA's
offline compiler to see the error: `cargo build -p compiler`, then
`target/debug/zoc module.ptx` (`--arch gfx1150` to compile for another GPU than the installed one).
