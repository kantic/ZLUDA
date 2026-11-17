# ZLUDA for Wolfram

This project is a fork of original [ZLUDA repository](https://github.com/vosen/ZLUDA).
The purpose of this fork is to implement missing functions that are needed by Wolfram's GPU computation capability and neural network functions.

Available commands:
- Building this project needs a stub for the `librocm_smi64.so` library, because it isn't shipped with the latest ROCm 10.1.0 installation. Thus, to build this project use the following command: `LIBRARY_PATH=~/Software/zluda/stubs cargo xtask --release`.
- The ROCm libraries are installed in `/opt/rocm/lib`. Thus, to start wolframscript with the built ZLUDA libraries and the ROCm libraries, run `LD_LIBRARY_PATH=/home/jonas/Software/zluda/ZLUDA/target/release:/opt/rocm/core-10.1/lib wolframscript`
- To generate trace files, run the command `ZLUDA_CUDA_LIB=/home/jonas/Software/zluda/ZLUDA/target/release/libcuda.so LD_LIBRARY_PATH=/home/jonas/Software/zluda/ZLUDA/target/release/trace:/opt/rocm/core-10.1/lib ZLUDA_LOG_DIR=/home/jonas/Software/zluda/logs wolframscript`. The logs will then be in the corresponding subdirectory of `/home/jonas/Software/zluda/logs`.
- Wolfram loads the CUDA libraries (cuBLAS, cuDNN, NVRTC, ...) from its WolframGPUSDK paclet by absolute path. `wolfram/ZLUDAInit.wl` redirects them to the ZLUDA libraries; it must be loaded before any GPU function is used, e.g. `Get["/home/jonas/Software/zluda/ZLUDA/wolfram/ZLUDAInit.wl"]` in `~/.Wolfram/Kernel/init.m` or at the top of a script. It picks the ZLUDA library directory from `LD_LIBRARY_PATH`, so it works for both normal and trace runs.
- For trace runs, the trace libraries need the paths of the ZLUDA libraries they wrap: `ZLUDA_BLAS_LIB=.../target/release/libcublas.so ZLUDA_BLASLT_LIB=.../target/release/libcublaslt.so ZLUDA_DNN8_LIB=.../target/release/libcudnn64_8.so ZLUDA_SPARSE_LIB=.../target/release/libcusparse.so ZLUDA_FFT_LIB=.../target/release/libcufft.so` in addition to `ZLUDA_CUDA_LIB`.
