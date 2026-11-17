(* ::Package:: *)

(* ZLUDA integration for the Wolfram Language.

   Wolfram's neural network runtime (MXNet, Torch, ...) loads the NVIDIA CUDA libraries
   (cuBLAS, cuDNN, cuFFT, ...) by absolute path from the WolframGPUSDK paclet. Those
   libraries contain NVIDIA machine code (SASS), which ZLUDA can't run. This file redirects
   the libraries for which ZLUDA has a replacement to the ZLUDA builds, while keeping the
   NVIDIA libraries that work on top of ZLUDA's libcuda (cudart, nvrtc, curand, ...).

   Usage: load this file before using any GPU functionality, e.g. by adding
       Get["/path/to/ZLUDA/wolfram/ZLUDAInit.wl"]
   to ~/.Wolfram/Kernel/init.m, and start the kernel with the ZLUDA libraries on
   LD_LIBRARY_PATH (see AGENTS.md).

   The ZLUDA library directory defaults to ../target/release relative to this file and can
   be overridden with the environment variable ZLUDA_LIB_DIR.
*)

BeginPackage["ZLUDAInit`"];

$ZLUDALibraryDirectory::usage = "$ZLUDALibraryDirectory is the directory from which the ZLUDA replacements for the CUDA libraries are loaded.";
$GPUArrayLibraryDirectory::usage = "$GPUArrayLibraryDirectory is the directory with the GPUArray libraries rebuilt for ZLUDA by BuildGPUArrayLibraries.wls.";
$RedirectGPUArrayLibraries::usage = "$RedirectGPUArrayLibraries specifies whether the GPUArray libraries in $GPUArrayLibraryDirectory are used instead of the original ones.";

Begin["`Private`"];

(* Directory with the ZLUDA libraries. Priority:
   1. ZLUDA_LIB_DIR environment variable
   2. the first directory on LD_LIBRARY_PATH that contains libcuda.so.1 (this is either the
      ZLUDA build directory or the ZLUDA trace directory)
   3. ../target/release relative to this file *)
$ZLUDALibraryDirectory = Which[
	DirectoryQ[Environment["ZLUDA_LIB_DIR"]],
		Environment["ZLUDA_LIB_DIR"],
	StringQ[Environment["LD_LIBRARY_PATH"]] && AnyTrue[
			StringSplit[Environment["LD_LIBRARY_PATH"], ":"],
			FileExistsQ[FileNameJoin[{#, "libcuda.so.1"}]]&],
		SelectFirst[
			StringSplit[Environment["LD_LIBRARY_PATH"], ":"],
			FileExistsQ[FileNameJoin[{#, "libcuda.so.1"}]]&],
	True,
		FileNameJoin[{ParentDirectory[DirectoryName[$InputFileName]], "target", "release"}]
];

(* NVIDIA library base name (without version suffix) -> ZLUDA replacement file name,
   or Nothing if the library is subsumed by another ZLUDA library. *)
$replacements = <|
	"libcublas" -> "libcublas.so",
	"libcublasLt" -> "libcublasLt.so",
	"libcufft" -> "libcufft.so",
	"libcusparse" -> "libcusparse.so",
	"libcudnn" -> "libcudnn.so.8",
	(* NVRTC shim: forwards to the NVIDIA NVRTC, but always produces PTX *)
	"libnvrtc" -> "libnvrtc.so",
	(* ZLUDA's cuDNN is a single library *)
	"libcudnn_ops_infer" -> Nothing,
	"libcudnn_ops_train" -> Nothing,
	"libcudnn_cnn_infer" -> Nothing,
	"libcudnn_cnn_train" -> Nothing,
	"libcudnn_adv_infer" -> Nothing,
	"libcudnn_adv_train" -> Nothing
|>;

libBaseName[path_String] := First[StringSplit[FileNameTake[path], "."]];

(* Find a ZLUDA library. Libraries without a trace wrapper (e.g. the NVRTC shim) are not in the
   ZLUDA trace directory, so fall back to its parent directory, the ZLUDA build directory. *)
findZLUDALibrary[name_String] := SelectFirst[
	{FileNameJoin[{$ZLUDALibraryDirectory, name}], FileNameJoin[{ParentDirectory[$ZLUDALibraryDirectory], name}]},
	FileExistsQ,
	None
];

redirectLibrary[path_String] := Module[{repl = Lookup[$replacements, libBaseName[path], None], zludaPath},
	Which[
		repl === None, path,
		repl === Nothing, Nothing,
		StringQ[zludaPath = findZLUDALibrary[repl]],
			(* The NVRTC shim needs to know where the real NVRTC is *)
			If[libBaseName[path] === "libnvrtc" && !StringQ[Environment["ZLUDA_NVRTC_LIB"]],
				SetEnvironment["ZLUDA_NVRTC_LIB" -> path]
			];
			zludaPath,
		True, path
	]
];
redirectLibrary[other_] := other;

(* ZLUDA's libcuda.so.1 is loaded first: the NVIDIA libraries usually pull in libcuda.so.1
   by themselves, but ZLUDA's replacements don't. Without it, libmxnet.so would resolve its
   libcuda.so.1 dependency through its DT_RPATH and pick up the stub library shipped in the
   NeuralNetworksResources paclet. *)
zludaDriver[] := With[{path = FileNameJoin[{$ZLUDALibraryDirectory, "libcuda.so.1"}]},
	If[FileExistsQ[path], {path}, {}]
];

redirectLibPaths[{dir_, libs_List}] := {dir, Join[zludaDriver[], Map[redirectLibrary, libs]]};
redirectLibPaths[other_] := other;

Needs["GeneralUtilities`"];

If[!TrueQ[$installed],
	$installed = True;

	(* HIP loads the ROCm code object manager (comgr), which brings ROCm's LLVM. That LLVM and
	   the one of the Wolfram compiler can't coexist in one namespace: FunctionCompile fails with
	   "Cannot verify module" after HIP has been loaded, and HIP hangs if the compiler was used
	   first. ZLUDA's comgr shim (soname libamd_comgr.so.3) loads the real comgr into a separate
	   namespace. It must be loaded before HIP, so that HIP's comgr dependency resolves to it. *)
	With[{comgr = findZLUDALibrary["libzluda_comgr.so"]},
		If[StringQ[comgr], LibraryLoad[comgr]]
	];

	Unprotect[GeneralUtilities`GetNNLibPaths];
	(* Villegas-Gayley wrapper around the original definition. It's explicitly prepended,
	   because the automatic rule ordering could put it after the original definition. *)
	DownValues[GeneralUtilities`GetNNLibPaths] = Prepend[
		DownValues[GeneralUtilities`GetNNLibPaths],
		HoldPattern[GeneralUtilities`GetNNLibPaths[args___]] :>
			Block[{$inGetNNLibPaths = True}, redirectLibPaths[GeneralUtilities`GetNNLibPaths[args]]] /; !TrueQ[$inGetNNLibPaths]
	];
	Protect[GeneralUtilities`GetNNLibPaths];

	(* CUDALink and GPUArray (CUDATools) look for the CUDA driver by searching a fixed list
	   of system directories for libcuda.so, unless CUDA_LIBRARY_PATH names the library. *)
	If[!StringQ[Environment["CUDA_LIBRARY_PATH"]] && zludaDriver[] =!= {},
		SetEnvironment["CUDA_LIBRARY_PATH" -> First[zludaDriver[]]]
	];

	(* CUDALink`InstallCUDA loads the CUDA toolkit libraries returned by FindCUDALibraries
	   (from the WolframGPUSDK paclet) into the global symbol scope. The GPUArray libraries
	   (Accelerators/LibraryResources/CUDAArrays_*.so) don't link against them, but bind
	   cuBLAS/cuFFT/... symbols from that scope, so the replacement has to happen here. *)
	(* The original definition has the literal left-hand side FindCUDALibraries[], which is
	   tried before any rule with a pattern, so the wrapper must use the same left-hand side. *)
	Needs["CUDATools`FindCUDAToolkit`"];
	Unprotect[CUDATools`FindCUDAToolkit`FindCUDALibraries];
	DownValues[CUDATools`FindCUDAToolkit`FindCUDALibraries] = Prepend[
		DownValues[CUDATools`FindCUDAToolkit`FindCUDALibraries],
		HoldPattern[CUDATools`FindCUDAToolkit`FindCUDALibraries[]] :>
			Block[{$inFindCUDALibraries = True},
				redirectLibraryList[CUDATools`FindCUDAToolkit`FindCUDALibraries[]]
			] /; !TrueQ[$inFindCUDALibraries]
	];

	(* The GPUArray libraries (Accelerators/LibraryResources/CUDAArrays_*.so) only contain NVIDIA
	   machine code. wolfram/BuildGPUArrayLibraries.wls rebuilds them with PTX into
	   $GPUArrayLibraryDirectory; load those instead of the originals when they exist. *)
	(* Loading CompiledLibrary` after GeneralUtilities` reports that SystemSymbolQ exists in both
	   GeneralUtilities` and CompileUtilities`Symbols`, which is harmless *)
	Quiet[Needs["CompiledLibrary`"], {General::shdw}];
	Unprotect[CompiledLibrary`CompiledLibraryLoadFunctions];
	DownValues[CompiledLibrary`CompiledLibraryLoadFunctions] = Prepend[
		DownValues[CompiledLibrary`CompiledLibraryLoadFunctions],
		HoldPattern[CompiledLibrary`CompiledLibraryLoadFunctions[CompiledLibrary`CompiledLibrary[path_String], opts___]] :>
			CompiledLibrary`CompiledLibraryLoadFunctions[CompiledLibrary`CompiledLibrary[gpuArrayLibrary[path]], opts] /;
				gpuArrayLibrary[path] =!= path
	];
	Protect[CompiledLibrary`CompiledLibraryLoadFunctions];
];

If[!ValueQ[$RedirectGPUArrayLibraries], $RedirectGPUArrayLibraries = True];

(* CUDA code compiled by the Wolfram Language (FunctionCompile with GPUArray types, CUDAFunctionLoad,
   ...) targets the architecture of the device by default: "sm_" <> compute capability. ZLUDA's
   compute capability (8.5) isn't a real NVIDIA architecture, so nvcc rejects it, and ZLUDA can't run
   SASS anyway. Use a virtual architecture instead, which makes nvcc emit PTX only. *)
If[!StringQ[CUDALink`$CUDAArchitecture], CUDALink`$CUDAArchitecture = "compute_80"];

(* CUDAFunctionLoad doesn't use $CUDAArchitecture: by default it builds a cubin for the device's
   architecture. Make it build a fatbinary with PTX only instead. Not plain PTX: CUDALink passes the
   contents of the .ptx file to cuModuleLoadData without a terminating NUL, while a fatbinary
   carries its size. Its defaults are set when CUDALink is loaded, so load it now (that doesn't
   initialize CUDA). *)
Quiet[Needs["CUDALink`"], {General::shdw}];
Scan[
	SetOptions[#,
		"CUDAArchitecture" -> CUDALink`$CUDAArchitecture,
		"CreateCUBIN" -> False, "CreatePTX" -> False, "CreateFatBinary" -> True
	] &,
	{CUDALink`CUDAFunctionLoad, CUDALink`Private`iCompileCUDAKernel}
];

(* The compiler builds CUDA templates with -dc and a device link per template library. The device
   link refers to the fatbinary of the CUDA device runtime (cudadevrt), which then isn't linked into
   the library, so loading it fails with an undefined __fatbinwrap_*_cuda_device_runtime_cu_* (and
   with several template libraries, the registration code is duplicated). The device runtime is only
   needed for launching kernels from the GPU, which ZLUDA doesn't support. *)
With[{flags = Replace[Environment["NVCC_APPEND_FLAGS"], Except[_String] -> ""]},
	If[!StringContainsQ[flags, "-cudadevrt"],
		SetEnvironment["NVCC_APPEND_FLAGS" -> StringTrim[flags <> " -cudadevrt none"]]
	]
];

(* Directory with the rebuilt GPUArray libraries: "wolfram" in the ZLUDA build directory
   (the parent of $ZLUDALibraryDirectory for trace runs) *)
$GPUArrayLibraryDirectory = FileNameJoin[{
	If[FileExistsQ[FileNameJoin[{$ZLUDALibraryDirectory, "libzluda_comgr.so"}]],
		$ZLUDALibraryDirectory,
		ParentDirectory[$ZLUDALibraryDirectory]
	],
	"wolfram"
}];

(* The Accelerators loader passes library names without directory and extension *)
gpuArrayLibrary[path_String] := With[{
		rebuilt = FileNameJoin[{$GPUArrayLibraryDirectory, FileBaseName[path] <> "." <> Internal`DynamicLibraryExtension[]}]
	},
	If[TrueQ[$RedirectGPUArrayLibraries] && StringStartsQ[FileNameTake[path], "CUDAArrays_"] &&
			!StringStartsQ[path, $GPUArrayLibraryDirectory] && FileExistsQ[rebuilt],
		rebuilt,
		path
	]
];

(* libcusolverMg needs libnvJitLink.so.12 but can't find it by itself. With the NVIDIA libraries,
   libcusparse loads it first, but ZLUDA's libcusparse doesn't, so load it explicitly. *)
redirectLibraryList[libs_List] := Module[{jitLink},
	jitLink = Take[Sort[FileNames["libnvJitLink.so*", DeleteDuplicates[DirectoryName /@ libs]]], UpTo[1]];
	DeleteDuplicates[Join[jitLink, Map[redirectLibrary, libs]]]
];
redirectLibraryList[other_] := other;

End[];
EndPackage[];
