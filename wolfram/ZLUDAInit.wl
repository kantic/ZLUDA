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
	Unprotect[GeneralUtilities`GetNNLibPaths];
	(* Villegas-Gayley wrapper around the original definition. It's explicitly prepended,
	   because the automatic rule ordering could put it after the original definition. *)
	DownValues[GeneralUtilities`GetNNLibPaths] = Prepend[
		DownValues[GeneralUtilities`GetNNLibPaths],
		HoldPattern[GeneralUtilities`GetNNLibPaths[args___]] :>
			Block[{$inGetNNLibPaths = True}, redirectLibPaths[GeneralUtilities`GetNNLibPaths[args]]] /; !TrueQ[$inGetNNLibPaths]
	];
	Protect[GeneralUtilities`GetNNLibPaths];
];

End[];
EndPackage[];
