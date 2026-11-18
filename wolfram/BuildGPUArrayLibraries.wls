#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Rebuilds the GPUArray libraries for ZLUDA.

   The GPUArray libraries shipped with the Wolfram Language (Accelerators/LibraryResources/
   CUDAArrays_*.so) only contain NVIDIA machine code (SASS) for most of their kernels, which
   ZLUDA can't run. This script compiles them again from the sources stored in the libraries
   themselves (every exported function carries its Wolfram Language source), with the CUDA
   template code compiled to PTX only. ZLUDA compiles that PTX for the AMD GPU.

   The libraries are written to ZLUDAInit`$GPUArrayLibraryDirectory ("wolfram" in the ZLUDA
   build directory), from where ZLUDAInit.wl loads them instead of the originals. Nothing is
   changed in the Wolfram installation.

   Requirements: the WolframGPUSDK paclet (nvcc), a C compiler, and the environment of a normal
   ZLUDA run (ZLUDA and ROCm libraries on LD_LIBRARY_PATH, see README.md), because the original
   libraries have to be loaded to read their sources:

       LD_LIBRARY_PATH=<ZLUDA>/target/release:/opt/rocm/core-10.1/lib \
           wolframscript -file wolfram/BuildGPUArrayLibraries.wls [library ...]

   Without arguments, all libraries are built; otherwise only the given ones (e.g.
   CUDAArrays_Random). Each library takes a few minutes. *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "ZLUDAInit.wl"}]];

Needs["CUDALink`"];
If[!TrueQ[CUDALink`CUDAQ[]],
	Print["CUDALink`CUDAQ[] is not True. Run this script with the ZLUDA libraries on LD_LIBRARY_PATH."];
	Exit[1]
];
Needs["CompiledLibrary`"];
Needs["GPUArray`"];
Needs["CUDACompileTools`"];

(* The libraries loaded by Accelerators/Kernel/Loader.m *)
$libraries = {
	"CUDAArrays_Arithmetic",
	"CUDAArrays_Arrays",
	"CUDAArrays_Base",
	"CUDAArrays_Complex",
	"CUDAArrays_DescriptiveStatistics",
	"CUDAArrays_Fourier",
	"CUDAArrays_LinearAlgebra",
	"CUDAArrays_OrderStatistics",
	"CUDAArrays_Random",
	"CUDAArrays_Univariate"
};

(* A virtual architecture makes nvcc emit PTX only (-gencode=arch=compute_80,code=compute_80).
   ZLUDAInit.wl also sets this as the default (CUDALink`$CUDAArchitecture), and adds
   -cudadevrt none to NVCC_APPEND_FLAGS, without which the libraries don't link. *)
$architecture = "compute_80";

$originalDirectory = FileNameJoin[{PacletObject["Accelerators"]["Location"], "LibraryResources", $SystemID}];
$outputDirectory = ZLUDAInit`$GPUArrayLibraryDirectory;
$extension = Internal`DynamicLibraryExtension[];

(* The libraries were built from the "GPUArray" component (declared through CUDAExtensions)
   plus template declarations from the CUDACompileTools components. *)
DeclareCompiledComponent["ZLUDAGPUArray",
	"Declarations" -> Map[CompiledComponent, {"CUDAExtensions", "CUDARandom"}]
];

(* {sources, exported} for the original library: sources is name -> Typed[signature][Function[...]]
   for every function in the library, exported are the names of the functions CompiledLibrary
   loads from it (without e.g. the initialization functions, which run when it's loaded) *)
originalLibrary[lib_String] := Block[{ZLUDAInit`$RedirectGPUArrayLibraries = False},
	Module[{funs},
		funs = CompiledLibraryLoadFunctions[
			CompiledLibrary[FileNameJoin[{$originalDirectory, lib <> "." <> $extension}]],
			"ExpressionExportOnly" -> False
		];
		If[!AssociationQ[funs] || Length[funs] == 0, Return[$Failed, Module]];
		(* CompiledCodeFunction[<|..., "Input" -> Compile`Program[component, <|name -> source, ...|>]|>],
		   the same program for every function of the library *)
		{SelectFirst[List @@ Lookup[First[First[funs]], "Input"], AssociationQ, $Failed], Keys[funs]}
	]
];

exportedNames[path_String] := Keys[CompiledLibraryLoadFunctions[CompiledLibrary[path], "ExpressionExportOnly" -> False]];

buildLibrary[lib_String] := Module[{original, sources, expected, tempDir, built, exported, target, time},
	original = originalLibrary[lib];
	If[!MatchQ[original, {_Association, _List}],
		Print[lib, ": couldn't read the sources from the original library"];
		Return[$Failed, Module]
	];
	{sources, expected} = original;
	tempDir = CreateDirectory[];
	{time, built} = AbsoluteTiming[FunctionCompileExportLibrary[
		FileNameJoin[{tempDir, lib <> "." <> $extension}],
		CompiledComponent["ZLUDAGPUArray"],
		sources,
		CompilerOptions -> {"PassOptions" -> {"TemplateExpansion" -> {"CUDAArchitecture" -> $architecture}}}
	]];
	If[!StringQ[built] || !FileExistsQ[built],
		Print[lib, ": build failed: ", built];
		DeleteDirectory[tempDir, DeleteContents -> True];
		Return[$Failed, Module]
	];
	exported = exportedNames[built];
	If[!ListQ[exported] || Sort[exported] =!= Sort[expected],
		Print[lib, ": the rebuilt library doesn't export the same functions as the original. ",
			If[ListQ[exported],
				{"Missing: ", Complement[expected, exported], ", additional: ", Complement[exported, expected]},
				{"Loading it gave ", Short[exported, 5]}
			]
		];
		DeleteDirectory[tempDir, DeleteContents -> True];
		Return[$Failed, Module]
	];
	If[!DirectoryQ[$outputDirectory], CreateDirectory[$outputDirectory]];
	target = FileNameJoin[{$outputDirectory, FileNameTake[built]}];
	CopyFile[built, target, OverwriteTarget -> True];
	DeleteDirectory[tempDir, DeleteContents -> True];
	Print[lib, ": ", Length[sources], " functions, ", Round[time], " s -> ", target];
	target
];

requested = If[Length[$ScriptCommandLine] > 1, Rest[$ScriptCommandLine], $libraries];
unknown = Complement[requested, $libraries];
If[unknown =!= {}, Print["Unknown libraries: ", unknown, ". Known libraries: ", $libraries]; Exit[1]];

results = AssociationMap[buildLibrary, requested];
failed = Keys[Select[results, FailureQ]];
If[failed =!= {}, Print["Failed: ", failed]; Exit[1]];
