; Function Attrs: denormal_fpenv(preservesign)
define amdgpu_kernel void @prefetch(ptr addrspace(4) byref(i64) %"40", ptr addrspace(4) byref(i64) %"41") #0 {
  %"42" = alloca i64, align 8, addrspace(5)
  %"43" = alloca i64, align 8, addrspace(5)
  %"44" = alloca i64, align 8, addrspace(5)
  %"45" = alloca i64, align 8, addrspace(5)
  br label %1

1:                                                ; preds = %0
  br label %"39"

"39":                                             ; preds = %1
  call void @llvm.amdgcn.s.dcache.inv()
  %2 = load i64, ptr addrspace(4) %"40", align 8
  store i64 %2, ptr addrspace(5) %"42", align 8
  %3 = load i64, ptr addrspace(4) %"41", align 8
  store i64 %3, ptr addrspace(5) %"43", align 8
  %4 = load i64, ptr addrspace(5) %"42", align 8
  %5 = inttoptr i64 %4 to ptr
  %6 = addrspacecast ptr %5 to ptr addrspace(1)
  %"48" = ptrtoint ptr addrspace(1) %6 to i64
  store i64 %"48", ptr addrspace(5) %"44", align 8
  %7 = load i64, ptr addrspace(5) %"44", align 8
  %"57" = inttoptr i64 %7 to ptr addrspace(1)
  %8 = load i64, ptr addrspace(5) %"42", align 8
  %"58" = inttoptr i64 %8 to ptr
  %9 = load i64, ptr addrspace(5) %"42", align 8
  %"59" = inttoptr i64 %9 to ptr
  %10 = load i64, ptr %"59", align 8
  store i64 %10, ptr addrspace(5) %"45", align 8
  %11 = load i64, ptr addrspace(5) %"43", align 8
  %12 = load i64, ptr addrspace(5) %"45", align 8
  %"60" = inttoptr i64 %11 to ptr
  store i64 %12, ptr %"60", align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.dcache.inv() #1

attributes #0 = { denormal_fpenv(preservesign) "amdgpu-ieee"="false" "amdgpu-unsafe-fp-atomics"="true" "no-trapping-math"="true" "target-features"="+wavefrontsize32,-wavefrontsize64,+cumode,+precise-memory" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn }
