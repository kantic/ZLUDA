//! Loads the ROCm code object manager (comgr) into its own link-map namespace.
//!
//! comgr is linked against ROCm's shared LLVM and Clang libraries (libLLVM.so.NNgit and
//! libclang-cpp.so.NNgit), which export their symbols without `-Bsymbolic`. In a process that
//! contains another copy of LLVM, the two copies interfere with each other:
//! * plain symbols of whichever copy is loaded first interpose the symbols of the other one
//! * GNU unique symbols (e.g. the `FoldingSet<T>::getFoldingSetInfo()::Info` statics, which
//!   contain function pointers into the code of one particular LLVM version) are shared by all
//!   libraries of a namespace, regardless of symbol versions, load order or `RTLD_DEEPBIND`
//!
//! The Wolfram Language is such a process: its compiler (LLVMLink) brings its own LLVM. With
//! ROCm's LLVM loaded first, `FunctionCompile` fails with "Cannot verify module"; with Wolfram's
//! LLVM loaded first, HIP hangs while initializing.
//!
//! This library has the soname of the real comgr (`libamd_comgr.so.3`) and must be loaded
//! before anything that depends on comgr (HIP, hiprtc, MIOpen, ...). The dynamic linker then
//! resolves their comgr dependency to this library, which loads the real comgr with `dlmopen`
//! into a new namespace and forwards every comgr function to it. comgr's C API only exchanges
//! opaque handles and caller-provided buffers, so it works across namespaces.
//!
//! The new namespace has its own copy of libc, whose `environ` is a copy of the main `environ`
//! made when the namespace was created. When the process later calls `setenv`, the main libc
//! may reallocate the environment array and free the one the copy points to, and comgr's next
//! `getenv` crashes. So every forwarding function first copies the main `environ` pointer into
//! the `environ` of the comgr namespace.
//!
//! The real comgr is found in this order:
//! 1. the `ZLUDA_COMGR_LIB` environment variable
//! 2. `libamd_comgr.so.3` in the directories of `LD_LIBRARY_PATH` (except this library itself)
//! 3. `$ROCM_PATH/lib/libamd_comgr.so.3`
//! 4. `/opt/rocm/lib/libamd_comgr.so.3`
//!
//! If loading into a new namespace fails, the real comgr is loaded normally (into the default
//! namespace), so ROCm keeps working.

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod imp {
    use std::{
        ffi::{c_char, c_int, c_void, CStr, CString},
        path::{Path, PathBuf},
        sync::atomic::{AtomicUsize, Ordering},
    };

    /// Status returned by the forwarding functions if the real comgr couldn't be loaded
    const AMD_COMGR_STATUS_ERROR: u32 = 1;

    extern "C" fn unavailable() -> u32 {
        AMD_COMGR_STATUS_ERROR
    }

    extern "C" {
        static mut environ: *const *const c_char;
    }

    /// Address of the `environ` variable of the comgr namespace (or of `ENVIRON_SINK` if there is
    /// none). Set by `init` before anything can call a forwarding function.
    static COMGR_ENVIRON: AtomicUsize = AtomicUsize::new(0);
    static ENVIRON_SINK: AtomicUsize = AtomicUsize::new(0);

    // Every forwarding function syncs `environ` (see the module documentation) and then jumps
    // through its slot, so the arguments and the return value are passed through untouched,
    // whatever the signature of the function. The sync only uses r10 and r11, which are
    // scratch registers that don't carry arguments in the System V ABI.
    macro_rules! forward {
        ($($name:ident),* $(,)?) => {
            $(
                #[allow(non_snake_case)]
                mod $name {
                    pub(super) static SLOT: super::AtomicUsize = super::AtomicUsize::new(0);
                }

                #[unsafe(no_mangle)]
                #[unsafe(naked)]
                pub unsafe extern "C" fn $name() {
                    core::arch::naked_asm!(
                        "mov r11, qword ptr [rip + {main_environ}@GOTPCREL]",
                        "mov r11, qword ptr [r11]",
                        "mov r10, qword ptr [rip + {comgr_environ}]",
                        "mov qword ptr [r10], r11",
                        "jmp qword ptr [rip + {slot}]",
                        main_environ = sym environ,
                        comgr_environ = sym COMGR_ENVIRON,
                        slot = sym $name::SLOT,
                    );
                }
            )*

            pub(crate) const FUNCTIONS: &[(&CStr, &AtomicUsize)] = &[
                $(
                    (
                        match CStr::from_bytes_with_nul(concat!(stringify!($name), "\0").as_bytes()) {
                            Ok(name) => name,
                            Err(_) => panic!(),
                        },
                        &$name::SLOT,
                    ),
                )*
            ];
        };
    }

    // All functions exported by comgr 3.x (ROCm 7)
    forward! {
        amd_comgr_action_data_count,
        amd_comgr_action_data_get_data,
        amd_comgr_action_info_get_block_sizes,
        amd_comgr_action_info_get_block_sizes_count,
        amd_comgr_action_info_get_bundle_entry_id,
        amd_comgr_action_info_get_bundle_entry_id_count,
        amd_comgr_action_info_get_isa_name,
        amd_comgr_action_info_get_language,
        amd_comgr_action_info_get_logging,
        amd_comgr_action_info_get_option_list_count,
        amd_comgr_action_info_get_option_list_item,
        amd_comgr_action_info_get_package_entry_id,
        amd_comgr_action_info_get_package_entry_id_count,
        amd_comgr_action_info_get_working_directory_path,
        amd_comgr_action_info_set_block_sizes,
        amd_comgr_action_info_set_bundle_entry_ids,
        amd_comgr_action_info_set_device_lib_linking,
        amd_comgr_action_info_set_isa_name,
        amd_comgr_action_info_set_language,
        amd_comgr_action_info_set_logging,
        amd_comgr_action_info_set_option_list,
        amd_comgr_action_info_set_package_entry_ids,
        amd_comgr_action_info_set_vfs,
        amd_comgr_action_info_set_working_directory_path,
        amd_comgr_create_action_info,
        amd_comgr_create_data,
        amd_comgr_create_data_set,
        amd_comgr_create_disassembly_info,
        amd_comgr_create_symbolizer_info,
        amd_comgr_data_set_add,
        amd_comgr_data_set_remove,
        amd_comgr_demangle_symbol_name,
        amd_comgr_destroy_action_info,
        amd_comgr_destroy_data_set,
        amd_comgr_destroy_disassembly_info,
        amd_comgr_destroy_metadata,
        amd_comgr_destroy_symbolizer_info,
        amd_comgr_disassemble_instruction,
        amd_comgr_do_action,
        amd_comgr_get_data,
        amd_comgr_get_data_isa_name,
        amd_comgr_get_data_kind,
        amd_comgr_get_data_metadata,
        amd_comgr_get_data_name,
        amd_comgr_get_isa_count,
        amd_comgr_get_isa_metadata,
        amd_comgr_get_isa_name,
        amd_comgr_get_mangled_name,
        amd_comgr_get_metadata_kind,
        amd_comgr_get_metadata_list_size,
        amd_comgr_get_metadata_map_size,
        amd_comgr_get_metadata_string,
        amd_comgr_get_version,
        amd_comgr_hotswap_rewrite,
        amd_comgr_hotswap_rewrite_with_options,
        amd_comgr_index_list_metadata,
        amd_comgr_iterate_map_metadata,
        amd_comgr_iterate_symbols,
        amd_comgr_lookup_code_object,
        amd_comgr_map_elf_virtual_address_to_code_object_offset,
        amd_comgr_map_name_expression_to_symbol_name,
        amd_comgr_metadata_lookup,
        amd_comgr_populate_mangled_names,
        amd_comgr_populate_name_expression_map,
        amd_comgr_release_data,
        amd_comgr_set_data,
        amd_comgr_set_data_from_file_slice,
        amd_comgr_set_data_name,
        amd_comgr_status_string,
        amd_comgr_symbol_get_info,
        amd_comgr_symbolize,
        amd_comgr_symbol_lookup,
    }

    const COMGR_NAME: &str = "libamd_comgr.so.3";

    fn own_path() -> Option<PathBuf> {
        let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
        if unsafe { libc::dladdr(unavailable as *const c_void, &mut info) } == 0
            || info.dli_fname.is_null()
        {
            return None;
        }
        let path = unsafe { CStr::from_ptr(info.dli_fname) };
        Some(PathBuf::from(path.to_str().ok()?))
    }

    fn same_file(a: &Path, b: &Path) -> bool {
        match (a.canonicalize(), b.canonicalize()) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }

    pub(crate) fn candidates(
        explicit: Option<String>,
        ld_library_path: Option<String>,
        rocm_path: Option<String>,
        own_path: Option<&Path>,
    ) -> Vec<PathBuf> {
        let mut result = Vec::new();
        if let Some(path) = explicit.filter(|path| !path.is_empty()) {
            result.push(PathBuf::from(path));
        }
        let search_dirs = ld_library_path
            .iter()
            .flat_map(|dirs| dirs.split(':').map(str::to_owned).collect::<Vec<_>>())
            .chain(rocm_path.map(|dir| format!("{dir}/lib")))
            .chain(std::iter::once("/opt/rocm/lib".to_owned()))
            .filter(|dir| !dir.is_empty());
        for dir in search_dirs {
            let path = Path::new(&dir).join(COMGR_NAME);
            if !path.is_file() || own_path.is_some_and(|own| same_file(own, &path)) {
                continue;
            }
            result.push(path);
        }
        result
    }

    fn dlerror() -> String {
        let err = unsafe { libc::dlerror() };
        if err.is_null() {
            "unknown error".to_owned()
        } else {
            unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned()
        }
    }

    unsafe fn open(path: &Path, namespace: Option<libc::Lmid_t>) -> Result<*mut c_void, String> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|e| e.to_string())?;
        let flags: c_int = libc::RTLD_NOW | libc::RTLD_LOCAL;
        let handle = match namespace {
            Some(ns) => libc::dlmopen(ns, c_path.as_ptr(), flags),
            None => libc::dlopen(c_path.as_ptr(), flags),
        };
        if handle.is_null() {
            Err(dlerror())
        } else {
            Ok(handle)
        }
    }

    unsafe fn load() -> Option<*mut c_void> {
        let own = own_path();
        let candidates = candidates(
            std::env::var("ZLUDA_COMGR_LIB").ok(),
            std::env::var("LD_LIBRARY_PATH").ok(),
            std::env::var("ROCM_PATH").ok(),
            own.as_deref(),
        );
        let Some(path) = candidates.first() else {
            eprintln!("[ZLUDA] {COMGR_NAME} not found, set ZLUDA_COMGR_LIB to its path");
            return None;
        };
        match open(path, Some(libc::LM_ID_NEWLM)) {
            Ok(handle) => Some(handle),
            Err(namespace_err) => match open(path, None) {
                Ok(handle) => {
                    eprintln!(
                        "[ZLUDA] Couldn't load {} into a separate namespace ({namespace_err}), loaded it normally",
                        path.display()
                    );
                    Some(handle)
                }
                Err(err) => {
                    eprintln!("[ZLUDA] Couldn't load {}: {err}", path.display());
                    None
                }
            },
        }
    }

    extern "C" fn init() {
        let handle = unsafe { load() };
        let main_environ = &raw const environ as usize;
        let comgr_environ = handle
            .map(|handle| unsafe { libc::dlsym(handle, c"environ".as_ptr()) } as usize)
            .filter(|&address| address != 0 && address != main_environ)
            .unwrap_or(&raw const ENVIRON_SINK as usize);
        COMGR_ENVIRON.store(comgr_environ, Ordering::Relaxed);
        for (name, slot) in FUNCTIONS {
            let target = handle
                .map(|handle| unsafe { libc::dlsym(handle, name.as_ptr() as *const c_char) })
                .filter(|f| !f.is_null())
                .map_or(unavailable as *const () as usize, |f| f as usize);
            slot.store(target, Ordering::Relaxed);
        }
    }

    // Resolve all functions when the library is loaded, before any other library can call them
    #[used]
    #[unsafe(link_section = ".init_array")]
    static INIT: extern "C" fn() = init;

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::fs;

        #[test]
        fn all_slots_are_resolved() {
            // The test binary links this crate as an rlib, so .init_array has run
            for (name, slot) in FUNCTIONS {
                assert_ne!(slot.load(Ordering::Relaxed), 0, "{name:?}");
            }
        }

        #[test]
        fn forwards_to_real_comgr() {
            let mut major = 0usize;
            let mut minor = 0usize;
            unsafe {
                let get_version: unsafe extern "C" fn(*mut usize, *mut usize) =
                    std::mem::transmute(amd_comgr_get_version as *const ());
                get_version(&mut major, &mut minor);
            }
            assert!(major >= 2, "comgr version {major}.{minor}");
            let mut count = 0usize;
            let status = unsafe {
                let get_isa_count: unsafe extern "C" fn(*mut usize) -> u32 =
                    std::mem::transmute(amd_comgr_get_isa_count as *const ());
                get_isa_count(&mut count)
            };
            assert_eq!(status, 0);
            assert!(count > 0);
        }

        #[test]
        fn environ_is_synced_into_comgr_namespace() {
            unsafe { std::env::set_var("ZLUDA_COMGR_TEST_VAR", "1") };
            // Force a reallocation of the environment array of the main libc
            for i in 0..256 {
                unsafe { std::env::set_var(format!("ZLUDA_COMGR_TEST_FILL_{i}"), "x") };
            }
            let mut major = 0usize;
            let mut minor = 0usize;
            unsafe {
                let get_version: unsafe extern "C" fn(*mut usize, *mut usize) =
                    std::mem::transmute(amd_comgr_get_version as *const ());
                get_version(&mut major, &mut minor);
            }
            let comgr_environ = COMGR_ENVIRON.load(Ordering::Relaxed) as *const *const *const c_char;
            assert_ne!(comgr_environ as usize, &raw const ENVIRON_SINK as usize);
            assert_eq!(unsafe { *comgr_environ }, unsafe { environ });
        }

        #[test]
        fn real_comgr_is_in_separate_namespace() {
            // The real comgr (and its LLVM) must not be visible in the default namespace
            let name = CString::new("libLLVM.so.24.0git").unwrap();
            let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_NOLOAD) };
            assert!(handle.is_null());
        }

        #[test]
        fn candidates_order_and_self_exclusion() {
            let dir = std::env::temp_dir().join(format!("zluda_comgr_test_{}", std::process::id()));
            let (own_dir, real_dir) = (dir.join("own"), dir.join("real"));
            fs::create_dir_all(&own_dir).unwrap();
            fs::create_dir_all(&real_dir).unwrap();
            fs::write(own_dir.join(COMGR_NAME), b"").unwrap();
            fs::write(real_dir.join(COMGR_NAME), b"").unwrap();
            let ld_path = format!("{}::{}", own_dir.display(), real_dir.display());
            let own = own_dir.join(COMGR_NAME);
            let found = candidates(None, Some(ld_path.clone()), None, Some(&own));
            assert_eq!(found.first(), Some(&real_dir.join(COMGR_NAME)));
            let found = candidates(Some("/explicit/comgr.so".into()), Some(ld_path), None, Some(&own));
            assert_eq!(found.first(), Some(&PathBuf::from("/explicit/comgr.so")));
            fs::remove_dir_all(&dir).unwrap();
        }
    }
}
