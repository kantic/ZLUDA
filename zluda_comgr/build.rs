fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        // The library takes the place of the real comgr: once it's loaded, the dynamic linker
        // resolves the libamd_comgr.so.3 dependency of HIP, hiprtc, MIOpen, ... to it.
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libamd_comgr.so.3");
    }
}
