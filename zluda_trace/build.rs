fn main() {
    // Same SONAME as the NVIDIA driver, see zluda/build.rs
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libcuda.so.1");
    }
}
