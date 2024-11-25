fn main() {
    #[cfg(not(debug_assertions))]
    {
        let mut soversion = std::env::var("CARGO_PKG_VERSION_MAJOR").unwrap();
        if soversion == "0" {
            soversion = format!("0.{}", std::env::var("CARGO_PKG_VERSION_MINOR").unwrap());
        }

        // Add soname and strip symbols in release builds
        println!(
            "cargo:rustc-cdylib-link-arg=-Wl,-soname,libnodeset.so.{},-s",
            soversion
        );
    }
}
