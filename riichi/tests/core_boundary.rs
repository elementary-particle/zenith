use std::{fs, path::PathBuf};

#[test]
fn state_only_core_has_no_binding_batch_thread_or_training_dependency() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("core/Cargo.toml");
    let text = fs::read_to_string(manifest)
        .expect("core manifest")
        .to_lowercase();
    for forbidden in [
        "pyo3",
        "numpy",
        "rayon",
        "torch",
        "batch",
        "threadpool",
        "thread_pool",
    ] {
        assert!(
            !text.contains(forbidden),
            "state-only manifest contains forbidden dependency or concept {forbidden}"
        );
    }
}
