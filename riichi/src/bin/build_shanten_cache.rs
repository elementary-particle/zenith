use std::{env, path::PathBuf, process::ExitCode};

fn main() -> ExitCode {
    let path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(riichi_core::game::rules::shanten::configured_cache_path);
    match riichi_core::game::rules::shanten::build_cache(&path) {
        Ok(()) => {
            println!("{}", path.display());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("failed to build {}: {error}", path.display());
            ExitCode::FAILURE
        }
    }
}
