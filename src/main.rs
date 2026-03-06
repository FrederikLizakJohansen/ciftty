mod app;
mod cif;
mod model;
mod spacegroup;
mod xrd;

use std::env;
use std::fs;
use std::io::stdout;
use std::path::Path;
use std::path::PathBuf;

use anyhow::{Context, Result};
use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

enum LaunchMode {
    Single(Option<PathBuf>),
    Ensemble {
        samples_dir: PathBuf,
        target_path: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let launch_mode = parse_args()?;

    let (structure, ensemble_input, open_dialog_dir) = match launch_mode {
        LaunchMode::Single(path) => {
            let structure = if let Some(path) = path {
                Some(cif::parse_cif_file(&path)?)
            } else {
                None
            };
            let open_dialog_dir =
                env::current_dir().context("Could not determine current directory")?;
            (structure, None, open_dialog_dir)
        }
        LaunchMode::Ensemble {
            samples_dir,
            target_path,
        } => {
            let ensemble = load_ensemble_input(&samples_dir, target_path.as_ref())?;
            let structure = ensemble.samples.first().map(|(_, s)| s.clone());
            (structure, Some(ensemble), samples_dir)
        }
    };

    let session = TerminalSession::enter().context("Failed to initialize terminal UI")?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))
        .context("Failed to create terminal backend")?;

    let run_result = app::run(&mut terminal, structure, open_dialog_dir, ensemble_input);
    let _ = terminal.show_cursor();
    drop(session);
    run_result
}

fn parse_args() -> Result<LaunchMode> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.as_slice() {
        [] => Ok(LaunchMode::Single(None)),
        [path] if path != "ensemble" => Ok(LaunchMode::Single(Some(PathBuf::from(path)))),
        [subcommand, samples_dir] if subcommand == "ensemble" => Ok(LaunchMode::Ensemble {
            samples_dir: PathBuf::from(samples_dir),
            target_path: None,
        }),
        [subcommand, samples_dir, target_path] if subcommand == "ensemble" => {
            Ok(LaunchMode::Ensemble {
                samples_dir: PathBuf::from(samples_dir),
                target_path: Some(PathBuf::from(target_path)),
            })
        }
        _ => anyhow::bail!(
            "Usage:\n  ciftty [path/to/structure.cif]\n  ciftty ensemble <samples_dir> [target.cif]"
        ),
    }
}

fn load_ensemble_input(
    samples_dir: &Path,
    target_path: Option<&PathBuf>,
) -> Result<app::EnsembleInput> {
    let read_dir = fs::read_dir(samples_dir).with_context(|| {
        format!(
            "Could not read ensemble samples directory {}",
            samples_dir.display()
        )
    })?;

    let mut cif_paths: Vec<PathBuf> = read_dir
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file() && is_cif_path(path))
        .collect();
    cif_paths.sort_by_key(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.to_ascii_lowercase())
            .unwrap_or_default()
    });

    let mut samples = Vec::new();
    let mut skipped_files = 0usize;
    for path in cif_paths {
        match cif::parse_cif_file(&path) {
            Ok(structure) => samples.push((path, structure)),
            Err(_) => skipped_files += 1,
        }
    }

    if samples.is_empty() {
        anyhow::bail!(
            "No valid CIF structures were loaded from {}",
            samples_dir.display()
        );
    }

    let target = if let Some(path) = target_path {
        Some((
            path.clone(),
            cif::parse_cif_file(path)
                .with_context(|| format!("Could not load target CIF {}", path.display()))?,
        ))
    } else {
        None
    };

    Ok(app::EnsembleInput {
        source_dir: samples_dir.to_path_buf(),
        samples,
        target,
        skipped_files,
    })
}

fn is_cif_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("cif"))
        .unwrap_or(false)
}

struct TerminalSession;

impl TerminalSession {
    fn enter() -> Result<Self> {
        enable_raw_mode().context("Could not enable raw mode")?;
        execute!(stdout(), EnterAlternateScreen, EnableMouseCapture)
            .context("Could not enter alternate screen")?;

        // Request repeat/release events via kitty keyboard protocol.
        // REPORT_ALL_KEYS_AS_ESCAPE_CODES is required for plain text keys.
        let _ = execute!(
            stdout(),
            PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                    | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                    | KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES
            )
        );
        Ok(Self)
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = execute!(stdout(), PopKeyboardEnhancementFlags);
        let _ = disable_raw_mode();
        let _ = execute!(stdout(), LeaveAlternateScreen, DisableMouseCapture);
    }
}
