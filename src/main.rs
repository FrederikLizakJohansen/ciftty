mod app;
mod cif;
mod model;

use std::env;
use std::io::stdout;
use std::path::PathBuf;

use anyhow::{Context, Result};
use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

fn main() -> Result<()> {
    let path = parse_args()?;
    let structure = if let Some(path) = path {
        Some(cif::parse_cif_file(&path)?)
    } else {
        None
    };
    let open_dialog_dir = env::current_dir().context("Could not determine current directory")?;

    let session = TerminalSession::enter().context("Failed to initialize terminal UI")?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))
        .context("Failed to create terminal backend")?;

    let run_result = app::run(&mut terminal, structure, open_dialog_dir);
    let _ = terminal.show_cursor();
    drop(session);
    run_result
}

fn parse_args() -> Result<Option<PathBuf>> {
    let mut args = env::args().skip(1);
    let path = args.next();

    if args.next().is_some() {
        anyhow::bail!("Usage: ciftty [path/to/structure.cif]");
    }

    Ok(path.map(PathBuf::from))
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
