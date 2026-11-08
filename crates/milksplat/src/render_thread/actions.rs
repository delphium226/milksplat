//! Hotkey actions on the render thread (the main thread handles
//! fullscreen, monitors and Esc).

use ms_engine::{Command, State, DEFAULT_PRESET_ID};

use super::{display_name, RenderThread, MAX_AV_OFFSET_MS};
use crate::hotkeys::Action;

impl RenderThread {
    /// The preset on screen (the incoming one during a blend).
    pub(super) fn shown_id(&self) -> String {
        match self.director.current() {
            State::Showing { cur } => cur,
            State::Blending { to, .. } => to,
        }
    }

    pub(super) fn action(&mut self, a: Action) {
        let now = self.clock.now();
        match a {
            Action::Next => self.director.command(Command::Next),
            Action::Prev => self.director.command(Command::Prev),
            Action::Random => self.director.command(Command::Random),
            Action::HardCut => self.director.command(Command::HardCut),
            Action::ToggleLock => {
                // Applied by the next update; the notice shows the new state.
                let text = if self.director.locked() {
                    "Unlocked"
                } else {
                    "Locked"
                };
                self.director.command(Command::ToggleLock);
                self.overlay.notice(text, now);
            }
            Action::Favourite => self.favourite(now),
            Action::Ban => self.ban(now),
            Action::AvOffset(d) => {
                self.av_offset_ms = self
                    .av_offset_ms
                    .saturating_add(d)
                    .clamp(0, MAX_AV_OFFSET_MS);
                let text = format!("A/V offset: {} ms", self.av_offset_ms);
                self.overlay.notice(text, now);
            }
            Action::CycleQuality => {
                self.quality = self.quality.next();
                self.cfg.quality = self.quality;
                self.director.set_mesh_cap(self.quality.mesh_cap());
                let text = format!("Quality: {}", self.quality.name());
                self.overlay.notice(text, now);
            }
            Action::ShowName => self.overlay.show_name(now),
            Action::ToggleFps => self.overlay.show_debug = !self.overlay.show_debug,
            Action::ToggleHelp => self.overlay.show_help = !self.overlay.show_help,
            // The main thread handles these.
            Action::ToggleFullscreen | Action::NextMonitor | Action::EscapeKey => {}
        }
    }

    /// The library id on screen, or `None` for the built-in default.
    fn library_id(&self) -> Option<String> {
        let id = self.shown_id();
        (id != DEFAULT_PRESET_ID).then_some(id)
    }

    fn favourite(&mut self, now: f64) {
        let Some(id) = self.library_id() else {
            self.overlay
                .notice("The built-in preset cannot be favourited", now);
            return;
        };
        let mut prefs = self.director.prefs().clone();
        let text = if prefs.favourites.remove(&id) {
            self.cfg.favourites.remove(&id);
            format!("Removed from favourites: {}", display_name(&id))
        } else {
            prefs.favourites.insert(id.clone());
            self.cfg.favourites.insert(id.clone());
            format!("Favourite: {}", display_name(&id))
        };
        self.director.set_prefs(prefs);
        self.overlay.notice(text, now);
    }

    fn ban(&mut self, now: f64) {
        let Some(id) = self.library_id() else {
            self.overlay
                .notice("The built-in preset cannot be banned", now);
            return;
        };
        let mut prefs = self.director.prefs().clone();
        prefs.bans.insert(id.clone());
        prefs.favourites.remove(&id);
        self.cfg.bans.insert(id.clone());
        self.cfg.favourites.remove(&id);
        self.director.set_prefs(prefs);
        self.director.command(Command::Next);
        self.overlay
            .notice(format!("Banned: {}", display_name(&id)), now);
    }
}
