//! The render thread's link to the rest of the run, outside the frame
//! work: the capture device's A/V offset, the silence state (hint and
//! keep-awake), the preset on screen for crash reports and the run marker,
//! hot reload, and copying config changes to the store. Only the marker
//! touches the disk here: a tiny synchronous write when the preset on
//! screen changes (rare), before that preset's first tick. The store and
//! the watcher have their own threads.

use std::sync::{Mutex, PoisonError};

use ms_engine::Director;

use ms_preset::Prefs;

use super::audio::DeviceStarted;
use super::{prefs_of, RenderThread};
use crate::crash::RunMarker;
use crate::hotreload::Watch;
use crate::keepawake::SILENCE_S;

impl RenderThread {
    /// Moves captured audio into the history; a capture that started on a
    /// device picks up that device's A/V offset.
    pub(super) fn drain_audio(&mut self) {
        if let Some(DeviceStarted { name, bluetooth }) = self.audio.drain(&mut self.hist) {
            self.av_offset_ms = self.cfg.av_offset_ms(&name, bluetooth);
            tracing::info!("A/V offset for {name}: {} ms", self.av_offset_ms);
            self.av_device = Some(name);
        }
    }

    /// Tracks 10 s of silence: the hint, and keep-awake on the main thread.
    pub(super) fn check_silence(&mut self, wall: f64) {
        let silent = self.hist.silent_for(wall) >= SILENCE_S;
        if silent != self.silent {
            self.silent = silent;
            self.overlay.no_audio = silent;
            (self.on_silence)(silent);
        }
    }

    /// Hot reload: hands a rescanned library to the director, and reloads
    /// the preset on screen if its file changed.
    pub(super) fn poll_reload(&mut self) {
        let Some(change) = self.watch.as_ref().and_then(Watch::try_recv) else {
            return;
        };
        if let Some(lib) = change.library {
            tracing::info!("preset library changed: {} presets", lib.entries.len());
            self.director.set_library(lib);
        }
        let shown = self.shown_id();
        if change.ids.contains(&shown) {
            if self.director.reload_current() {
                tracing::info!("{shown} changed on disk; reloading it");
            } else {
                tracing::info!("{shown} changed on disk; not reloaded (a switch is loading)");
            }
        }
    }

    /// The director's preferences from the config and the session skips.
    pub(super) fn prefs(&self) -> Prefs {
        prefs_of(&self.cfg, &self.session_skip)
    }

    /// Copies the render thread's config fields to the store, which saves
    /// them on its own thread (debounced). Called on changes only.
    pub(super) fn persist(&self) {
        let c = &self.cfg;
        let (favourites, bans, blocklist) =
            (c.favourites.clone(), c.bans.clone(), c.blocklist.clone());
        let (quality, offsets) = (c.quality, c.av_offsets_ms.clone());
        self.store.update(move |s| {
            s.favourites = favourites;
            s.bans = bans;
            s.blocklist = blocklist;
            s.quality = quality;
            s.av_offsets_ms = offsets;
        });
    }
}

/// Records the preset on screen (`director.shown()`) for crash reports and
/// in the run marker, synchronously. The render thread calls it after
/// `Director::update` started a switch and before `tick_and_render` runs
/// the new preset. (A free function: the caller holds other borrows.)
pub(super) fn mark_showing(
    director: &Director,
    marked: &mut Option<String>,
    slot: &Mutex<Option<String>>,
    marker: Option<&RunMarker>,
) {
    let shown = director.shown();
    *marked = Some(shown.to_string());
    *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(shown.to_string());
    if let Some(m) = marker {
        m.set_showing(shown);
    }
}
