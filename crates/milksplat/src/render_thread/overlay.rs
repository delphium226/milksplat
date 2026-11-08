//! The on-screen text: the preset name after a change (and on `N`), short
//! notices for hotkeys, the fps/debug line (`` ` ``), the help panel
//! (`H`) and the no-audio hint (§8.2).

use ms_render::TextItem;

use crate::hotkeys::HELP;

/// How long the preset name stays up after a change or `N`.
pub const NAME_SECONDS: f64 = 3.0;
/// How long a notice ("Locked", "Quality: Low", ...) stays up.
pub const NOTICE_SECONDS: f64 = 2.0;
/// Shown at the bottom centre after 10 s of silence (an ASCII hyphen:
/// font8x8 has no em dash).
pub const NO_AUDIO_HINT: &str = "No audio detected - is something playing?";

const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const YELLOW: [f32; 4] = [1.0, 0.9, 0.4, 1.0];
const GREY: [f32; 4] = [0.85, 0.85, 0.85, 1.0];
const MARGIN: f32 = 12.0;

/// What text is up and until when (times are app-clock seconds).
#[derive(Debug, Default)]
pub(super) struct Overlay {
    shown_id: Option<String>,
    name_until: f64,
    notice: Option<(String, f64)>,
    pub show_debug: bool,
    pub show_help: bool,
    /// Show [`NO_AUDIO_HINT`].
    pub no_audio: bool,
}

impl Overlay {
    /// Notes the preset on screen; a change shows its name.
    pub(super) fn set_shown(&mut self, id: &str, now: f64) {
        if self.shown_id.as_deref() != Some(id) {
            self.shown_id = Some(id.to_string());
            self.name_until = now + NAME_SECONDS;
        }
    }

    /// Shows the preset name again (`N`).
    pub(super) fn show_name(&mut self, now: f64) {
        self.name_until = now + NAME_SECONDS;
    }

    /// Shows a short notice.
    pub(super) fn notice(&mut self, text: impl Into<String>, now: f64) {
        self.notice_for(text, now, NOTICE_SECONDS);
    }

    /// Shows a notice for `seconds`.
    pub(super) fn notice_for(&mut self, text: impl Into<String>, now: f64, seconds: f64) {
        self.notice = Some((text.into(), now + seconds));
    }

    /// The text items for a `size` target at `now`; `scale` is the glyph
    /// magnification (from the DPI) and `debug` the fps line.
    pub(super) fn items(&self, now: f64, size: [u32; 2], scale: u32, debug: &str) -> Vec<TextItem> {
        let mut items = Vec::new();
        let line = 8.0 * scale as f32;
        let bottom = size[1] as f32 - MARGIN;
        if self.show_debug {
            items.push(item(debug, [MARGIN, MARGIN], scale, GREY));
        }
        // The name sits at the bottom left, a notice just above it.
        let name_scale = scale * 2;
        let name_y = bottom - 8.0 * name_scale as f32;
        if now < self.name_until {
            if let Some(id) = &self.shown_id {
                items.push(item(display_name(id), [MARGIN, name_y], name_scale, WHITE));
            }
        }
        if let Some((text, until)) = &self.notice {
            if now < *until {
                let y = (name_y - line * 1.5).max(MARGIN);
                items.push(item(text, [MARGIN, y], scale, YELLOW));
            }
        }
        if self.no_audio {
            let width = NO_AUDIO_HINT.len() as f32 * line;
            let x = ((size[0] as f32 - width) / 2.0).max(MARGIN);
            // Above the name band (the name and the notice above it).
            let y = (name_y - line * 3.0).max(MARGIN);
            items.push(item(NO_AUDIO_HINT, [x, y], scale, GREY));
        }
        if self.show_help {
            let top = if self.show_debug {
                MARGIN + line * 2.0
            } else {
                MARGIN
            };
            items.push(item("Milksplat hotkeys", [MARGIN, top], scale, YELLOW));
            let keys_w = HELP.iter().map(|(k, _)| k.len()).max().unwrap_or(0) + 2;
            for (i, (keys, what)) in HELP.iter().enumerate() {
                let y = top + line * (i as f32 + 1.5);
                items.push(item(
                    &format!("{keys:<keys_w$}{what}"),
                    [MARGIN, y],
                    scale,
                    WHITE,
                ));
            }
        }
        items
    }
}

fn item(text: &str, pos: [f32; 2], scale: u32, color: [f32; 4]) -> TextItem {
    TextItem {
        text: text.to_string(),
        pos,
        scale,
        color,
    }
}

/// A preset id as shown: without the `.splat` extension.
pub fn display_name(id: &str) -> &str {
    id.strip_suffix(".splat").unwrap_or(id)
}
