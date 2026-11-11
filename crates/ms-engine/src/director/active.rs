//! The presets the director holds: loaded ones waiting to be shown, and
//! the ones on screen (with their instance and latest uniforms).

use std::sync::Arc;

use ms_analysis::AudioFrame;

use ms_preset::CompiledPreset;
use ms_render::{
    FeedbackFormat, PresetPicks, RendererShared, ShaderSet, ShaderSources, TextureResolver,
    Uniforms,
};

use super::TickInputBase;
use crate::blend::BlendPattern;
use crate::env::{Globals, TickInput};
use crate::loader::{feedback_format, PICKS_SALT};
use crate::reactivity::Reactivity;
use crate::tick::{PresetInstance, TickOutput};

/// A preset's shaders: compiled on the current device, or, after
/// `Director::release_gpu`, only what is needed to compile them again (no
/// GPU object, so nothing keeps a lost device alive).
pub(super) enum Shaders {
    Compiled(ShaderSet),
    Released(ShaderSources),
}

impl Shaders {
    /// The compiled set, unless released.
    pub(super) fn compiled(&self) -> Option<&ShaderSet> {
        match self {
            Shaders::Compiled(set) => Some(set),
            Shaders::Released(_) => None,
        }
    }

    pub(super) fn blur_levels(&self) -> u8 {
        match self {
            Shaders::Compiled(set) => set.blur_levels,
            Shaders::Released(src) => src.blur_levels,
        }
    }

    /// Drops the GPU objects, keeping the sources.
    pub(super) fn release(&mut self) {
        let old = std::mem::replace(self, Shaders::Released(ShaderSources::default()));
        *self = match old {
            Shaders::Compiled(set) => Shaders::Released(set.into_sources()),
            released => released,
        };
    }

    /// Compiles again with `shared` for `comp_format`, from the old set or
    /// the kept sources; user textures come from the old images first, then
    /// from `resolver` (with the preset's `randNN` picks for `seed`).
    pub(super) fn rebuild(
        &mut self,
        shared: &RendererShared,
        resolver: &TextureResolver,
        comp_format: wgpu::TextureFormat,
        seed: u64,
    ) {
        let picks = std::cell::RefCell::new(PresetPicks::new(seed ^ PICKS_SALT));
        let tex = |name: &str| resolver.resolve_with(name, &mut picks.borrow_mut());
        let set = match self {
            Shaders::Compiled(old) => shared.recompile(old, comp_format, &tex),
            Shaders::Released(src) => shared.compile_sources(src, comp_format, &tex),
        };
        *self = Shaders::Compiled(set);
    }

    pub(super) fn is_released(&self) -> bool {
        matches!(self, Shaders::Released(_))
    }

    /// A copy (compiled GPU objects are shared, not duplicated).
    pub(super) fn duplicate(&self) -> Shaders {
        match self {
            Shaders::Compiled(set) => Shaders::Compiled(clone_set(set)),
            Shaders::Released(src) => Shaders::Released(src.clone()),
        }
    }
}

/// A loaded preset, not yet on screen.
pub(super) struct Loaded {
    pub(super) id: String,
    pub(super) preset: Arc<CompiledPreset>,
    pub(super) shaders: Shaders,
    pub(super) seed: u64,
}

/// A preset on screen.
pub(super) struct Active {
    pub(super) id: String,
    pub(super) preset: Arc<CompiledPreset>,
    pub(super) shaders: Shaders,
    pub(super) seed: u64,
    pub(super) instance: PresetInstance,
    /// When it became current.
    pub(super) since: f64,
    /// Its display time (infinite without auto-advance).
    pub(super) duration: f64,
    pub(super) frames: u64,
    /// The uniforms of its latest tick, for the display-rate composite.
    pub(super) uniforms: Uniforms,
    /// Its reactivity, and the audio scaled by it (see
    /// [`prepare_audio`](Self::prepare_audio)).
    pub(super) reactivity: Reactivity,
}

impl Active {
    pub(super) fn new(l: Loaded, since: f64, duration: f64, mesh_cap: [u32; 2]) -> Self {
        Active {
            instance: PresetInstance::new(Arc::clone(&l.preset), l.seed, mesh_cap),
            id: l.id,
            preset: l.preset,
            shaders: l.shaders,
            seed: l.seed,
            since,
            duration,
            frames: 0,
            uniforms: Uniforms::default(),
            reactivity: Reactivity::new(1.0),
        }
    }

    fn time(&self, now: f64) -> f64 {
        (now - self.since).max(0.0)
    }

    /// Scales the tick's audio by reactivity `level` for the next
    /// [`tick`](Self::tick); returns the scaled frame.
    pub(super) fn prepare_audio<'a>(
        &'a mut self,
        audio: &'a AudioFrame,
        level: f32,
    ) -> &'a AudioFrame {
        self.reactivity.set(level);
        self.reactivity.apply(audio)
    }

    /// Ticks the instance on the audio [`prepare_audio`](Self::prepare_audio)
    /// scaled; returns its output and shaders.
    pub(super) fn tick(
        &mut self,
        now: f64,
        base: &TickInputBase,
        g: &mut Globals,
    ) -> (&TickOutput, &Shaders) {
        let time = self.time(now);
        let progress = if self.duration.is_finite() && self.duration > 0.0 {
            (time / self.duration).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let input = TickInput {
            time,
            fps: base.fps,
            frame: self.frames,
            progress,
            audio: self.reactivity.applied(base.audio),
            aspect: base.aspect,
            pixels: base.pixels,
        };
        self.frames += 1;
        let out = self.instance.tick(&input, g);
        self.uniforms = out.uniforms;
        (out, &self.shaders)
    }

    /// The latest uniforms with the exact (not tick-quantized) time.
    pub(super) fn display_uniforms(&self, now: f64) -> Uniforms {
        let t = self.time(now) as f32;
        Uniforms {
            time: if t.is_finite() { t } else { 0.0 },
            ..self.uniforms
        }
    }

    /// The preset's simulation rate (`[params] sim_rate`, 1..=240 Hz).
    pub(super) fn sim_rate(&self) -> f64 {
        self.preset.source.params.sim_rate
    }

    pub(super) fn feedback_format(&self) -> FeedbackFormat {
        feedback_format(self.preset.source.params.feedback_format)
    }
}

pub(super) struct Blend {
    pub(super) to: Active,
    pub(super) start: f64,
    pub(super) dur: f64,
    pub(super) pattern: BlendPattern,
}

impl Blend {
    pub(super) fn progress(&self, now: f64) -> f32 {
        let p = ((now - self.start) / self.dur) as f32;
        if p.is_finite() {
            p.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// The default preset's shaders (the built-ins, unless the file has some).
pub(super) fn compile_default(
    shared: &RendererShared,
    comp_format: wgpu::TextureFormat,
    p: &CompiledPreset,
) -> ShaderSet {
    let s = &p.source.shader;
    shared.compile_shaders(s.warp.as_deref(), s.comp.as_deref(), comp_format, &|_| None)
}

/// A copy of `s` (the GPU objects are shared, not duplicated).
fn clone_set(s: &ShaderSet) -> ShaderSet {
    ShaderSet {
        warp: s.warp.clone(),
        comp: s.comp.clone(),
        errors: s.errors.clone(),
        blur_levels: s.blur_levels,
        user_textures: s.user_textures.clone(),
        warp_source: s.warp_source.clone(),
        comp_source: s.comp_source.clone(),
        texture_images: s.texture_images.clone(),
    }
}
