//! The director's state machine internals: commands, requests to the
//! loader and their answers, switches, blends and the history.

use ms_preset::next_duration;

use super::active::{Active, Blend, Loaded};
use super::{Command, Director, HISTORY_CAP, MAX_FAILURES, RETRY_S};
use crate::blend::BlendPattern;
use crate::loader::LoadResult;

/// How a switch to a requested preset affects the history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Hist {
    /// Append it.
    Push,
    /// It is the history entry at this index (a `Prev`).
    Back(usize),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Goal {
    /// The next preset, kept ready for auto-advance, hard cuts and `Next`.
    Prefetch,
    /// Switch to it as soon as it is ready. `repick`: on failure, request
    /// another pick (for `Next`/`HardCut`) instead of giving up.
    Switch { cut: bool, hist: Hist, repick: bool },
}

pub(super) struct Pending {
    /// The loader ticket, or `None` if not sent yet.
    pub(super) req: Option<u64>,
    pub(super) id: String,
    pub(super) seed: u64,
    pub(super) goal: Goal,
}

/// A switch waiting for the current blend to finish.
pub(super) struct Queued {
    pub(super) loaded: Loaded,
    pub(super) cut: bool,
    pub(super) hist: Hist,
}

impl Director {
    /// A display time for a preset becoming current.
    pub(super) fn display_time(&mut self) -> f64 {
        let base = self.cfg.auto_advance_s;
        if base.is_finite() && base > 0.0 {
            next_duration(base, &mut self.rng)
        } else {
            f64::INFINITY
        }
    }

    fn blend_s(&self) -> Option<f64> {
        let s = self.cfg.blend_s;
        (s.is_finite() && s > 0.0).then_some(s)
    }

    /// The id now on screen (the incoming one during a blend).
    fn shown_id(&self) -> &str {
        self.blend.as_ref().map_or(&self.cur.id, |b| &b.to.id)
    }

    pub(super) fn apply(&mut self, c: Command, now: f64) {
        match c {
            Command::ToggleLock => {
                self.locked = !self.locked;
                tracing::info!("preset lock {}", if self.locked { "on" } else { "off" });
            }
            Command::Next | Command::HardCut => {
                let cut = c == Command::HardCut;
                if let Some(next) = self.ready.take() {
                    self.switch(next, cut, Hist::Push, now);
                    return;
                }
                let goal = Goal::Switch {
                    cut,
                    hist: Hist::Push,
                    repick: true,
                };
                match &mut self.pending {
                    // The next preset is on its way: switch when it arrives.
                    Some(p) if p.goal == Goal::Prefetch => p.goal = goal,
                    _ => self.request_pick(goal, now),
                }
            }
            Command::Prev => {
                let Some(i) = self.cursor.filter(|&i| i > 0) else {
                    return;
                };
                if let Some(id) = self.history.get(i - 1).cloned() {
                    self.want(id, Hist::Back(i - 1), now);
                }
            }
            Command::Random => {
                let shown = self.shown_id().to_string();
                let cands: Vec<&str> = self
                    .lib
                    .entries
                    .iter()
                    .map(|e| e.id.as_str())
                    .filter(|id| *id != shown && self.allowed(id) && !self.skip.contains(*id))
                    .collect();
                if cands.is_empty() {
                    return;
                }
                let i = (self.rng.next_u64() % cands.len() as u64) as usize;
                let id = cands[i].to_string();
                self.selector.push_manual(id.clone());
                self.want(id, Hist::Push, now);
            }
            Command::Load(id) => {
                if !self.lib.entries.iter().any(|e| e.id == id) {
                    tracing::warn!("load {id}: no such preset in the library");
                } else if !self.allowed(&id) {
                    tracing::warn!("load {id}: the preset is banned or blocklisted");
                } else {
                    self.selector.push_manual(id.clone());
                    self.want(id, Hist::Push, now);
                }
            }
        }
    }

    fn allowed(&self, id: &str) -> bool {
        !self.prefs.bans.contains(id) && !self.prefs.blocklist.contains(id)
    }

    /// Switches to `id` as soon as it is ready (now, if it is the ready
    /// next preset).
    fn want(&mut self, id: String, hist: Hist, now: f64) {
        if self.ready.as_ref().is_some_and(|l| l.id == id) {
            if let Some(next) = self.ready.take() {
                self.switch(next, false, hist, now);
            }
            return;
        }
        let seed = self.rng.next_u64();
        self.pending = Some(Pending {
            req: None,
            id,
            seed,
            goal: Goal::Switch {
                cut: false,
                hist,
                repick: false,
            },
        });
    }

    /// Whether requests are allowed at `now` (not paused after failures).
    fn can_request(&mut self, now: f64) -> bool {
        if now < self.retry_at {
            return false;
        }
        if self.failures >= MAX_FAILURES {
            self.failures = 0;
        }
        true
    }

    /// Requests a pick from the selector for `goal`; with nothing to pick,
    /// pauses requests for [`RETRY_S`].
    fn request_pick(&mut self, goal: Goal, now: f64) {
        if !self.can_request(now) {
            return;
        }
        let pick = self
            .selector
            .next(&self.lib, &self.pick_prefs)
            .filter(|id| id != self.shown_id());
        match pick {
            Some(id) => {
                let seed = self.rng.next_u64();
                self.pending = Some(Pending {
                    req: None,
                    id,
                    seed,
                    goal,
                });
            }
            None => {
                self.pending = None;
                self.retry_at = now + RETRY_S;
            }
        }
    }

    /// Requests the next preset if none is ready or on its way.
    pub(super) fn prefetch(&mut self, now: f64) {
        if self.pending.is_none() && self.ready.is_none() && !self.lib.entries.is_empty() {
            self.request_pick(Goal::Prefetch, now);
        }
    }

    pub(super) fn on_result(&mut self, r: LoadResult, now: f64) {
        let current = self
            .pending
            .as_ref()
            .is_some_and(|p| p.req == Some(r.req()) && p.id == r.id());
        if !current {
            tracing::debug!(
                "ignoring the answer to superseded request {} ({})",
                r.req(),
                r.id()
            );
            return;
        }
        let Some(pending) = self.pending.take() else {
            return;
        };
        match r {
            LoadResult::Ready {
                id,
                preset,
                shaders,
                warnings,
                ..
            } => {
                self.failures = 0;
                for w in warnings.iter().chain(&shaders.errors) {
                    tracing::warn!("{id}: {w}");
                }
                tracing::info!("preset ready: {id}");
                let loaded = Loaded {
                    id,
                    preset,
                    shaders,
                    seed: pending.seed,
                };
                match pending.goal {
                    Goal::Prefetch => self.ready = Some(loaded),
                    Goal::Switch { cut, hist, .. } => self.switch(loaded, cut, hist, now),
                }
            }
            LoadResult::Failed { id, error, .. } => {
                tracing::warn!("preset {id} failed to load, skipping it: {error}");
                self.skip.insert(id.clone());
                self.pick_prefs.blocklist.insert(id);
                self.failures += 1;
                if self.failures >= MAX_FAILURES {
                    tracing::warn!(
                        "{MAX_FAILURES} presets in a row failed to load; trying again in {RETRY_S} s"
                    );
                    self.retry_at = now + RETRY_S;
                }
                if let Goal::Switch { repick: true, .. } = pending.goal {
                    self.request_pick(pending.goal, now);
                }
            }
        }
    }

    /// Makes `next` current: a blend, or an instant switch for a cut or
    /// without a blend duration. During a blend the switch is queued until
    /// the blend finishes (a newer queued switch replaces an older one).
    pub(super) fn switch(&mut self, next: Loaded, cut: bool, hist: Hist, now: f64) {
        if self.blend.is_some() {
            self.queued = Some(Queued {
                loaded: next,
                cut,
                hist,
            });
            return;
        }
        tracing::info!(
            "{} {} -> {}",
            if cut { "cut" } else { "switch" },
            self.cur.id,
            next.id
        );
        if self.ready.as_ref().is_some_and(|l| l.id == next.id) {
            self.ready = None;
        }
        self.record(&next.id, hist);
        let duration = self.display_time();
        let to = Active::new(next, now, duration, self.mesh_cap);
        match self.blend_s().filter(|_| !cut) {
            None => self.cur = to,
            Some(dur) => {
                let pattern = BlendPattern::random(&mut self.rng);
                self.blend = Some(Blend {
                    to,
                    start: now,
                    dur,
                    pattern,
                });
            }
        }
    }

    fn record(&mut self, id: &str, hist: Hist) {
        match hist {
            Hist::Back(i) if self.history.get(i).is_some_and(|h| h == id) => {
                self.cursor = Some(i);
            }
            _ => {
                self.history.push(id.to_string());
                if self.history.len() > HISTORY_CAP {
                    self.history.remove(0);
                }
                self.cursor = Some(self.history.len() - 1);
            }
        }
    }

    pub(super) fn finish_blend(&mut self, now: f64) {
        let done = self.blend.as_ref().is_some_and(|b| now >= b.start + b.dur);
        if !done {
            return;
        }
        if let Some(b) = self.blend.take() {
            self.cur = b.to;
        }
        if let Some(q) = self.queued.take() {
            self.switch(q.loaded, q.cut, q.hist, now);
        }
    }
}
