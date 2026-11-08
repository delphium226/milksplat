//! Preset selection (spec section 7.3).

use crate::library::Library;
use ms_expr::Rng;
use std::collections::HashSet;

const HISTORY_CAP: usize = 50;

#[derive(Debug, Clone, Default)]
pub struct Prefs {
    pub favourites: HashSet<String>,
    pub bans: HashSet<String>,
    pub blocklist: HashSet<String>,
}

pub struct Selector {
    rng: Rng,
    history: Vec<String>,
    /// Index into `history` of the current preset.
    cursor: usize,
}

impl Selector {
    pub fn new(seed: u64) -> Self {
        Selector {
            rng: Rng::seeded(seed),
            history: Vec::new(),
            cursor: 0,
        }
    }

    fn push(&mut self, id: String) {
        self.history.push(id);
        if self.history.len() > HISTORY_CAP {
            self.history.remove(0);
        }
        self.cursor = self.history.len() - 1;
    }

    /// Pick the next preset id and append it to the history.
    pub fn next(&mut self, lib: &Library, prefs: &Prefs) -> Option<String> {
        let cands: Vec<&str> = lib
            .entries
            .iter()
            .map(|e| e.id.as_str())
            .filter(|id| !prefs.bans.contains(*id) && !prefs.blocklist.contains(*id))
            .collect();
        if cands.is_empty() {
            return None;
        }
        let k = (cands.len() * 3).div_ceil(10);
        let recent: HashSet<&str> = self
            .history
            .iter()
            .rev()
            .take(k)
            .map(String::as_str)
            .collect();
        let fresh: Vec<&str> = cands
            .iter()
            .copied()
            .filter(|id| !recent.contains(id))
            .collect();
        let pool = if fresh.is_empty() { &cands } else { &fresh };

        let weight = |id: &str| {
            if prefs.favourites.contains(id) {
                2.0
            } else {
                1.0
            }
        };
        let total: f64 = pool.iter().map(|id| weight(id)).sum();
        let mut r = self.rng.next_f64() * total;
        let mut pick = pool[pool.len() - 1];
        for id in pool.iter() {
            let w = weight(id);
            if r < w {
                pick = id;
                break;
            }
            r -= w;
        }
        let pick = pick.to_string();
        self.push(pick.clone());
        Some(pick)
    }

    /// Step back to the entry before the current one.
    pub fn prev(&mut self) -> Option<String> {
        if self.cursor == 0 || self.history.is_empty() {
            return None;
        }
        self.cursor -= 1;
        self.history.get(self.cursor).cloned()
    }

    /// Record a manually chosen preset as the newest history entry.
    pub fn push_manual(&mut self, id: String) {
        self.push(id);
    }
}

/// Display time for the next preset: `base_s` +/- 25%.
pub fn next_duration(base_s: f64, rng: &mut Rng) -> f64 {
    base_s * (0.75 + 0.5 * rng.next_f64())
}
