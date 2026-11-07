//! Variable name interning.

use std::collections::HashMap;

/// Index of a variable in the caller's `vars` slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(pub u32);

/// Append-only, case-insensitive table of variable names.
#[derive(Debug, Clone, Default)]
pub struct VarTable {
    names: Vec<String>,
    index: HashMap<String, u32>,
}

impl VarTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the id for `name`, registering it if new.
    pub fn get_or_insert(&mut self, name: &str) -> VarId {
        let lower = name.to_lowercase();
        if let Some(&i) = self.index.get(&lower) {
            return VarId(i);
        }
        let i = self.names.len() as u32;
        self.names.push(lower.clone());
        self.index.insert(lower, i);
        VarId(i)
    }

    pub fn lookup(&self, name: &str) -> Option<VarId> {
        self.index.get(&name.to_lowercase()).map(|&i| VarId(i))
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Lowercased name of `id`; empty string if `id` is out of range.
    pub fn name(&self, id: VarId) -> &str {
        self.names.get(id.0 as usize).map(String::as_str).unwrap_or("")
    }
}
