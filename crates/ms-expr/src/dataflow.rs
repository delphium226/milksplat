//! Carried-variable analysis over bytecode.
//!
//! A forward "definitely written" dataflow over basic blocks: the set of
//! variables every path from the program entry has stored to. A `LoadVar` of a
//! variable outside that set reads a value left over from before the run, so
//! the variable carries state between runs (between mesh points, when a
//! per-vertex program is run once per point).

use crate::bytecode::Op;
use crate::vars::VarId;

/// Fixed-size bit set over variable indices.
#[derive(Clone, PartialEq)]
struct Bits(Vec<u64>);

impl Bits {
    fn empty(n: usize) -> Self {
        Bits(vec![0; n.div_ceil(64)])
    }
    fn full(n: usize) -> Self {
        Bits(vec![!0; n.div_ceil(64)])
    }
    fn has(&self, i: usize) -> bool {
        self.0.get(i / 64).is_some_and(|w| w & (1 << (i % 64)) != 0)
    }
    fn insert(&mut self, i: usize) {
        if let Some(w) = self.0.get_mut(i / 64) {
            *w |= 1 << (i % 64);
        }
    }
    fn intersect(&mut self, other: &Bits) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a &= *b;
        }
    }
}

/// Fall-through and jump successors of the op at `pc` (may be `ops.len()`,
/// meaning the program ends).
fn successors(op: &Op, pc: usize) -> (Option<usize>, Option<usize>) {
    match *op {
        Op::Jump { target } => (None, Some(target as usize)),
        Op::JumpIfFalse { target, .. } | Op::JumpIfTrue { target, .. } => {
            (Some(pc + 1), Some(target as usize))
        }
        Op::LoopNext { exit, .. } | Op::LoopGuard { exit } => (Some(pc + 1), Some(exit as usize)),
        _ => (Some(pc + 1), None),
    }
}

/// Variables some execution path reads before writing, sorted by id and
/// deduplicated.
pub(crate) fn reads_before_write(ops: &[Op]) -> Vec<VarId> {
    let len = ops.len();
    if len == 0 {
        return Vec::new();
    }
    let nvars = ops
        .iter()
        .filter_map(|op| match *op {
            Op::LoadVar { var, .. } | Op::StoreVar { var, .. } => Some(var.0 as usize + 1),
            _ => None,
        })
        .max()
        .unwrap_or(0);

    // Basic blocks: leaders are the entry, every jump target and every op
    // following a jump.
    let mut leader = vec![false; len];
    leader[0] = true;
    for (pc, op) in ops.iter().enumerate() {
        if let (_, Some(t)) = successors(op, pc) {
            if let Some(l) = leader.get_mut(t) {
                *l = true;
            }
            if let Some(l) = leader.get_mut(pc + 1) {
                *l = true;
            }
        }
    }
    let starts: Vec<usize> = (0..len).filter(|&pc| leader[pc]).collect();
    let mut block_of = vec![0usize; len];
    for (b, w) in starts.iter().enumerate() {
        let end = starts.get(b + 1).copied().unwrap_or(len);
        for slot in &mut block_of[*w..end] {
            *slot = b;
        }
    }
    let nblocks = starts.len();
    let block_range = |b: usize| starts[b]..starts.get(b + 1).copied().unwrap_or(len);
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); nblocks];
    for b in 0..nblocks {
        let last = block_range(b).end - 1;
        let (fall, jump) = successors(&ops[last], last);
        for s in [fall, jump].into_iter().flatten() {
            if s < len {
                preds[block_of[s]].push(b);
            }
        }
    }

    // `out` of each block = `in` plus its stores. Unvisited blocks start at
    // "everything written" (the top of the intersection lattice).
    let gen: Vec<Bits> = (0..nblocks)
        .map(|b| {
            let mut g = Bits::empty(nvars);
            for op in &ops[block_range(b)] {
                if let Op::StoreVar { var, .. } = *op {
                    g.insert(var.0 as usize);
                }
            }
            g
        })
        .collect();
    let mut ins = vec![Bits::full(nvars); nblocks];
    ins[0] = Bits::empty(nvars);
    let mut outs = vec![Bits::full(nvars); nblocks];
    let mut changed = true;
    while changed {
        changed = false;
        for b in 0..nblocks {
            let mut inb = if b == 0 {
                Bits::empty(nvars)
            } else {
                Bits::full(nvars)
            };
            for &p in &preds[b] {
                inb.intersect(&outs[p]);
            }
            let mut outb = inb.clone();
            for (o, g) in outb.0.iter_mut().zip(&gen[b].0) {
                *o |= *g;
            }
            if outb != outs[b] || inb != ins[b] {
                ins[b] = inb;
                outs[b] = outb;
                changed = true;
            }
        }
    }

    let mut read = Bits::empty(nvars);
    for (b, inb) in ins.iter().enumerate() {
        let mut cur = inb.clone();
        for op in &ops[block_range(b)] {
            match *op {
                Op::LoadVar { var, .. } if !cur.has(var.0 as usize) => read.insert(var.0 as usize),
                Op::StoreVar { var, .. } => cur.insert(var.0 as usize),
                _ => {}
            }
        }
    }
    (0..nvars)
        .filter(|&v| read.has(v))
        .map(|v| VarId(v as u32))
        .collect()
}
