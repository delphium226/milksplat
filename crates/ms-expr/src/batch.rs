//! Batched (structure-of-arrays) interpreter: one program over many lanes.
//!
//! Lanes are processed in blocks of [`BLOCK`]. Within a block each opcode
//! runs across all active lanes before the next opcode. Control flow uses
//! lane masks scheduled by a work-list keyed by pc:
//!
//! - A *group* is `(pc, mask)`: the lanes in `mask` are all about to execute
//!   the op at `pc`.
//! - The scheduler always resumes the group with the lowest pc. It runs
//!   straight-line ops for the whole mask until it reaches a control op
//!   (jump, conditional jump, loop head). There the mask is split by each
//!   lane's own outcome and the parts are put back on the work-list; a part
//!   landing on a pc that already has a group is merged into it.
//! - The block is done when the work-list is empty (every lane ran off the
//!   end of the program).
//!
//! Correctness: lanes never share state (registers, variables, loop
//! counters are all per lane), so any interleaving in which every lane
//! executes exactly its own scalar op sequence produces the scalar results.
//! Each lane's path is decided only by its own values, and an op applied to a
//! mask touches only the lanes in that mask, so this holds for every
//! schedule, including backward jumps. Lowest-pc-first scheduling only adds
//! reconvergence: lanes that left an `if` arm early wait at the join point
//! (a higher pc) until the other arm's lanes arrive, and lanes that left a
//! loop wait at its exit until the remaining lanes finish iterating.

use crate::ast::{BinOp, UnOp};
use crate::builtins::{call1, call2, Fn1, Fn2};
use crate::bytecode::Op;
use crate::compile::Program;
use crate::mem::Megabuf;
use crate::ops::{binary, truthy, unary};
use crate::rng::Rng;
use crate::vars::VarId;
use crate::vm::{start_iteration, ExecEnv, LoopBudget};

/// Lanes per block.
pub const BLOCK: usize = 256;

/// Per-lane variable storage: `cols[var.0]` holds that variable for each of
/// the `n` lanes.
#[derive(Debug, Clone, Default)]
pub struct Lanes {
    pub n: usize,
    pub cols: Vec<Vec<f64>>,
}

/// State shared by all lanes. Batchable programs never touch it; it is used
/// by the lane-by-lane fallback for non-batchable programs.
pub struct BatchShared<'a> {
    pub megabuf: &'a mut Megabuf,
    pub gmegabuf: &'a mut Megabuf,
    pub rng: &'a mut Rng,
}

type Mask = [u64; BLOCK / 64];

const EMPTY: Mask = [0; BLOCK / 64];

fn first_lanes(len: usize) -> Mask {
    let mut m = EMPTY;
    for (w, word) in m.iter_mut().enumerate() {
        let lo = w * 64;
        if len >= lo + 64 {
            *word = !0;
        } else if len > lo {
            *word = (1u64 << (len - lo)) - 1;
        }
    }
    m
}

fn is_empty(m: &Mask) -> bool {
    m.iter().all(|&w| w == 0)
}

/// Calls `f(lane)` for every lane in `mask`; `full` means lanes `0..len`.
#[inline(always)]
fn each(mask: &Mask, full: bool, len: usize, mut f: impl FnMut(usize)) {
    if full {
        for i in 0..len {
            f(i);
        }
    } else {
        for (w, &bits) in mask.iter().enumerate() {
            let mut b = bits;
            while b != 0 {
                f(w * 64 + b.trailing_zeros() as usize);
                b &= b - 1;
            }
        }
    }
}

/// Splits `mask` into (lanes where `pred` holds, lanes where it does not).
#[inline(always)]
fn split(mask: &Mask, mut pred: impl FnMut(usize) -> bool) -> (Mask, Mask) {
    let (mut yes, mut no) = (EMPTY, EMPTY);
    for (w, &bits) in mask.iter().enumerate() {
        let mut b = bits;
        while b != 0 {
            let t = b.trailing_zeros();
            if pred(w * 64 + t as usize) {
                yes[w] |= 1 << t;
            } else {
                no[w] |= 1 << t;
            }
            b &= b - 1;
        }
    }
    (yes, no)
}

/// Adds `mask` at `pc` to the work-list, merging with an existing group.
fn push(work: &mut Vec<(usize, Mask)>, pc: usize, mask: Mask) {
    if is_empty(&mask) {
        return;
    }
    if let Some((_, m)) = work.iter_mut().find(|(p, _)| *p == pc) {
        for (a, b) in m.iter_mut().zip(mask) {
            *a |= b;
        }
    } else {
        work.push((pc, mask));
    }
}

/// Per-block scratch: registers (`regs[r * BLOCK + lane]`) and per-lane loop
/// cap state, exactly as one scalar run per lane would have.
struct Block {
    regs: Vec<f64>,
    iters: [u64; BLOCK],
    capped: [bool; BLOCK],
}

/// Heap buffers [`Program::run_batched_with`] reuses between calls, so that
/// running the same program again allocates nothing.
#[derive(Debug, Default)]
pub struct BatchScratch {
    regs: Vec<f64>,
    work: Vec<(usize, Mask)>,
}

impl Program {
    /// Variables some execution path reads before writing them (sorted by
    /// id). If any of these is also written, running the program over mesh
    /// points carries values from one point to the next.
    pub fn reads_before_write(&self) -> &[VarId] {
        &self.rbw
    }

    /// False if the program uses `rand`, `megabuf` or `gmegabuf`, whose
    /// results depend on the order lanes run in.
    pub fn batchable(&self) -> bool {
        !self.ops.iter().any(|op| {
            matches!(
                op,
                Op::Rand { .. } | Op::MegaGet { .. } | Op::MegaSet { .. }
            )
        })
    }

    /// Runs the program once per lane, as if the scalar VM ran each lane
    /// alone (fresh registers and loop-cap counter per lane). Returns true if
    /// any lane hit the loop cap.
    ///
    /// Columns that are missing or shorter than `n` read as 0 and drop
    /// writes. A non-batchable program falls back to running the scalar VM
    /// lane by lane in order.
    pub fn run_batched(&self, lanes: &mut Lanes, shared: &mut BatchShared) -> bool {
        self.run_batched_with(lanes, shared, &mut BatchScratch::default())
    }

    /// [`run_batched`](Self::run_batched), reusing `scratch`'s buffers: once
    /// they have grown to this program's size, a batchable program runs
    /// without allocating.
    pub fn run_batched_with(
        &self,
        lanes: &mut Lanes,
        shared: &mut BatchShared,
        scratch: &mut BatchScratch,
    ) -> bool {
        self.run_batched_budget(lanes, shared, scratch, &mut LoopBudget::unlimited())
    }

    /// [`run_batched_with`](Self::run_batched_with), where every lane's loop
    /// iterations also take from `budget` (charged with the total over all
    /// lanes; see [`LoopBudget`]). Once it is spent, every lane's loops
    /// exit. The return value reports the per-lane cap only.
    pub fn run_batched_budget(
        &self,
        lanes: &mut Lanes,
        shared: &mut BatchShared,
        scratch: &mut BatchScratch,
        budget: &mut LoopBudget,
    ) -> bool {
        debug_assert!(
            self.batchable(),
            "run_batched on a program that is not batchable"
        );
        let mut nregs = 1;
        let mut nvars = 0;
        for op in &self.ops {
            for r in regs_of(op).into_iter().flatten() {
                nregs = nregs.max(r as usize + 1);
            }
            if let Op::LoadVar { var, .. } | Op::StoreVar { var, .. } = *op {
                nvars = nvars.max(var.0 as usize + 1);
            }
        }
        debug_assert!(
            lanes.cols.len() >= nvars && lanes.cols.iter().take(nvars).all(|c| c.len() >= lanes.n),
            "Lanes: need {nvars} columns of length {}",
            lanes.n
        );
        if !self.batchable() {
            return self.run_lanes_scalar(lanes, shared, budget);
        }
        let mut regs = std::mem::take(&mut scratch.regs);
        regs.clear();
        regs.resize(nregs * BLOCK, 0.0);
        let mut blk = Block {
            regs,
            iters: [0; BLOCK],
            capped: [false; BLOCK],
        };
        let mut any_capped = false;
        let mut base = 0;
        while base < lanes.n {
            let len = (lanes.n - base).min(BLOCK);
            blk.regs.fill(0.0);
            blk.iters = [0; BLOCK];
            blk.capped = [false; BLOCK];
            self.run_block(
                &mut blk,
                &mut scratch.work,
                &mut lanes.cols,
                base,
                len,
                budget,
            );
            any_capped |= blk.capped[..len].iter().any(|&c| c);
            base += len;
        }
        scratch.regs = blk.regs;
        any_capped
    }

    fn run_block(
        &self,
        blk: &mut Block,
        work: &mut Vec<(usize, Mask)>,
        cols: &mut [Vec<f64>],
        base: usize,
        len: usize,
        budget: &mut LoopBudget,
    ) {
        let ops = &self.ops[..];
        let all = first_lanes(len);
        work.clear();
        work.push((0usize, all));
        let regs = &mut blk.regs;
        let r = |reg: u16| reg as usize * BLOCK;
        while let Some(k) = work
            .iter()
            .enumerate()
            .min_by_key(|(_, (pc, _))| *pc)
            .map(|(k, _)| k)
        {
            let (mut pc, mask) = work.swap_remove(k);
            let full = mask == all;
            let m = &mask;
            while let Some(op) = ops.get(pc) {
                match *op {
                    Op::Const { dst, value } => {
                        let d = r(dst);
                        each(m, full, len, |i| regs[d + i] = value);
                    }
                    Op::LoadVar { dst, var } => {
                        let d = r(dst);
                        match cols.get(var.0 as usize) {
                            Some(c) => each(m, full, len, |i| {
                                regs[d + i] = c.get(base + i).copied().unwrap_or(0.0)
                            }),
                            None => each(m, full, len, |i| regs[d + i] = 0.0),
                        }
                    }
                    Op::StoreVar { var, src } => {
                        let s = r(src);
                        if let Some(c) = cols.get_mut(var.0 as usize) {
                            each(m, full, len, |i| {
                                if let Some(slot) = c.get_mut(base + i) {
                                    *slot = regs[s + i];
                                }
                            });
                        }
                    }
                    Op::Un { op, dst, a } => {
                        let (d, a) = (r(dst), r(a));
                        macro_rules! per_op {
                            ($($o:ident),*) => {
                                match op {
                                    $(UnOp::$o => each(m, full, len, |i| {
                                        regs[d + i] = unary(UnOp::$o, regs[a + i])
                                    }),)*
                                }
                            };
                        }
                        per_op!(Neg, Plus, Not);
                    }
                    Op::Bin { op, dst, a, b } => {
                        let (d, a, b) = (r(dst), r(a), r(b));
                        // One loop per operator so the per-lane body has no
                        // operator dispatch.
                        macro_rules! per_op {
                            ($($o:ident),*) => {
                                match op {
                                    $(BinOp::$o => each(m, full, len, |i| {
                                        regs[d + i] = binary(BinOp::$o, regs[a + i], regs[b + i])
                                    }),)*
                                }
                            };
                        }
                        per_op!(
                            Add, Sub, Mul, Div, Mod, Pow, Eq, Ne, Lt, Le, Gt, Ge, BitAnd, BitOr,
                            LogAnd, LogOr
                        );
                    }
                    Op::Truthy { dst, a } => {
                        let (d, a) = (r(dst), r(a));
                        each(m, full, len, |i| {
                            regs[d + i] = if truthy(regs[a + i]) { 1.0 } else { 0.0 }
                        });
                    }
                    Op::Mov { dst, src } => {
                        let (d, s) = (r(dst), r(src));
                        each(m, full, len, |i| regs[d + i] = regs[s + i]);
                    }
                    Op::Call1 { f, dst, a } => {
                        let (d, a) = (r(dst), r(a));
                        macro_rules! per_fn {
                            ($($o:ident),*) => {
                                match f {
                                    $(Fn1::$o => each(m, full, len, |i| {
                                        regs[d + i] = call1(Fn1::$o, regs[a + i])
                                    }),)*
                                }
                            };
                        }
                        per_fn!(
                            Sin, Cos, Tan, Asin, Acos, Atan, Sqrt, Exp, Log, Log10, Abs, Sign,
                            Floor, Ceil, Sqr
                        );
                    }
                    Op::Call2 { f, dst, a, b } => {
                        let (d, a, b) = (r(dst), r(a), r(b));
                        macro_rules! per_fn {
                            ($($o:ident),*) => {
                                match f {
                                    $(Fn2::$o => each(m, full, len, |i| {
                                        regs[d + i] = call2(Fn2::$o, regs[a + i], regs[b + i])
                                    }),)*
                                }
                            };
                        }
                        per_fn!(Atan2, Pow, Min, Max, Sigmoid);
                    }
                    Op::LoopInit { cnt, n } => {
                        let (c, n) = (r(cnt), r(n));
                        each(m, full, len, |i| {
                            let v = regs[n + i];
                            regs[c + i] = if v >= 1.0 { v.floor() } else { 0.0 };
                        });
                    }
                    // Never reached: `run_batched` only gets here for
                    // batchable programs.
                    Op::Rand { .. } | Op::MegaGet { .. } | Op::MegaSet { .. } => {}
                    Op::Jump { target } => {
                        push(work, target as usize, mask);
                        break;
                    }
                    Op::JumpIfFalse { cond, target } => {
                        let c = r(cond);
                        let (go, stay) = split(m, |i| !truthy(regs[c + i]));
                        push(work, target as usize, go);
                        push(work, pc + 1, stay);
                        break;
                    }
                    Op::JumpIfTrue { cond, target } => {
                        let c = r(cond);
                        let (go, stay) = split(m, |i| truthy(regs[c + i]));
                        push(work, target as usize, go);
                        push(work, pc + 1, stay);
                        break;
                    }
                    Op::LoopNext { cnt, exit } => {
                        let c = r(cnt);
                        let (iters, capped) = (&mut blk.iters, &mut blk.capped);
                        let (body, out) = split(m, |i| {
                            let left = regs[c + i];
                            if left >= 1.0 && start_iteration(&mut iters[i], &mut capped[i], budget)
                            {
                                regs[c + i] = left - 1.0;
                                true
                            } else {
                                false
                            }
                        });
                        push(work, pc + 1, body);
                        push(work, exit as usize, out);
                        break;
                    }
                    Op::LoopGuard { exit } => {
                        let (iters, capped) = (&mut blk.iters, &mut blk.capped);
                        let (body, out) = split(m, |i| {
                            start_iteration(&mut iters[i], &mut capped[i], budget)
                        });
                        push(work, pc + 1, body);
                        push(work, exit as usize, out);
                        break;
                    }
                }
                pc += 1;
            }
        }
    }

    /// Fallback for non-batchable programs: the scalar VM, lane by lane in
    /// order, sharing `shared` between lanes.
    fn run_lanes_scalar(
        &self,
        lanes: &mut Lanes,
        shared: &mut BatchShared,
        budget: &mut LoopBudget,
    ) -> bool {
        let mut vars = vec![0.0; lanes.cols.len().max(self.num_vars)];
        let mut capped = false;
        for k in 0..lanes.n {
            vars.fill(0.0);
            for (v, c) in vars.iter_mut().zip(&lanes.cols) {
                *v = c.get(k).copied().unwrap_or(0.0);
            }
            let mut env = ExecEnv {
                vars: &mut vars,
                megabuf: shared.megabuf,
                gmegabuf: shared.gmegabuf,
                rng: shared.rng,
            };
            capped |= self.run_with_budget(&mut env, budget).loop_cap_hit;
            for (v, c) in vars.iter().zip(lanes.cols.iter_mut()) {
                if let Some(slot) = c.get_mut(k) {
                    *slot = *v;
                }
            }
        }
        capped
    }
}

/// Registers an op reads or writes.
fn regs_of(op: &Op) -> [Option<u16>; 3] {
    match *op {
        Op::Const { dst, .. } | Op::LoadVar { dst, .. } => [Some(dst), None, None],
        Op::StoreVar { src, .. } => [Some(src), None, None],
        Op::Un { dst, a, .. } | Op::Truthy { dst, a } | Op::Call1 { dst, a, .. } => {
            [Some(dst), Some(a), None]
        }
        Op::Rand { dst, a } => [Some(dst), Some(a), None],
        Op::Bin { dst, a, b, .. } | Op::Call2 { dst, a, b, .. } => [Some(dst), Some(a), Some(b)],
        Op::JumpIfFalse { cond, .. } | Op::JumpIfTrue { cond, .. } => [Some(cond), None, None],
        Op::Jump { .. } | Op::LoopGuard { .. } => [None; 3],
        Op::Mov { dst, src } => [Some(dst), Some(src), None],
        Op::MegaGet { dst, idx, .. } => [Some(dst), Some(idx), None],
        Op::MegaSet { idx, src, .. } => [Some(idx), Some(src), None],
        Op::LoopInit { cnt, n } => [Some(cnt), Some(n), None],
        Op::LoopNext { cnt, .. } => [Some(cnt), None, None],
    }
}
