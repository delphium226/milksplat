//! Scalar interpreter over [`Program`] bytecode.

use crate::builtins::{call1, call2, rand_scaled};
use crate::bytecode::Op;
use crate::compile::Program;
use crate::mem::Megabuf;
use crate::ops::{binary, truthy, unary};
use crate::rng::Rng;
use crate::LOOP_CAP;

/// Mutable state a program runs against.
pub struct ExecEnv<'a> {
    pub vars: &'a mut [f64],
    pub megabuf: &'a mut Megabuf,
    pub gmegabuf: &'a mut Megabuf,
    pub rng: &'a mut Rng,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunResult {
    pub value: f64,
    pub loop_cap_hit: bool,
}

const STACK_REGS: usize = 32;

/// A loop-iteration allowance shared by several runs.
///
/// Every loop iteration started by [`Program::run_with_budget`] (and the
/// batched equivalent) takes one iteration from the budget, on top of the
/// per-run [`LOOP_CAP`]. Once a loop is refused an iteration the budget is
/// *exhausted*: every later loop sharing it exits at once, without running
/// its body. The engine gives each preset one budget of
/// [`TICK_LOOP_BUDGET`] per tick (spec §5.1's cap applied per tick), so no
/// preset can make a tick do unbounded loop work however many vertices,
/// wave points or shape instances run code.
///
/// [`TICK_LOOP_BUDGET`]: crate::TICK_LOOP_BUDGET
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopBudget {
    remaining: u64,
    exhausted: bool,
}

impl LoopBudget {
    /// A budget of `iterations` loop iterations.
    pub const fn new(iterations: u64) -> Self {
        LoopBudget {
            remaining: iterations,
            exhausted: false,
        }
    }

    /// A budget that never runs out (only the per-run [`LOOP_CAP`] applies).
    pub const fn unlimited() -> Self {
        Self::new(u64::MAX)
    }

    /// Iterations left.
    pub fn remaining(&self) -> u64 {
        self.remaining
    }

    /// True once a loop was refused an iteration (the budget is spent).
    pub fn exhausted(&self) -> bool {
        self.exhausted
    }

    /// Takes one iteration; false (and exhausted from then on) if none is
    /// left.
    #[inline]
    pub(crate) fn take(&mut self) -> bool {
        if self.exhausted || self.remaining == 0 {
            self.exhausted = true;
            return false;
        }
        self.remaining -= 1;
        true
    }

    /// Moves the share `part / whole` of the remaining iterations into a
    /// new budget (for work that runs in parallel). Give it back with
    /// [`refund`](Self::refund). An exhausted budget hands out exhausted
    /// shares.
    pub fn share(&mut self, part: usize, whole: usize) -> LoopBudget {
        if self.exhausted {
            return LoopBudget {
                remaining: 0,
                exhausted: true,
            };
        }
        let whole = whole.max(1) as u128;
        let part = (part as u128).min(whole);
        // part <= whole, so the share fits in u64.
        let n = (u128::from(self.remaining) * part / whole) as u64;
        self.remaining -= n;
        LoopBudget::new(n)
    }

    /// Returns what is left of a [`share`](Self::share); if the share was
    /// exhausted, so is this budget.
    pub fn refund(&mut self, share: LoopBudget) {
        self.remaining = self.remaining.saturating_add(share.remaining);
        self.exhausted |= share.exhausted;
    }
}

impl Program {
    /// Runs the program with only the per-run [`LOOP_CAP`].
    pub fn run(&self, env: &mut ExecEnv) -> RunResult {
        self.run_with_budget(env, &mut LoopBudget::unlimited())
    }

    /// Runs the program; every loop iteration also takes one from `budget`
    /// (see [`LoopBudget`]). `loop_cap_hit` reports the per-run cap only;
    /// check [`LoopBudget::exhausted`] for the shared budget.
    pub fn run_with_budget(&self, env: &mut ExecEnv, budget: &mut LoopBudget) -> RunResult {
        debug_assert!(
            env.vars.len() >= self.num_vars,
            "vars slice ({}) shorter than VarTable::len() ({})",
            env.vars.len(),
            self.num_vars
        );
        let mut small = [0.0f64; STACK_REGS];
        let mut big;
        let regs: &mut [f64] = if self.num_regs <= STACK_REGS {
            &mut small[..]
        } else {
            big = vec![0.0f64; self.num_regs];
            &mut big[..]
        };
        let ops = &self.ops[..];
        let mut pc = 0usize;
        // Loop iterations started so far in this run, and whether the cap hit.
        let mut iters = 0u64;
        let mut capped = false;
        while let Some(op) = ops.get(pc) {
            pc += 1;
            match *op {
                Op::Const { dst, value } => set(regs, dst, value),
                Op::LoadVar { dst, var } => {
                    let v = env.vars.get(var.0 as usize).copied().unwrap_or(0.0);
                    set(regs, dst, v);
                }
                Op::StoreVar { var, src } => {
                    if let Some(slot) = env.vars.get_mut(var.0 as usize) {
                        *slot = get(regs, src);
                    }
                }
                Op::Un { op, dst, a } => set(regs, dst, unary(op, get(regs, a))),
                Op::Bin { op, dst, a, b } => set(regs, dst, binary(op, get(regs, a), get(regs, b))),
                Op::Truthy { dst, a } => {
                    set(regs, dst, if truthy(get(regs, a)) { 1.0 } else { 0.0 })
                }
                Op::JumpIfFalse { cond, target } => {
                    if !truthy(get(regs, cond)) {
                        pc = target as usize;
                    }
                }
                Op::JumpIfTrue { cond, target } => {
                    if truthy(get(regs, cond)) {
                        pc = target as usize;
                    }
                }
                Op::Jump { target } => pc = target as usize,
                Op::Mov { dst, src } => set(regs, dst, get(regs, src)),
                Op::Call1 { f, dst, a } => set(regs, dst, call1(f, get(regs, a))),
                Op::Call2 { f, dst, a, b } => set(regs, dst, call2(f, get(regs, a), get(regs, b))),
                Op::Rand { dst, a } => {
                    let r = env.rng.next_f64();
                    set(regs, dst, rand_scaled(r, get(regs, a)));
                }
                Op::MegaGet { global, dst, idx } => {
                    let i = get(regs, idx);
                    let v = if global {
                        env.gmegabuf.get(i)
                    } else {
                        env.megabuf.get(i)
                    };
                    set(regs, dst, v);
                }
                Op::MegaSet { global, idx, src } => {
                    let (i, v) = (get(regs, idx), get(regs, src));
                    if global {
                        env.gmegabuf.set(i, v);
                    } else {
                        env.megabuf.set(i, v);
                    }
                }
                Op::LoopInit { cnt, n } => {
                    let n = get(regs, n);
                    set(regs, cnt, if n >= 1.0 { n.floor() } else { 0.0 });
                }
                Op::LoopNext { cnt, exit } => {
                    let left = get(regs, cnt);
                    if left >= 1.0 && start_iteration(&mut iters, &mut capped, budget) {
                        set(regs, cnt, left - 1.0);
                    } else {
                        pc = exit as usize;
                    }
                }
                Op::LoopGuard { exit } => {
                    if !start_iteration(&mut iters, &mut capped, budget) {
                        pc = exit as usize;
                    }
                }
            }
        }
        RunResult {
            value: get(regs, 0),
            loop_cap_hit: capped,
        }
    }
}

/// Counts one loop iteration against the run's cap and the shared budget.
/// Returns false when no more iterations are allowed; a per-run cap hit
/// latches `capped`, a spent budget latches the budget's exhausted flag.
#[inline]
pub(crate) fn start_iteration(
    iters: &mut u64,
    capped: &mut bool,
    budget: &mut LoopBudget,
) -> bool {
    if *capped {
        return false;
    }
    if *iters >= LOOP_CAP {
        *capped = true;
        return false;
    }
    if !budget.take() {
        return false;
    }
    *iters += 1;
    true
}

#[inline]
fn get(regs: &[f64], r: u16) -> f64 {
    regs.get(r as usize).copied().unwrap_or(0.0)
}

#[inline]
fn set(regs: &mut [f64], r: u16, v: f64) {
    if let Some(slot) = regs.get_mut(r as usize) {
        *slot = v;
    }
}
