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

impl Program {
    pub fn run(&self, env: &mut ExecEnv) -> RunResult {
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
                    if left >= 1.0 && start_iteration(&mut iters, &mut capped) {
                        set(regs, cnt, left - 1.0);
                    } else {
                        pc = exit as usize;
                    }
                }
                Op::LoopGuard { exit } => {
                    if !start_iteration(&mut iters, &mut capped) {
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

/// Counts one loop iteration against the run's cap. Returns false (and
/// latches `capped`) when no more iterations are allowed.
#[inline]
fn start_iteration(iters: &mut u64, capped: &mut bool) -> bool {
    if *capped {
        return false;
    }
    if *iters >= LOOP_CAP {
        *capped = true;
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
