//! Flat register bytecode.
//!
//! Scratch registers (`u16`) are separate from variables (`VarId`). Jump
//! targets are absolute indices into the op vector; only `Jump` goes backward
//! (loop back-edges), all conditional jumps and loop exits go forward. The program's result is
//! scratch register 0 at the end.

use crate::ast::{BinOp, UnOp};
use crate::builtins::{Fn1, Fn2};
use crate::vars::VarId;

pub type Reg = u16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    /// `dst = value`
    Const { dst: Reg, value: f64 },
    /// `dst = vars[var]`
    LoadVar { dst: Reg, var: VarId },
    /// `vars[var] = src`
    StoreVar { var: VarId, src: Reg },
    /// `dst = op a` (Neg/Plus/Not)
    Un { op: UnOp, dst: Reg, a: Reg },
    /// `dst = a op b` (never LogAnd/LogOr; those lower to jumps)
    Bin { op: BinOp, dst: Reg, a: Reg, b: Reg },
    /// `dst = truthy(a) ? 1 : 0`
    Truthy { dst: Reg, a: Reg },
    /// Jump to `target` if `!truthy(cond)`.
    JumpIfFalse { cond: Reg, target: u32 },
    /// Jump to `target` if `truthy(cond)`.
    JumpIfTrue { cond: Reg, target: u32 },
    /// Unconditional jump (loop back-edge).
    Jump { target: u32 },
    /// `dst = src`
    Mov { dst: Reg, src: Reg },
    /// `dst = f(a)`
    Call1 { f: Fn1, dst: Reg, a: Reg },
    /// `dst = f(a, b)`
    Call2 { f: Fn2, dst: Reg, a: Reg, b: Reg },
    /// `dst = rand(a)`: advances the RNG.
    Rand { dst: Reg, a: Reg },
    /// `dst = (g)megabuf[idx]`
    MegaGet { global: bool, dst: Reg, idx: Reg },
    /// `(g)megabuf[idx] = src`
    MegaSet { global: bool, idx: Reg, src: Reg },
    /// `cnt = floor(n)` if `n >= 1`, else 0 (NaN gives 0). `cnt` is a
    /// dedicated scratch register holding the remaining trip count.
    LoopInit { cnt: Reg, n: Reg },
    /// Start of a `loop` iteration. Jumps to `exit` when the loop cap has been
    /// hit or `cnt < 1`; otherwise counts one iteration against the run's cap
    /// (hitting the cap sets the flag and exits) and decrements `cnt`.
    LoopNext { cnt: Reg, exit: u32 },
    /// Start of a `while` iteration: jumps to `exit` if the cap was hit,
    /// otherwise counts one iteration against the cap (as `LoopNext`).
    LoopGuard { exit: u32 },
}
