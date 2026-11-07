//! Flat register bytecode.
//!
//! Scratch registers (`u16`) are separate from variables (`VarId`). Jump
//! targets are absolute indices into the op vector. The program's result is
//! scratch register 0 at the end.

use crate::ast::{BinOp, UnOp};
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
}
