//! Operator semantics shared by the constant folder and the VM.

use crate::ast::{BinOp, UnOp};

const EPS: f64 = 1e-5;

#[inline]
pub fn is_zero(x: f64) -> bool {
    x.abs() < EPS
}

#[inline]
pub fn truthy(x: f64) -> bool {
    !is_zero(x)
}

#[inline]
fn b2f(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

#[inline]
pub fn unary(op: UnOp, a: f64) -> f64 {
    match op {
        UnOp::Neg => -a,
        UnOp::Plus => a,
        UnOp::Not => b2f(is_zero(a)),
    }
}

#[inline]
pub fn binary(op: BinOp, a: f64, b: f64) -> f64 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => {
            if b == 0.0 {
                0.0
            } else {
                a / b
            }
        }
        BinOp::Mod => {
            let (x, y) = (a as i64, b as i64);
            if y == 0 {
                0.0
            } else {
                x.wrapping_rem(y) as f64
            }
        }
        BinOp::Pow => a.powf(b),
        BinOp::Eq => b2f((a - b).abs() < EPS),
        BinOp::Ne => b2f((a - b).abs() >= EPS),
        BinOp::Lt => b2f(a < b),
        BinOp::Le => b2f(a <= b),
        BinOp::Gt => b2f(a > b),
        BinOp::Ge => b2f(a >= b),
        BinOp::BitAnd => ((a as i64) & (b as i64)) as f64,
        BinOp::BitOr => ((a as i64) | (b as i64)) as f64,
        BinOp::LogAnd => b2f(truthy(a) && truthy(b)),
        BinOp::LogOr => b2f(truthy(a) || truthy(b)),
    }
}
