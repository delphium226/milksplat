//! Built-in function table and the pure function implementations shared by
//! the constant folder and the VM.

use crate::ast::BinOp;

/// One-argument pure functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fn1 {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sqrt,
    Exp,
    Log,
    Log10,
    Abs,
    Sign,
    Floor,
    Ceil,
    Sqr,
}

/// Two-argument pure functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fn2 {
    Atan2,
    Pow,
    Min,
    Max,
    Sigmoid,
}

/// What a function name means to the compiler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Builtin {
    F1(Fn1),
    F2(Fn2),
    /// `band`/`bor`/`equal`/`above`/`below`: rewritten to the operator.
    Op2(BinOp),
    /// `bnot`: rewritten to `!x`.
    Not,
    Rand,
    /// `megabuf` (false) / `gmegabuf` (true).
    Mega(bool),
    If,
    Loop,
    While,
}

/// Looks up a function (case-insensitive); returns it with its arity.
pub fn lookup(name: &str) -> Option<(Builtin, usize)> {
    use Builtin::*;
    let lower = name.to_ascii_lowercase();
    Some(match lower.as_str() {
        "sin" => (F1(Fn1::Sin), 1),
        "cos" => (F1(Fn1::Cos), 1),
        "tan" => (F1(Fn1::Tan), 1),
        "asin" => (F1(Fn1::Asin), 1),
        "acos" => (F1(Fn1::Acos), 1),
        "atan" => (F1(Fn1::Atan), 1),
        "sqrt" => (F1(Fn1::Sqrt), 1),
        "exp" => (F1(Fn1::Exp), 1),
        "log" => (F1(Fn1::Log), 1),
        "log10" => (F1(Fn1::Log10), 1),
        "abs" => (F1(Fn1::Abs), 1),
        "sign" => (F1(Fn1::Sign), 1),
        "floor" | "int" => (F1(Fn1::Floor), 1),
        "ceil" => (F1(Fn1::Ceil), 1),
        "sqr" => (F1(Fn1::Sqr), 1),
        "atan2" => (F2(Fn2::Atan2), 2),
        "pow" => (F2(Fn2::Pow), 2),
        "min" => (F2(Fn2::Min), 2),
        "max" => (F2(Fn2::Max), 2),
        "sigmoid" => (F2(Fn2::Sigmoid), 2),
        "band" => (Op2(BinOp::LogAnd), 2),
        "bor" => (Op2(BinOp::LogOr), 2),
        "equal" => (Op2(BinOp::Eq), 2),
        "above" => (Op2(BinOp::Gt), 2),
        "below" => (Op2(BinOp::Lt), 2),
        "bnot" => (Not, 1),
        "rand" => (Rand, 1),
        "megabuf" => (Mega(false), 1),
        "gmegabuf" => (Mega(true), 1),
        "if" => (If, 3),
        "loop" => (Loop, 2),
        "while" => (While, 1),
        _ => return None,
    })
}

/// Evaluates a one-argument function. Domain errors yield 0, never NaN
/// (`sqrt` takes the absolute value, as EEL2 does).
#[inline]
pub fn call1(f: Fn1, x: f64) -> f64 {
    nan_guard1(x, match f {
        Fn1::Sin => x.sin(),
        Fn1::Cos => x.cos(),
        Fn1::Tan => x.tan(),
        Fn1::Asin => {
            if x.abs() > 1.0 {
                0.0
            } else {
                x.asin()
            }
        }
        Fn1::Acos => {
            if x.abs() > 1.0 {
                0.0
            } else {
                x.acos()
            }
        }
        Fn1::Atan => x.atan(),
        Fn1::Sqrt => x.abs().sqrt(),
        Fn1::Exp => x.exp(),
        Fn1::Log => {
            if x > 0.0 {
                x.ln()
            } else {
                0.0
            }
        }
        Fn1::Log10 => {
            if x > 0.0 {
                x.log10()
            } else {
                0.0
            }
        }
        Fn1::Abs => x.abs(),
        Fn1::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        Fn1::Floor => x.floor(),
        Fn1::Ceil => x.ceil(),
        Fn1::Sqr => x * x,
    })
}

/// Evaluates a two-argument function.
#[inline]
pub fn call2(f: Fn2, a: f64, b: f64) -> f64 {
    nan_guard2(a, b, match f {
        Fn2::Atan2 => a.atan2(b),
        Fn2::Pow => a.powf(b),
        Fn2::Min => a.min(b),
        Fn2::Max => a.max(b),
        Fn2::Sigmoid => {
            // 0 * inf is NaN; treat it as 0, giving 0.5.
            let p = -a * b;
            let p = if p.is_nan() && !a.is_nan() && !b.is_nan() { 0.0 } else { p };
            1.0 / (1.0 + p.exp())
        }
    })
}

/// A NaN result from non-NaN input (a domain error such as `pow(-2, 0.5)`
/// or `inf - inf` inside a function) becomes 0. NaN inputs may propagate.
#[inline]
fn nan_guard1(x: f64, r: f64) -> f64 {
    if r.is_nan() && !x.is_nan() {
        0.0
    } else {
        r
    }
}

#[inline]
fn nan_guard2(a: f64, b: f64, r: f64) -> f64 {
    if r.is_nan() && !a.is_nan() && !b.is_nan() {
        0.0
    } else {
        r
    }
}

/// `rand(x)`: a uniform sample `r` in `[0, 1)` scaled by `max(x, 1)`.
#[inline]
pub fn rand_scaled(r: f64, x: f64) -> f64 {
    r * x.max(1.0)
}
