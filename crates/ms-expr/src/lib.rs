//! EEL2-compatible expression language.

mod ast;
mod batch;
mod builtins;
mod bytecode;
mod compile;
mod dataflow;
mod lexer;
mod mem;
mod ops;
mod parser;
mod rng;
mod vars;
mod vm;

pub use ast::{AssignOp, BinOp, Expr, UnOp};
pub use batch::{BatchScratch, BatchShared, Lanes, BLOCK};
pub use builtins::{Fn1, Fn2};
pub use bytecode::{Op, Reg};
pub use compile::{compile, Program};
pub use lexer::{lex, Tok, Token};
pub use mem::Megabuf;
pub use parser::parse;
pub use rng::Rng;
pub use vars::{VarId, VarTable};
pub use vm::{ExecEnv, LoopBudget, RunResult};

/// Maximum total loop iterations per `Program::run`.
pub const LOOP_CAP: u64 = 1_048_576;

/// Loop iterations one preset may run per tick, across all of its code
/// (init, frame, every vertex, wave point and shape instance): spec §5.1's
/// cap applied per tick. See [`LoopBudget`].
pub const TICK_LOOP_BUDGET: u64 = LOOP_CAP;

use std::fmt;

/// An error produced while lexing, parsing or compiling an expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    pub line: u32,
    pub col: u32,
    pub message: String,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.col, self.message)
    }
}

impl std::error::Error for CompileError {}
