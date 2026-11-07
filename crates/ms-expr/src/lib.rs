//! EEL2-compatible expression language.

mod ast;
mod lexer;
mod parser;

pub use lexer::{lex, Tok, Token};
pub use ast::{AssignOp, BinOp, Expr, UnOp};
pub use parser::parse;

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
