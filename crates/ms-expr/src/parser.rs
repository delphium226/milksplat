//! Pratt / recursive-descent parser producing [`Expr`].

use crate::ast::{AssignOp, BinOp, Expr, UnOp};
use crate::{lex, CompileError, Tok, Token};

/// Maximum syntactic nesting before the parser gives up.
const MAX_DEPTH: u32 = 256;

/// Maximum height of the AST the parser will build (covers left-deep chains).
const MAX_AST_HEIGHT: u32 = 1024;

/// Parse source text into a top-level [`Expr::Block`].
pub fn parse(src: &str) -> Result<Expr, CompileError> {
    let toks = lex(src)?;
    let mut p = Parser {
        toks,
        pos: 0,
        depth: 0,
        height: 0,
    };
    let mut stmts = Vec::new();
    loop {
        while matches!(p.peek().tok, Tok::Semi) {
            p.pos += 1;
        }
        if matches!(p.peek().tok, Tok::Eof) {
            break;
        }
        stmts.push(p.assign()?);
        match p.peek().tok {
            Tok::Semi | Tok::Eof => {}
            _ => return Err(p.unexpected()),
        }
    }
    Ok(Expr::Block(stmts))
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
    depth: u32,
    /// Height of the most recently returned expression.
    height: u32,
}

/// Binary operator info: (operator, precedence, right-associative).
fn binop(op: &str) -> Option<(BinOp, u8, bool)> {
    Some(match op {
        "||" => (BinOp::LogOr, 1, false),
        "&&" => (BinOp::LogAnd, 2, false),
        "|" => (BinOp::BitOr, 3, false),
        "&" => (BinOp::BitAnd, 4, false),
        "==" => (BinOp::Eq, 5, false),
        "!=" => (BinOp::Ne, 5, false),
        "<" => (BinOp::Lt, 5, false),
        "<=" => (BinOp::Le, 5, false),
        ">" => (BinOp::Gt, 5, false),
        ">=" => (BinOp::Ge, 5, false),
        "+" => (BinOp::Add, 6, false),
        "-" => (BinOp::Sub, 6, false),
        "*" => (BinOp::Mul, 7, false),
        "/" => (BinOp::Div, 7, false),
        "%" => (BinOp::Mod, 7, false),
        "^" => (BinOp::Pow, 8, true),
        _ => return None,
    })
}

fn assign_op(op: &str) -> Option<AssignOp> {
    Some(match op {
        "=" => AssignOp::Set,
        "+=" => AssignOp::Add,
        "-=" => AssignOp::Sub,
        "*=" => AssignOp::Mul,
        "/=" => AssignOp::Div,
        "%=" => AssignOp::Mod,
        _ => return None,
    })
}

impl Parser {
    fn peek(&self) -> &Token {
        // The lexer always ends with Eof and we never advance past it.
        &self.toks[self.pos.min(self.toks.len() - 1)]
    }

    fn advance(&mut self) -> Token {
        let t = self.peek().clone();
        if !matches!(t.tok, Tok::Eof) {
            self.pos += 1;
        }
        t
    }

    fn err_at(&self, t: &Token, message: impl Into<String>) -> CompileError {
        CompileError {
            line: t.line,
            col: t.col,
            message: message.into(),
        }
    }

    fn unexpected(&self) -> CompileError {
        let t = self.peek();
        let what = match &t.tok {
            Tok::Eof => "unexpected end of input".to_string(),
            Tok::Num(_) => "unexpected number".to_string(),
            Tok::Ident(s) => format!("unexpected '{s}'"),
            Tok::Op(o) => format!("unexpected '{o}'"),
            Tok::LParen => "unexpected '('".to_string(),
            Tok::RParen => "unexpected ')'".to_string(),
            Tok::Comma => "unexpected ','".to_string(),
            Tok::Semi => "unexpected ';'".to_string(),
        };
        self.err_at(t, what)
    }

    fn enter(&mut self) -> Result<(), CompileError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err_at(self.peek(), "nesting too deep"));
        }
        Ok(())
    }

    /// Records the height of a node built from children of heights `h`.
    fn set_height(&mut self, at: &Token, h: u32) -> Result<(), CompileError> {
        if h > MAX_AST_HEIGHT {
            return Err(self.err_at(at, "expression too deeply nested"));
        }
        self.height = h;
        Ok(())
    }

    fn leave(&mut self) {
        self.depth -= 1;
    }

    /// Assignment level (right-associative, lowest precedence).
    fn assign(&mut self) -> Result<Expr, CompileError> {
        self.enter()?;
        let r = self.assign_inner();
        self.leave();
        r
    }

    fn assign_inner(&mut self) -> Result<Expr, CompileError> {
        let lhs = self.binary(1)?;
        let lhs_h = self.height;
        let op_tok = self.peek().clone();
        let Tok::Op(o) = op_tok.tok else {
            return Ok(lhs);
        };
        let Some(op) = assign_op(o) else {
            return Ok(lhs);
        };
        let name = match lhs {
            Expr::Var(name) => name,
            Expr::Call {
                name,
                mut args,
                line,
                col,
            } => {
                let lower = name.to_ascii_lowercase();
                if lower != "megabuf" && lower != "gmegabuf" {
                    return Err(self.err_at(&op_tok, "cannot assign to a non-variable"));
                }
                let (1, Some(index)) = (args.len(), args.pop()) else {
                    return Err(CompileError {
                        line,
                        col,
                        message: format!("'{name}' expects 1 argument(s)"),
                    });
                };
                self.pos += 1;
                let value = Box::new(self.assign()?);
                let h = self.height.max(lhs_h) + 1;
                self.set_height(&op_tok, h)?;
                return Ok(Expr::MemAssign {
                    global: lower == "gmegabuf",
                    index: Box::new(index),
                    op,
                    value,
                });
            }
            _ => return Err(self.err_at(&op_tok, "cannot assign to a non-variable")),
        };
        self.pos += 1;
        let value = Box::new(self.assign()?);
        let h = self.height + 1;
        self.set_height(&op_tok, h)?;
        Ok(Expr::Assign { op, name, value })
    }

    fn binary(&mut self, min_prec: u8) -> Result<Expr, CompileError> {
        self.enter()?;
        let r = self.binary_inner(min_prec);
        self.leave();
        r
    }

    fn binary_inner(&mut self, min_prec: u8) -> Result<Expr, CompileError> {
        let mut lhs = self.unary()?;
        let mut lh = self.height;
        while let Tok::Op(o) = self.peek().tok {
            let Some((op, prec, right)) = binop(o) else {
                break;
            };
            if prec < min_prec {
                break;
            }
            let op_tok = self.advance();
            let rhs = self.binary(if right { prec } else { prec + 1 })?;
            lh = 1 + lh.max(self.height);
            self.set_height(&op_tok, lh)?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs));
        }
        self.height = lh;
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr, CompileError> {
        let op = match self.peek().tok {
            Tok::Op("-") => UnOp::Neg,
            Tok::Op("+") => UnOp::Plus,
            Tok::Op("!") => UnOp::Not,
            _ => return self.primary(),
        };
        let op_tok = self.advance();
        self.enter()?;
        let r = self.unary();
        self.leave();
        let inner = r?;
        let h = self.height + 1;
        self.set_height(&op_tok, h)?;
        Ok(Expr::Unary(op, Box::new(inner)))
    }

    fn primary(&mut self) -> Result<Expr, CompileError> {
        match self.peek().tok.clone() {
            Tok::Num(v) => {
                self.pos += 1;
                self.height = 1;
                Ok(Expr::Num(v))
            }
            Tok::Ident(name) => {
                let t = self.advance();
                if matches!(self.peek().tok, Tok::LParen) {
                    self.pos += 1;
                    let args = self.call_args()?;
                    let h = self.height + 1;
                    self.set_height(&t, h)?;
                    Ok(Expr::Call {
                        name,
                        args,
                        line: t.line,
                        col: t.col,
                    })
                } else {
                    self.height = 1;
                    Ok(Expr::Var(name))
                }
            }
            Tok::LParen => {
                self.pos += 1;
                let e = self.sequence()?;
                self.expect_rparen()?;
                Ok(e)
            }
            _ => Err(self.unexpected()),
        }
    }

    fn expect_rparen(&mut self) -> Result<(), CompileError> {
        if matches!(self.peek().tok, Tok::RParen) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.unexpected())
        }
    }

    /// Arguments after `(`, consuming the closing `)`.
    fn call_args(&mut self) -> Result<Vec<Expr>, CompileError> {
        let mut args = Vec::new();
        if matches!(self.peek().tok, Tok::RParen) {
            self.pos += 1;
            self.height = 0;
            return Ok(args);
        }
        let mut max_h = 0;
        loop {
            args.push(self.sequence()?);
            max_h = max_h.max(self.height);
            if matches!(self.peek().tok, Tok::Comma) {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.expect_rparen()?;
        self.height = max_h;
        Ok(args)
    }

    /// One or more `;`-separated statements (empty ones skipped), ending at
    /// `,` or `)`. A single statement is returned bare, several as a `Block`.
    fn sequence(&mut self) -> Result<Expr, CompileError> {
        let mut stmts = Vec::new();
        let mut max_h = 0;
        loop {
            while matches!(self.peek().tok, Tok::Semi) {
                self.pos += 1;
            }
            if matches!(self.peek().tok, Tok::RParen | Tok::Comma | Tok::Eof) {
                break;
            }
            stmts.push(self.assign()?);
            max_h = max_h.max(self.height);
            match self.peek().tok {
                Tok::Semi | Tok::RParen | Tok::Comma | Tok::Eof => {}
                _ => return Err(self.unexpected()),
            }
        }
        match stmts.len() {
            0 => Err(self.unexpected()),
            1 => Ok(stmts.remove(0)),
            _ => {
                let t = self.peek().clone();
                self.set_height(&t, max_h + 1)?;
                Ok(Expr::Block(stmts))
            }
        }
    }
}
