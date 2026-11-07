//! AST -> constant folding -> register bytecode.

use crate::ast::{AssignOp, BinOp, Expr};
use crate::bytecode::{Op, Reg};
use crate::ops::{binary, truthy, unary};
use crate::parser::parse;
use crate::vars::{VarId, VarTable};
use crate::CompileError;

/// A compiled program. Immutable; share freely between threads.
#[derive(Debug, Clone)]
pub struct Program {
    pub(crate) ops: Vec<Op>,
    pub(crate) num_regs: usize,
    /// Minimum `vars.len()` the program needs.
    pub(crate) num_vars: usize,
}

impl Program {
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }
    pub fn num_regs(&self) -> usize {
        self.num_regs
    }
}

pub fn compile(src: &str, vars: &mut VarTable) -> Result<Program, CompileError> {
    let ast = parse(src)?;
    register_vars(&ast, vars);
    let folded = fold(ast);
    let mut c = Compiler { ops: Vec::new(), max_reg: 0, vars };
    c.expr(&folded, 0)?;
    let num_vars = c.vars.len();
    Ok(Program { ops: c.ops, num_regs: c.max_reg as usize + 1, num_vars })
}

fn register_vars(e: &Expr, vars: &mut VarTable) {
    match e {
        Expr::Num(_) => {}
        Expr::Var(n) => {
            vars.get_or_insert(n);
        }
        Expr::Assign { name, value, .. } => {
            vars.get_or_insert(name);
            register_vars(value, vars);
        }
        Expr::Unary(_, a) => register_vars(a, vars),
        Expr::Binary(_, a, b) => {
            register_vars(a, vars);
            register_vars(b, vars);
        }
        Expr::Call { args, .. } => args.iter().for_each(|a| register_vars(a, vars)),
        Expr::Block(v) => v.iter().for_each(|a| register_vars(a, vars)),
    }
}

fn fold(e: Expr) -> Expr {
    match e {
        Expr::Unary(op, a) => match fold(*a) {
            Expr::Num(x) => Expr::Num(unary(op, x)),
            a => Expr::Unary(op, Box::new(a)),
        },
        Expr::Binary(op, a, b) => {
            let a = fold(*a);
            // A constant-false `&&` / constant-true `||` never evaluates `b`.
            if let Expr::Num(x) = a {
                match op {
                    BinOp::LogAnd if !truthy(x) => return Expr::Num(0.0),
                    BinOp::LogOr if truthy(x) => return Expr::Num(1.0),
                    _ => {}
                }
            }
            let b = fold(*b);
            match (&a, &b) {
                (Expr::Num(x), Expr::Num(y)) => Expr::Num(binary(op, *x, *y)),
                _ => Expr::Binary(op, Box::new(a), Box::new(b)),
            }
        }
        Expr::Assign { op, name, value } => {
            Expr::Assign { op, name, value: Box::new(fold(*value)) }
        }
        Expr::Call { name, args, line, col } => {
            Expr::Call { name, args: args.into_iter().map(fold).collect(), line, col }
        }
        Expr::Block(v) => Expr::Block(v.into_iter().map(fold).collect()),
        other => other,
    }
}

struct Compiler<'a> {
    ops: Vec<Op>,
    max_reg: Reg,
    vars: &'a mut VarTable,
}

impl Compiler<'_> {
    fn reg(&mut self, r: usize) -> Result<Reg, CompileError> {
        let r = Reg::try_from(r).map_err(|_| CompileError {
            line: 1,
            col: 1,
            message: "expression too deeply nested".to_string(),
        })?;
        self.max_reg = self.max_reg.max(r);
        Ok(r)
    }

    fn var(&mut self, name: &str) -> VarId {
        self.vars.get_or_insert(name)
    }

    /// Compiles `e` so its value ends up in register `dst`. Registers above
    /// `dst` are free scratch.
    fn expr(&mut self, e: &Expr, dst: usize) -> Result<(), CompileError> {
        let d = self.reg(dst)?;
        match e {
            Expr::Num(v) => self.ops.push(Op::Const { dst: d, value: *v }),
            Expr::Var(n) => {
                let var = self.var(n);
                self.ops.push(Op::LoadVar { dst: d, var });
            }
            Expr::Assign { op, name, value } => {
                let var = self.var(name);
                match op {
                    AssignOp::Set => self.expr(value, dst)?,
                    _ => {
                        let t = self.reg(dst + 1)?;
                        self.ops.push(Op::LoadVar { dst: d, var });
                        self.expr(value, dst + 1)?;
                        let bop = match op {
                            AssignOp::Add => BinOp::Add,
                            AssignOp::Sub => BinOp::Sub,
                            AssignOp::Mul => BinOp::Mul,
                            AssignOp::Div => BinOp::Div,
                            _ => BinOp::Mod,
                        };
                        self.ops.push(Op::Bin { op: bop, dst: d, a: d, b: t });
                    }
                }
                self.ops.push(Op::StoreVar { var, src: d });
            }
            Expr::Unary(op, a) => {
                self.expr(a, dst)?;
                self.ops.push(Op::Un { op: *op, dst: d, a: d });
            }
            Expr::Binary(op @ (BinOp::LogAnd | BinOp::LogOr), a, b) => {
                self.expr(a, dst)?;
                self.ops.push(Op::Truthy { dst: d, a: d });
                let jump_at = self.ops.len();
                self.ops.push(Op::JumpIfFalse { cond: d, target: 0 });
                self.expr(b, dst)?;
                self.ops.push(Op::Truthy { dst: d, a: d });
                let end = self.ops.len() as u32;
                self.ops[jump_at] = if *op == BinOp::LogAnd {
                    Op::JumpIfFalse { cond: d, target: end }
                } else {
                    Op::JumpIfTrue { cond: d, target: end }
                };
            }
            Expr::Binary(op, a, b) => {
                let t = self.reg(dst + 1)?;
                self.expr(a, dst)?;
                self.expr(b, dst + 1)?;
                self.ops.push(Op::Bin { op: *op, dst: d, a: d, b: t });
            }
            Expr::Call { name, line, col, .. } => {
                return Err(CompileError {
                    line: *line,
                    col: *col,
                    message: format!("unknown function '{name}'"),
                });
            }
            Expr::Block(stmts) => {
                if stmts.is_empty() {
                    self.ops.push(Op::Const { dst: d, value: 0.0 });
                }
                for s in stmts {
                    self.expr(s, dst)?;
                }
            }
        }
        Ok(())
    }
}
