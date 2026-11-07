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
    register_vars(&ast, vars)?;
    let mut folded = ast;
    fold(&mut folded);
    let mut c = Compiler { ops: Vec::new(), max_reg: 0, vars };
    c.expr(&folded, 0)?;
    let num_vars = c.vars.len();
    Ok(Program { ops: c.ops, num_regs: c.max_reg as usize + 1, num_vars })
}

/// Registers every variable and rejects unsupported calls over the whole AST,
/// before folding can discard dead branches.
fn register_vars(e: &Expr, vars: &mut VarTable) -> Result<(), CompileError> {
    match e {
        Expr::Num(_) => {}
        Expr::Var(n) => {
            vars.get_or_insert(n);
        }
        Expr::Assign { name, value, .. } => {
            vars.get_or_insert(name);
            register_vars(value, vars)?;
        }
        Expr::Unary(_, a) => register_vars(a, vars)?,
        Expr::Binary(_, a, b) => {
            register_vars(a, vars)?;
            register_vars(b, vars)?;
        }
        Expr::Call { name, line, col, .. } => {
            return Err(unknown_call(name, *line, *col));
        }
        Expr::Block(v) => {
            for a in v {
                register_vars(a, vars)?;
            }
        }
    }
    Ok(())
}

/// Constant folding, in place (small stack frames: depth can reach 1024).
fn fold(e: &mut Expr) {
    let replacement = match e {
        Expr::Unary(op, a) => {
            fold(a);
            match **a {
                Expr::Num(x) => Some(unary(*op, x)),
                _ => None,
            }
        }
        Expr::Binary(op, a, b) => {
            fold(a);
            // A constant-false `&&` / constant-true `||` never evaluates `b`.
            match (&**a, *op) {
                (Expr::Num(x), BinOp::LogAnd) if !truthy(*x) => Some(0.0),
                (Expr::Num(x), BinOp::LogOr) if truthy(*x) => Some(1.0),
                _ => {
                    fold(b);
                    match (&**a, &**b) {
                        (Expr::Num(x), Expr::Num(y)) => Some(binary(*op, *x, *y)),
                        _ => None,
                    }
                }
            }
        }
        Expr::Assign { value, .. } => {
            fold(value);
            None
        }
        Expr::Call { args, .. } => {
            args.iter_mut().for_each(fold);
            None
        }
        Expr::Block(v) => {
            v.iter_mut().for_each(fold);
            None
        }
        Expr::Num(_) | Expr::Var(_) => None,
    };
    if let Some(v) = replacement {
        *e = Expr::Num(v);
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
    /// `dst` are free scratch. Kept small, with the bulky arms in separate
    /// functions, because it recurses up to the parser's height limit.
    fn expr(&mut self, e: &Expr, dst: usize) -> Result<(), CompileError> {
        let d = self.reg(dst)?;
        match e {
            Expr::Num(v) => self.ops.push(Op::Const { dst: d, value: *v }),
            Expr::Var(n) => self.load_var(n, d),
            Expr::Assign { op, name, value } => self.assign(*op, name, value, dst)?,
            Expr::Unary(op, a) => {
                self.expr(a, dst)?;
                self.ops.push(Op::Un { op: *op, dst: d, a: d });
            }
            Expr::Binary(op @ (BinOp::LogAnd | BinOp::LogOr), a, b) => {
                self.logic(*op, a, b, dst)?
            }
            Expr::Binary(op, a, b) => {
                let t = self.reg(dst + 1)?;
                self.expr(a, dst)?;
                self.expr(b, dst + 1)?;
                self.ops.push(Op::Bin { op: *op, dst: d, a: d, b: t });
            }
            Expr::Call { name, line, col, .. } => return Err(unknown_call(name, *line, *col)),
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

    #[inline(never)]
    fn load_var(&mut self, name: &str, d: Reg) {
        let var = self.var(name);
        self.ops.push(Op::LoadVar { dst: d, var });
    }

    #[inline(never)]
    fn assign(
        &mut self,
        op: AssignOp,
        name: &str,
        value: &Expr,
        dst: usize,
    ) -> Result<(), CompileError> {
        let d = self.reg(dst)?;
        let var = self.var(name);
        let bop = match op {
            AssignOp::Set => None,
            AssignOp::Add => Some(BinOp::Add),
            AssignOp::Sub => Some(BinOp::Sub),
            AssignOp::Mul => Some(BinOp::Mul),
            AssignOp::Div => Some(BinOp::Div),
            AssignOp::Mod => Some(BinOp::Mod),
        };
        match bop {
            None => self.expr(value, dst)?,
            Some(bop) => {
                let t = self.reg(dst + 1)?;
                self.ops.push(Op::LoadVar { dst: d, var });
                self.expr(value, dst + 1)?;
                self.ops.push(Op::Bin { op: bop, dst: d, a: d, b: t });
            }
        }
        self.ops.push(Op::StoreVar { var, src: d });
        Ok(())
    }

    #[inline(never)]
    fn logic(&mut self, op: BinOp, a: &Expr, b: &Expr, dst: usize) -> Result<(), CompileError> {
        let d = self.reg(dst)?;
        self.expr(a, dst)?;
        self.ops.push(Op::Truthy { dst: d, a: d });
        let jump_at = self.ops.len();
        self.ops.push(Op::JumpIfFalse { cond: d, target: 0 });
        self.expr(b, dst)?;
        self.ops.push(Op::Truthy { dst: d, a: d });
        let end = self.ops.len() as u32;
        self.ops[jump_at] = if op == BinOp::LogAnd {
            Op::JumpIfFalse { cond: d, target: end }
        } else {
            Op::JumpIfTrue { cond: d, target: end }
        };
        Ok(())
    }
}

#[cold]
#[inline(never)]
fn unknown_call(name: &str, line: u32, col: u32) -> CompileError {
    CompileError { line, col, message: format!("unknown function '{name}'") }
}
