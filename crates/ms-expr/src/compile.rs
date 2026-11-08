//! AST -> constant folding -> register bytecode.

use crate::ast::{AssignOp, BinOp, Expr, UnOp};
use crate::builtins::{call1, call2, lookup, Builtin};
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
    /// Sorted variables some path reads before writing (see `dataflow`).
    pub(crate) rbw: Vec<VarId>,
}

impl Program {
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }
    pub fn num_regs(&self) -> usize {
        self.num_regs
    }

    /// This program with its variables renumbered densely: the variables it
    /// references, in increasing order of their original ids, become
    /// `VarId(0)`, `VarId(1)`, .... Returns the renumbered program and, for
    /// each new id, the original one. The two programs compute the same
    /// thing; the compact one needs only that many `Lanes` columns.
    pub fn compact_vars(&self) -> (Program, Vec<VarId>) {
        let mut used: Vec<VarId> = self
            .ops
            .iter()
            .filter_map(|op| match *op {
                Op::LoadVar { var, .. } | Op::StoreVar { var, .. } => Some(var),
                _ => None,
            })
            .collect();
        used.sort_by_key(|v| v.0);
        used.dedup();
        let new_id = |v: VarId| {
            used.binary_search_by_key(&v.0, |u| u.0)
                .map_or(VarId(u32::MAX), |i| VarId(i as u32))
        };
        let ops = self
            .ops
            .iter()
            .map(|op| match *op {
                Op::LoadVar { dst, var } => Op::LoadVar {
                    dst,
                    var: new_id(var),
                },
                Op::StoreVar { var, src } => Op::StoreVar {
                    var: new_id(var),
                    src,
                },
                other => other,
            })
            .collect();
        let rbw = self.rbw.iter().map(|&v| new_id(v)).collect();
        let p = Program {
            ops,
            num_regs: self.num_regs,
            num_vars: used.len(),
            rbw,
        };
        (p, used)
    }
}

pub fn compile(src: &str, vars: &mut VarTable) -> Result<Program, CompileError> {
    let ast = parse(src)?;
    register_vars(&ast, vars)?;
    let mut folded = ast;
    fold(&mut folded);
    let mut c = Compiler {
        ops: Vec::new(),
        max_reg: 0,
        vars,
    };
    c.expr(&folded, 0)?;
    let num_vars = c.vars.len();
    let rbw = crate::dataflow::reads_before_write(&c.ops);
    Ok(Program {
        rbw,
        ops: c.ops,
        num_regs: c.max_reg as usize + 1,
        num_vars,
    })
}

/// Registers every variable and checks every call's name and arity over the
/// whole AST, before folding can discard dead branches.
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
        Expr::Call {
            name,
            args,
            line,
            col,
        } => {
            check_call(name, args.len(), *line, *col)?;
            for a in args {
                register_vars(a, vars)?;
            }
        }
        Expr::MemAssign { index, value, .. } => {
            register_vars(index, vars)?;
            register_vars(value, vars)?;
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
    if matches!(e, Expr::Call { .. }) {
        return fold_call(e);
    }
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
        Expr::MemAssign { index, value, .. } => {
            fold(index);
            fold(value);
            None
        }
        Expr::Block(v) => {
            v.iter_mut().for_each(fold);
            None
        }
        Expr::Num(_) | Expr::Var(_) | Expr::Call { .. } => None,
    };
    if let Some(v) = replacement {
        *e = Expr::Num(v);
    }
}

/// Folds a call: operator-like builtins become operators, pure builtins with
/// constant arguments evaluate, `if` with a constant condition becomes its
/// chosen branch. `rand`, `megabuf`, `loop` and `while` are never evaluated.
#[inline(never)]
fn fold_call(e: &mut Expr) {
    let Expr::Call { name, args, .. } = e else {
        return;
    };
    let Some((builtin, _)) = lookup(name) else {
        return;
    };
    match (builtin, args.len()) {
        (Builtin::Op2(op), 2) => {
            let b = args.pop().unwrap_or(Expr::Num(0.0));
            let a = args.pop().unwrap_or(Expr::Num(0.0));
            *e = Expr::Binary(op, Box::new(a), Box::new(b));
            return fold(e);
        }
        (Builtin::Not, 1) => {
            let a = args.pop().unwrap_or(Expr::Num(0.0));
            *e = Expr::Unary(UnOp::Not, Box::new(a));
            return fold(e);
        }
        (Builtin::If, 3) => {
            fold(&mut args[0]);
            if let Expr::Num(c) = args[0] {
                let mut chosen =
                    std::mem::replace(&mut args[if truthy(c) { 1 } else { 2 }], Expr::Num(0.0));
                fold(&mut chosen);
                *e = chosen;
                return;
            }
            args.iter_mut().skip(1).for_each(fold);
            return;
        }
        _ => {}
    }
    args.iter_mut().for_each(fold);
    let value = match (builtin, args.as_slice()) {
        (Builtin::F1(f), [Expr::Num(x)]) => call1(f, *x),
        (Builtin::F2(f), [Expr::Num(x), Expr::Num(y)]) => call2(f, *x, *y),
        _ => return,
    };
    *e = Expr::Num(value);
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
                self.ops.push(Op::Un {
                    op: *op,
                    dst: d,
                    a: d,
                });
            }
            Expr::Binary(op @ (BinOp::LogAnd | BinOp::LogOr), a, b) => {
                self.logic(*op, a, b, dst)?
            }
            Expr::Binary(op, a, b) => {
                let t = self.reg(dst + 1)?;
                self.expr(a, dst)?;
                self.expr(b, dst + 1)?;
                self.ops.push(Op::Bin {
                    op: *op,
                    dst: d,
                    a: d,
                    b: t,
                });
            }
            Expr::Call {
                name,
                args,
                line,
                col,
            } => self.call(name, args, dst, *line, *col)?,
            Expr::MemAssign {
                global,
                index,
                op,
                value,
            } => self.mem_assign(*global, index, *op, value, dst)?,
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
        match compound_op(op) {
            None => self.expr(value, dst)?,
            Some(bop) => {
                let t = self.reg(dst + 1)?;
                self.ops.push(Op::LoadVar { dst: d, var });
                self.expr(value, dst + 1)?;
                self.ops.push(Op::Bin {
                    op: bop,
                    dst: d,
                    a: d,
                    b: t,
                });
            }
        }
        self.ops.push(Op::StoreVar { var, src: d });
        Ok(())
    }

    /// `megabuf(i) op= v`. The index is evaluated once, before the value.
    #[inline(never)]
    fn mem_assign(
        &mut self,
        global: bool,
        index: &Expr,
        op: AssignOp,
        value: &Expr,
        dst: usize,
    ) -> Result<(), CompileError> {
        let d = self.reg(dst)?;
        let v = self.reg(dst + 1)?;
        self.expr(index, dst)?;
        match compound_op(op) {
            None => self.expr(value, dst + 1)?,
            Some(bop) => {
                let t = self.reg(dst + 2)?;
                self.ops.push(Op::MegaGet {
                    global,
                    dst: v,
                    idx: d,
                });
                self.expr(value, dst + 2)?;
                self.ops.push(Op::Bin {
                    op: bop,
                    dst: v,
                    a: v,
                    b: t,
                });
            }
        }
        self.ops.push(Op::MegaSet {
            global,
            idx: d,
            src: v,
        });
        self.ops.push(Op::Mov { dst: d, src: v });
        Ok(())
    }

    #[inline(never)]
    fn call(
        &mut self,
        name: &str,
        args: &[Expr],
        dst: usize,
        line: u32,
        col: u32,
    ) -> Result<(), CompileError> {
        let d = self.reg(dst)?;
        let Some((builtin, arity)) = lookup(name) else {
            return Err(unknown_call(name, line, col));
        };
        match (builtin, args) {
            (Builtin::F1(f), [a]) => {
                self.expr(a, dst)?;
                self.ops.push(Op::Call1 { f, dst: d, a: d });
            }
            (Builtin::F2(f), [a, b]) => {
                let t = self.reg(dst + 1)?;
                self.expr(a, dst)?;
                self.expr(b, dst + 1)?;
                self.ops.push(Op::Call2 {
                    f,
                    dst: d,
                    a: d,
                    b: t,
                });
            }
            (Builtin::Op2(op), [a, b]) => {
                if matches!(op, BinOp::LogAnd | BinOp::LogOr) {
                    self.logic(op, a, b, dst)?;
                } else {
                    let t = self.reg(dst + 1)?;
                    self.expr(a, dst)?;
                    self.expr(b, dst + 1)?;
                    self.ops.push(Op::Bin {
                        op,
                        dst: d,
                        a: d,
                        b: t,
                    });
                }
            }
            (Builtin::Not, [a]) => {
                self.expr(a, dst)?;
                self.ops.push(Op::Un {
                    op: UnOp::Not,
                    dst: d,
                    a: d,
                });
            }
            (Builtin::Rand, [a]) => {
                self.expr(a, dst)?;
                self.ops.push(Op::Rand { dst: d, a: d });
            }
            (Builtin::Mega(global), [a]) => {
                self.expr(a, dst)?;
                self.ops.push(Op::MegaGet {
                    global,
                    dst: d,
                    idx: d,
                });
            }
            (Builtin::If, [c, a, b]) => self.if_(c, a, b, dst)?,
            (Builtin::Loop, [n, body]) => self.loop_(n, body, dst)?,
            (Builtin::While, [body]) => self.while_(body, dst)?,
            _ => return Err(arity_error(name, arity, line, col)),
        }
        Ok(())
    }

    /// `if(c, a, b)`: only the chosen branch runs.
    #[inline(never)]
    fn if_(&mut self, c: &Expr, a: &Expr, b: &Expr, dst: usize) -> Result<(), CompileError> {
        let d = self.reg(dst)?;
        self.expr(c, dst)?;
        let to_else = self.ops.len();
        self.ops.push(Op::JumpIfFalse { cond: d, target: 0 });
        self.expr(a, dst)?;
        let to_end = self.ops.len();
        self.ops.push(Op::Jump { target: 0 });
        let else_at = self.ops.len() as u32;
        self.expr(b, dst)?;
        let end = self.ops.len() as u32;
        self.ops[to_else] = Op::JumpIfFalse {
            cond: d,
            target: else_at,
        };
        self.ops[to_end] = Op::Jump { target: end };
        Ok(())
    }

    /// `loop(n, body)`: `n` is evaluated once into `dst`, the trip counter
    /// lives in `dst + 1` and the body runs from `dst + 2`. Value is 0.
    #[inline(never)]
    fn loop_(&mut self, n: &Expr, body: &Expr, dst: usize) -> Result<(), CompileError> {
        let d = self.reg(dst)?;
        let cnt = self.reg(dst + 1)?;
        self.reg(dst + 2)?;
        self.expr(n, dst)?;
        self.ops.push(Op::LoopInit { cnt, n: d });
        self.ops.push(Op::Const { dst: d, value: 0.0 });
        let top = self.ops.len() as u32;
        self.ops.push(Op::LoopNext { cnt, exit: 0 });
        self.expr(body, dst + 2)?;
        self.ops.push(Op::Jump { target: top });
        let exit = self.ops.len() as u32;
        self.ops[top as usize] = Op::LoopNext { cnt, exit };
        Ok(())
    }

    /// `while(body)`: runs the body, repeating while its value is truthy.
    #[inline(never)]
    fn while_(&mut self, body: &Expr, dst: usize) -> Result<(), CompileError> {
        let d = self.reg(dst)?;
        self.ops.push(Op::Const { dst: d, value: 0.0 });
        let top = self.ops.len() as u32;
        self.ops.push(Op::LoopGuard { exit: 0 });
        self.expr(body, dst)?;
        let test = self.ops.len();
        self.ops.push(Op::JumpIfFalse { cond: d, target: 0 });
        self.ops.push(Op::Jump { target: top });
        let exit = self.ops.len() as u32;
        self.ops[top as usize] = Op::LoopGuard { exit };
        self.ops[test] = Op::JumpIfFalse {
            cond: d,
            target: exit,
        };
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
            Op::JumpIfFalse {
                cond: d,
                target: end,
            }
        } else {
            Op::JumpIfTrue {
                cond: d,
                target: end,
            }
        };
        Ok(())
    }
}

fn compound_op(op: AssignOp) -> Option<BinOp> {
    match op {
        AssignOp::Set => None,
        AssignOp::Add => Some(BinOp::Add),
        AssignOp::Sub => Some(BinOp::Sub),
        AssignOp::Mul => Some(BinOp::Mul),
        AssignOp::Div => Some(BinOp::Div),
        AssignOp::Mod => Some(BinOp::Mod),
    }
}

/// Checks a call's name and argument count.
fn check_call(name: &str, nargs: usize, line: u32, col: u32) -> Result<(), CompileError> {
    match lookup(name) {
        None => Err(unknown_call(name, line, col)),
        Some((_, arity)) if arity != nargs => Err(arity_error(name, arity, line, col)),
        Some(_) => Ok(()),
    }
}

#[cold]
#[inline(never)]
fn arity_error(name: &str, arity: usize, line: u32, col: u32) -> CompileError {
    CompileError {
        line,
        col,
        message: format!("'{name}' expects {arity} argument(s)"),
    }
}

#[cold]
#[inline(never)]
fn unknown_call(name: &str, line: u32, col: u32) -> CompileError {
    CompileError {
        line,
        col,
        message: format!("unknown function '{name}'"),
    }
}
