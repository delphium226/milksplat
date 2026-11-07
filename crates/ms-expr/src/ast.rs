//! Abstract syntax tree produced by the parser.

/// Assignment operator: `=` or a compound form such as `+=`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AssignOp {
    Set,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Prefix operator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnOp {
    Neg,
    Plus,
    Not,
}

/// Infix operator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    BitAnd,
    BitOr,
    LogAnd,
    LogOr,
}

/// An expression node. Everything, including statements, is an expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Num(f64),
    Var(String),
    Assign { op: AssignOp, name: String, value: Box<Expr> },
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    /// Function call; `line`/`col` locate the function name.
    Call { name: String, args: Vec<Expr>, line: u32, col: u32 },
    /// A `;`-separated sequence; evaluates to its last statement.
    Block(Vec<Expr>),
}
