use ms_expr::{parse, AssignOp, BinOp, Expr};

fn n(v: f64) -> Box<Expr> {
    Box::new(Expr::Num(v))
}

fn stmts(src: &str) -> Vec<Expr> {
    match parse(src).unwrap() {
        Expr::Block(v) => v,
        other => panic!("expected Block, got {other:?}"),
    }
}

fn one(src: &str) -> Expr {
    let mut v = stmts(src);
    assert_eq!(v.len(), 1);
    v.remove(0)
}

#[test]
fn precedence() {
    let want = Expr::Binary(
        BinOp::Add,
        n(1.0),
        Box::new(Expr::Binary(
            BinOp::Mul,
            n(2.0),
            Box::new(Expr::Binary(BinOp::Pow, n(3.0), n(2.0))),
        )),
    );
    assert_eq!(one("1+2*3^2"), want);
}

#[test]
fn pow_right_assoc() {
    let want = Expr::Binary(
        BinOp::Pow,
        n(2.0),
        Box::new(Expr::Binary(BinOp::Pow, n(3.0), n(2.0))),
    );
    assert_eq!(one("2^3^2"), want);
}

#[test]
fn assign_chain() {
    let want = Expr::Assign {
        op: AssignOp::Set,
        name: "a".into(),
        value: Box::new(Expr::Assign { op: AssignOp::Set, name: "b".into(), value: n(3.0) }),
    };
    assert_eq!(one("a=b=3"), want);
}

#[test]
fn compound() {
    let want = Expr::Assign { op: AssignOp::Add, name: "a".into(), value: n(1.0) };
    assert_eq!(one("a+=1"), want);
}

#[test]
fn call_args() {
    match one("if(a,b,c)") {
        Expr::Call { name, args, line, col } => {
            assert_eq!(name, "if");
            assert_eq!(args.len(), 3);
            assert_eq!((line, col), (1, 1));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn call_arg_sequence_is_block() {
    match one("while(i+=1; i<10)") {
        Expr::Call { args, .. } => {
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expr::Block(v) => assert_eq!(v.len(), 2),
                other => panic!("{other:?}"),
            }
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn paren_block() {
    match one("(a=1; b=2)") {
        Expr::Block(v) => assert_eq!(v.len(), 2),
        other => panic!("{other:?}"),
    }
    assert_eq!(one("(7)"), Expr::Num(7.0));
}

#[test]
fn empty_statements_ok() {
    assert_eq!(stmts(";;a;;"), vec![Expr::Var("a".into())]);
    assert_eq!(stmts(""), vec![]);
}

#[test]
fn unary_and_logic_levels() {
    assert_eq!(
        one("-a^2"),
        Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Unary(ms_expr::UnOp::Neg, Box::new(Expr::Var("a".into())))),
            n(2.0)
        )
    );
    // || is lower than &&
    match one("a||b&&c") {
        Expr::Binary(BinOp::LogOr, _, r) => assert!(matches!(*r, Expr::Binary(BinOp::LogAnd, _, _))),
        other => panic!("{other:?}"),
    }
}

#[test]
fn assign_to_non_var_errors() {
    let e = parse("1=2").unwrap_err();
    assert_eq!((e.line, e.col), (1, 2));
}

#[test]
fn unclosed_paren_located() {
    let e = parse("sin(1").unwrap_err();
    assert_eq!((e.line, e.col), (1, 6));
}

#[test]
fn nesting_cap() {
    let deep = format!("{}1{}", "(".repeat(300), ")".repeat(300));
    let e = parse(&deep).unwrap_err();
    assert!(e.message.contains("nesting too deep"));
    let unary = format!("{}1", "-".repeat(1000));
    assert!(parse(&unary).unwrap_err().message.contains("nesting too deep"));
    let ok = format!("{}1{}", "(".repeat(100), ")".repeat(100));
    assert!(parse(&ok).is_ok());
}
