use ms_expr::*;

fn eval(src: &str) -> f64 {
    let mut vt = VarTable::new();
    let p = compile(src, &mut vt).unwrap();
    let mut vars = vec![0.0; vt.len()];
    let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(1));
    let mut env = ExecEnv { vars: &mut vars, megabuf: &mut mb, gmegabuf: &mut gm, rng: &mut rng };
    let r = p.run(&mut env);
    assert!(!r.loop_cap_hit);
    r.value
}

#[test]
fn div_zero() {
    assert_eq!(eval("1/0"), 0.0);
    assert_eq!(eval("5%0"), 0.0);
    assert_eq!(eval("a=0; 1/a"), 0.0);
    assert_eq!(eval("a=0; 5%a"), 0.0);
}

#[test]
fn modulo_truncates() {
    assert_eq!(eval("7.9%3.2"), 1.0);
    assert_eq!(eval("-7%3"), -1.0);
}

#[test]
fn fuzzy_eq() {
    assert_eq!(eval("1==1.000001"), 1.0);
    assert_eq!(eval("1==1.0001"), 0.0);
    assert_eq!(eval("1!=1.000001"), 0.0);
    assert_eq!(eval("1!=1.0001"), 1.0);
}

#[test]
fn logic_short_circuit() {
    assert_eq!(eval("a=0; 0&&(a=5); a"), 0.0);
    assert_eq!(eval("a=0; b=0; b&&(a=5); a"), 0.0);
    assert_eq!(eval("a=0; b=1; b||(a=5); a"), 0.0);
    assert_eq!(eval("a=0; b=0; b||(a=5); a"), 5.0);
    assert_eq!(eval("a=0; b=1; b&&(a=5); a"), 5.0);
    assert_eq!(eval("3&&7"), 1.0);
    assert_eq!(eval("0||0"), 0.0);
    assert_eq!(eval("0||7"), 1.0);
}

#[test]
fn bitwise() {
    assert_eq!(eval("6&3"), 2.0);
    assert_eq!(eval("6|3"), 7.0);
    assert_eq!(eval("6.9&3.9"), 2.0);
}

#[test]
fn case_insensitive() {
    assert_eq!(eval("Foo=2; foo*FOO"), 4.0);
}

#[test]
fn block_value_is_last() {
    assert_eq!(eval("1;2;3"), 3.0);
    assert_eq!(eval(""), 0.0);
}

#[test]
fn assign_returns_value() {
    assert_eq!(eval("a=(b=4)+1; a+b"), 9.0);
}

#[test]
fn compound_ops() {
    assert_eq!(eval("a=10; a-=3; a*=2; a/=7; a%=3"), 2.0);
    assert_eq!(eval("a=1; a+=a+1"), 3.0);
}

#[test]
fn not_comparisons_pow_unary() {
    assert_eq!(eval("!0"), 1.0);
    assert_eq!(eval("!0.000001"), 1.0);
    assert_eq!(eval("!2"), 0.0);
    assert_eq!(eval("2^10"), 1024.0);
    assert_eq!(eval("-(3)+ +4"), 1.0);
    assert_eq!(eval("1<2"), 1.0);
    assert_eq!(eval("2<=1"), 0.0);
    assert_eq!(eval("2>1"), 1.0);
    assert_eq!(eval("1>=2"), 0.0);
}

#[test]
fn vars_persist_between_runs() {
    let mut vt = VarTable::new();
    let p = compile("x=x+1", &mut vt).unwrap();
    let mut vars = vec![0.0; vt.len()];
    let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(1));
    let mut last = 0.0;
    for _ in 0..3 {
        let mut env =
            ExecEnv { vars: &mut vars, megabuf: &mut mb, gmegabuf: &mut gm, rng: &mut rng };
        last = p.run(&mut env).value;
    }
    assert_eq!(last, 3.0);
    assert_eq!(vars[vt.lookup("X").unwrap().0 as usize], 3.0);
}

#[test]
fn registers_all_mentioned_vars() {
    let mut vt = VarTable::new();
    compile("a=1; 0&&(b=5); c", &mut vt).unwrap();
    assert_eq!(vt.len(), 3);
    assert_eq!(vt.name(vt.lookup("B").unwrap()), "b");
}

#[test]
fn unknown_function_errors_at_call() {
    let mut vt = VarTable::new();
    let e = compile("a=1;\n  foo(2)", &mut vt).unwrap_err();
    assert_eq!((e.line, e.col), (2, 3));
    assert!(e.message.contains("unknown function 'foo'"));
}

#[test]
fn program_is_send_sync_clone() {
    fn check<T: Clone + Send + Sync>() {}
    check::<Program>();
}

#[test]
fn deep_expression_uses_many_registers() {
    let src = format!("{}1{}", "1+(".repeat(60), ")".repeat(60));
    assert_eq!(eval(&src), 61.0);
}
