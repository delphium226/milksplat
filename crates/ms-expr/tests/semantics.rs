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
    assert_eq!(eval("z=0; 1/z"), 0.0);
    assert_eq!(eval("z=0; 5%z"), 0.0);
}

#[test]
fn modulo_truncates() {
    assert_eq!(eval("7.9%3.2"), 1.0);
    assert_eq!(eval("-7%3"), -1.0);
    assert_eq!(eval("a=7.9; b=3.2; a%b"), 1.0);
    assert_eq!(eval("a=-7; a%3"), -1.0);
}

#[test]
fn fuzzy_eq() {
    assert_eq!(eval("1==1.000001"), 1.0);
    assert_eq!(eval("1==1.0001"), 0.0);
    assert_eq!(eval("1!=1.000001"), 0.0);
    assert_eq!(eval("1!=1.0001"), 1.0);
    assert_eq!(eval("a=1; a==1.000001"), 1.0);
    assert_eq!(eval("a=1; a==1.0001"), 0.0);
    assert_eq!(eval("a=1; a!=1.000001"), 0.0);
    assert_eq!(eval("a=1; a!=1.0001"), 1.0);
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
    assert_eq!(eval("a=6; a&3"), 2.0);
    assert_eq!(eval("a=6; a|3"), 7.0);
    assert_eq!(eval("a=6.9; b=3.9; a&b"), 2.0);
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
    // constant (folded)
    assert_eq!(eval("!0"), 1.0);
    assert_eq!(eval("2^10"), 1024.0);
    // variable operands (executed by the VM)
    assert_eq!(eval("z=0; !z"), 1.0);
    assert_eq!(eval("z=0.000001; !z"), 1.0);
    assert_eq!(eval("z=2; !z"), 0.0);
    assert_eq!(eval("a=2; b=10; a^b"), 1024.0);
    assert_eq!(eval("a=3; -a+ +4"), 1.0);
    assert_eq!(eval("a=1; b=2; a<b"), 1.0);
    assert_eq!(eval("a=1; b=2; b<=a"), 0.0);
    assert_eq!(eval("a=1; b=2; b>a"), 1.0);
    assert_eq!(eval("a=1; b=2; a>=b"), 0.0);
    assert_eq!(eval("a=2; b=2; a<=b"), 1.0);
    assert_eq!(eval("a=2; b=2; a>=b"), 1.0);
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
    let src = format!("x=1; {}x{}", "1+(".repeat(40), ")".repeat(40));
    assert_eq!(eval(&src), 41.0);
    // Folds to a single constant.
    let src = format!("{}1{}", "1+(".repeat(60), ")".repeat(60));
    assert_eq!(eval(&src), 61.0);
}

#[test]
fn fold_and_vm_agree_for_every_binop() {
    let ops = [
        "+", "-", "*", "/", "%", "^", "==", "!=", "<", "<=", ">", ">=", "&", "|", "&&", "||",
    ];
    let pairs = [
        (6.0, 3.0),
        (7.9, 3.2),
        (-7.0, 3.0),
        (1.0, 1.000001),
        (0.0, 0.0),
        (5.0, 0.0),
        (0.0, 2.0),
        (2.5, -1.5),
    ];
    for op in ops {
        for (a, b) in pairs {
            let folded = eval(&format!("({a:?}){op}({b:?})"));
            let vm = eval(&format!("a=({a:?}); b=({b:?}); a{op}b"));
            assert!(
                folded == vm || (folded.is_nan() && vm.is_nan()),
                "{a} {op} {b}: folded {folded} vs vm {vm}"
            );
        }
    }
}

#[test]
fn dead_branch_still_checks_calls() {
    for src in ["0&&foo(1)", "1||foo(1)"] {
        let mut vt = VarTable::new();
        let e = compile(src, &mut vt).unwrap_err();
        assert!(e.message.contains("unknown function 'foo'"), "{src}: {e}");
        assert_eq!((e.line, e.col), (1, src.find("foo").unwrap() as u32 + 1));
    }
}

fn chain(n: usize) -> String {
    vec!["a"; n].join("+")
}

#[test]
fn very_long_chain_is_rejected_not_crashed() {
    let mut vt = VarTable::new();
    let e = compile(&chain(5000), &mut vt).unwrap_err();
    assert!(e.message.contains("too deeply nested"), "{e}");
    let e = compile(&format!("x={}", chain(5000)), &mut vt).unwrap_err();
    assert!(e.message.contains("too deeply nested"), "{e}");
}

#[test]
fn thousand_term_chain_compiles_and_runs_on_small_stack() {
    // Default test threads have a 2 MB stack; make that explicit.
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(|| {
            let src = format!("a=1; {}", chain(1000));
            assert_eq!(eval(&src), 1000.0);
        })
        .unwrap()
        .join()
        .unwrap();
}
