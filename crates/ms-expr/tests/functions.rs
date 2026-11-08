use ms_expr::*;
use std::time::Instant;

fn run_with(
    src: &str,
    mb: &mut Megabuf,
    gm: &mut Megabuf,
    rng: &mut Rng,
) -> (RunResult, VarTable, Vec<f64>) {
    let mut vt = VarTable::new();
    let p = compile(src, &mut vt).unwrap();
    let mut vars = vec![0.0; vt.len()];
    let mut env = ExecEnv {
        vars: &mut vars,
        megabuf: mb,
        gmegabuf: gm,
        rng,
    };
    let r = p.run(&mut env);
    (r, vt, vars)
}

fn run(src: &str) -> RunResult {
    let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(1));
    run_with(src, &mut mb, &mut gm, &mut rng).0
}

fn eval(src: &str) -> f64 {
    let r = run(src);
    assert!(!r.loop_cap_hit, "{src}");
    r.value
}

fn var(src: &str, name: &str) -> f64 {
    let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(1));
    let (_, vt, vars) = run_with(src, &mut mb, &mut gm, &mut rng);
    vars[vt.lookup(name).unwrap().0 as usize]
}

fn err(src: &str) -> CompileError {
    compile(src, &mut VarTable::new()).unwrap_err()
}

#[test]
fn trig() {
    assert!((eval("atan2(1,1)") - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    assert!((eval("y=1; x=1; atan2(y,x)") - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
}

#[test]
fn int_floors() {
    assert_eq!(eval("int(-1.5)"), -2.0);
    assert_eq!(eval("a=-1.5; int(a)"), -2.0);
}

#[test]
fn sigmoid() {
    assert_eq!(eval("sigmoid(0,5)"), 0.5);
    assert_eq!(eval("a=0; sigmoid(a,5)"), 0.5);
}

#[test]
fn sign_values() {
    assert_eq!(eval("a=-3; sign(a)"), -1.0);
    assert_eq!(eval("a=0; sign(a)"), 0.0);
    assert_eq!(eval("a=9; sign(a)"), 1.0);
}

#[test]
fn if_lazy() {
    assert_eq!(eval("a=0; if(1, 2, a=9); a"), 0.0);
    assert_eq!(eval("a=0; c=1; if(c, 2, a=9); a"), 0.0);
    assert_eq!(eval("a=0; c=0; if(c, a=9, 2); a"), 0.0);
    assert_eq!(eval("c=0; if(c, 1, 7)"), 7.0);
    assert_eq!(eval("c=1; if(c, 1, 7)"), 1.0);
    assert_eq!(eval("c=1; 10 + if(c, 1, 7)"), 11.0);
}

#[test]
fn loop_counts() {
    assert_eq!(eval("n=0; loop(5, n+=1); n"), 5.0);
    assert_eq!(eval("n=0; k=5; loop(k, n+=1); n"), 5.0);
    assert_eq!(eval("n=0; loop(2.9, n+=1); n"), 2.0);
}

#[test]
fn loop_negative() {
    assert_eq!(eval("n=0; loop(-3, n+=1); n"), 0.0);
    assert_eq!(eval("n=0; k=-3; loop(k, n+=1); n"), 0.0);
    assert_eq!(eval("n=0; k=0.5; loop(k, n+=1); n"), 0.0);
}

#[test]
fn loop_n_evaluated_once() {
    assert_eq!(var("k=0; m=0; loop(k+=1; 3, m+=1)", "k"), 1.0);
    assert_eq!(var("k=0; m=0; loop(k+=1; 3, m+=1)", "m"), 3.0);
    // Mutating the count variable in the body does not change the trip count.
    assert_eq!(eval("n=3; c=0; loop(n, n+=1; c+=1); c"), 3.0);
}

#[test]
fn nested_loops() {
    assert_eq!(eval("s=0; loop(4, loop(5, s+=1)); s"), 20.0);
    assert_eq!(
        eval("s=0; i=0; loop(3, i+=1; j=0; loop(i, j+=1; s+=1)); s"),
        6.0
    );
}

#[test]
fn while_runs() {
    assert_eq!(eval("i=0; while(i+=1; i<10); i"), 10.0);
    // Body always runs at least once.
    assert_eq!(eval("i=0; while(i+=1; 0); i"), 1.0);
}

#[test]
fn loop_cap() {
    let t = Instant::now();
    let r = run("while(1)");
    assert!(r.loop_cap_hit);
    assert!(t.elapsed().as_secs_f64() < 1.0, "{:?}", t.elapsed());
}

#[test]
fn loop_cap_exact_and_global() {
    let r = run("x=0; loop(1048576, x+=1); x");
    assert!(!r.loop_cap_hit);
    assert_eq!(r.value, LOOP_CAP as f64);
    let r = run("x=0; loop(1e9, x+=1); x");
    assert!(r.loop_cap_hit);
    assert_eq!(r.value, LOOP_CAP as f64);
    // After the cap every later loop exits at once, including nested and while.
    let r = run("y=0; loop(2000000, 1); loop(5, y+=1); while(y+=1); y");
    assert!(r.loop_cap_hit);
    assert_eq!(r.value, 0.0);
    let r = run("x=0; loop(2000, loop(2000, x+=1)); x");
    assert!(r.loop_cap_hit);
    assert!(r.value <= LOOP_CAP as f64);
    // The outer loop stops once the inner loop hits the cap.
    let outer = var("x=0; o=0; loop(2000, o+=1; loop(2000, x+=1))", "o");
    assert!(outer > 1.0 && outer < 1000.0, "outer ran {outer} times");
    // The cap is per run.
    let mut vt = VarTable::new();
    let p = compile("loop(2000000, 1)", &mut vt).unwrap();
    let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(1));
    for _ in 0..2 {
        let mut env = ExecEnv {
            vars: &mut [],
            megabuf: &mut mb,
            gmegabuf: &mut gm,
            rng: &mut rng,
        };
        assert!(p.run(&mut env).loop_cap_hit);
    }
}

#[test]
fn megabuf_rw() {
    assert_eq!(eval("megabuf(10)=3; megabuf(10.7)"), 3.0);
    assert_eq!(eval("i=10; megabuf(i)=3; megabuf(i)"), 3.0);
    assert_eq!(eval("i=10; (megabuf(i)=3) + 1"), 4.0);
    // megabuf and gmegabuf are separate buffers.
    assert_eq!(eval("megabuf(1)=3; gmegabuf(1)"), 0.0);
}

#[test]
fn megabuf_compound() {
    assert_eq!(eval("megabuf(2)=5; megabuf(2)+=3; megabuf(2)"), 8.0);
    assert_eq!(eval("i=2; megabuf(i)=5; megabuf(i)*=3; megabuf(i)"), 15.0);
    assert_eq!(eval("i=2; gmegabuf(i)=5; gmegabuf(i)-=1; gmegabuf(i)"), 4.0);
    assert_eq!(eval("i=2; megabuf(i)=9; megabuf(i)/=3; megabuf(i)"), 3.0);
    assert_eq!(eval("i=2; megabuf(i)=9; megabuf(i)%=4; megabuf(i)"), 1.0);
    assert_eq!(eval("i=2; megabuf(i)=1; megabuf(i)+=4"), 5.0);
    // The index expression is evaluated once.
    assert_eq!(var("i=0; megabuf(i+=1)+=5", "i"), 1.0);
    assert_eq!(eval("i=0; megabuf(i+=1)+=5; megabuf(1)"), 5.0);
}

#[test]
fn megabuf_oob() {
    assert_eq!(
        eval("megabuf(-1)=3; megabuf(2000000)=3; megabuf(-1)+megabuf(2000000)"),
        0.0
    );
    assert_eq!(
        eval("a=-1; b=2000000; megabuf(a)=3; megabuf(b)=3; megabuf(a)+megabuf(b)"),
        0.0
    );
    assert_eq!(eval("a=-1; megabuf(a)+=3; megabuf(a)"), 0.0);
}

#[test]
fn gmegabuf_shared() {
    let mut vt = VarTable::new();
    let w = compile("gmegabuf(5)=42; megabuf(5)=1", &mut vt).unwrap();
    let r = compile("gmegabuf(5)+megabuf(5)", &mut vt).unwrap();
    let mut gm = Megabuf::new();
    let (mut mb1, mut mb2, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(1));
    w.run(&mut ExecEnv {
        vars: &mut [],
        megabuf: &mut mb1,
        gmegabuf: &mut gm,
        rng: &mut rng,
    });
    let out = r.run(&mut ExecEnv {
        vars: &mut [],
        megabuf: &mut mb2,
        gmegabuf: &mut gm,
        rng: &mut rng,
    });
    assert_eq!(out.value, 42.0);
}

#[test]
fn rand_range() {
    let mut vt = VarTable::new();
    let p = compile("rand(10)", &mut vt).unwrap();
    let q = compile("rand(0.2)", &mut vt).unwrap();
    let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(7));
    let mut seen_high = false;
    for _ in 0..10_000 {
        let mut env = ExecEnv {
            vars: &mut [],
            megabuf: &mut mb,
            gmegabuf: &mut gm,
            rng: &mut rng,
        };
        let v = p.run(&mut env).value;
        assert!((0.0..10.0).contains(&v), "{v}");
        seen_high |= v > 5.0;
        let v = q.run(&mut env).value;
        assert!((0.0..1.0).contains(&v), "{v}");
    }
    assert!(seen_high);
}

#[test]
fn rand_deterministic() {
    let seq = |seed| {
        let mut vt = VarTable::new();
        let p = compile("rand(1000)", &mut vt).unwrap();
        let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(seed));
        (0..20)
            .map(|_| {
                p.run(&mut ExecEnv {
                    vars: &mut [],
                    megabuf: &mut mb,
                    gmegabuf: &mut gm,
                    rng: &mut rng,
                })
                .value
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(seq(5), seq(5));
    assert_ne!(seq(5), seq(6));
}

#[test]
fn logic_functions() {
    assert_eq!(eval("a=3; b=0; band(a,b)"), 0.0);
    assert_eq!(eval("a=3; b=2; band(a,b)"), 1.0);
    assert_eq!(eval("a=3; b=0; bor(a,b)"), 1.0);
    assert_eq!(eval("a=0; b=0; bor(a,b)"), 0.0);
    assert_eq!(eval("a=0; bnot(a)"), 1.0);
    assert_eq!(eval("a=2; bnot(a)"), 0.0);
    assert_eq!(eval("a=1; equal(a,1.000001)"), 1.0);
    assert_eq!(eval("a=1; equal(a,1.01)"), 0.0);
    assert_eq!(eval("a=2; above(a,1)"), 1.0);
    assert_eq!(eval("a=1; above(a,1)"), 0.0);
    assert_eq!(eval("a=0; below(a,1)"), 1.0);
    assert_eq!(eval("a=0; b=0; band(a, b=5); b"), 0.0);
    assert_eq!(eval("a=1; b=0; bor(a, b=5); b"), 0.0);
}

#[test]
fn unknown_fn_located() {
    let e = err("a=foo(1)");
    assert_eq!((e.line, e.col), (1, 3));
    assert!(e.message.contains("foo"), "{e}");
    // Dead branches are still checked.
    assert!(compile("0&&foo(1)", &mut VarTable::new()).is_err());
    assert!(compile("if(1, 2, foo(1))", &mut VarTable::new()).is_err());
    assert!(compile("if(1, 2, sin(1,2))", &mut VarTable::new()).is_err());
}

#[test]
fn arity_checked() {
    let e = err("sin(1,2)");
    assert_eq!((e.line, e.col), (1, 1));
    assert_eq!(e.message, "'sin' expects 1 argument(s)");
    let e = err("x=1;\n  atan2(1)");
    assert_eq!(e.message, "'atan2' expects 2 argument(s)");
    assert_eq!((e.line, e.col), (2, 3));
    for bad in [
        "sin()",
        "if(1,2)",
        "loop(1)",
        "while()",
        "rand(1,2)",
        "megabuf(1,2)",
        "megabuf(1,2)=3",
        "sin(1)=3",
    ] {
        assert!(compile(bad, &mut VarTable::new()).is_err(), "{bad}");
    }
}

#[test]
fn case_insensitive_names() {
    assert_eq!(eval("SIN(0)+Max(1,2)"), 2.0);
}

/// Every pure builtin: constant-folded result must equal the VM result.
#[test]
fn fold_matches_vm() {
    let unary = [
        "sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "exp", "log", "log10", "abs", "sign",
        "floor", "ceil", "int", "sqr", "bnot",
    ];
    let binary = [
        "atan2", "pow", "min", "max", "sigmoid", "band", "bor", "equal", "above", "below",
    ];
    let xs = [-2.5, -1.0, -0.3, 0.0, 0.3, 0.7, 1.0, 2.0, 7.5, 1e3];
    let same = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
    for f in unary {
        for x in xs {
            let folded = eval(&format!("{f}({x:?})"));
            let vm = eval(&format!("a={x:?}; {f}(a)"));
            assert!(same(folded, vm), "{f}({x}): fold {folded} vm {vm}");
            assert!(!folded.is_nan(), "{f}({x}) is NaN");
        }
    }
    for f in binary {
        for x in xs {
            for y in xs {
                let folded = eval(&format!("{f}({x:?},{y:?})"));
                let vm = eval(&format!("a={x:?}; b={y:?}; {f}(a,b)"));
                assert!(same(folded, vm), "{f}({x},{y}): fold {folded} vm {vm}");
                assert!(!folded.is_nan(), "{f}({x},{y}) is NaN");
            }
        }
    }
}

#[test]
fn builtin_values() {
    assert_eq!(eval("a=-4; abs(a)"), 4.0);
    assert_eq!(eval("a=2; b=3; min(a,b)"), 2.0);
    assert_eq!(eval("a=2; b=3; max(a,b)"), 3.0);
    assert_eq!(eval("a=2.1; ceil(a)"), 3.0);
    assert_eq!(eval("a=2.9; floor(a)"), 2.0);
    assert_eq!(eval("a=3; sqr(a)"), 9.0);
    assert_eq!(eval("a=9; sqrt(a)"), 3.0);
    assert_eq!(eval("a=2; b=10; pow(a,b)"), 1024.0);
    assert_eq!(eval("a=0; exp(a)"), 1.0);
    assert_eq!(eval("a=1; log(a)"), 0.0);
    assert_eq!(eval("a=100; log10(a)"), 2.0);
}

#[test]
fn nonfinite_inputs_never_give_nan() {
    assert_eq!(eval("pow(-2.5, 0.3)"), 0.0);
    assert_eq!(eval("a=-2.5; b=0.3; pow(a,b)"), 0.0);
    assert_eq!(eval("pow(-2, 3)"), -8.0);
    assert_eq!(eval("a=0; b=1/0; sigmoid(a,b)"), eval("0.5"));
    assert_eq!(eval("sigmoid(0, 1e999)"), 0.5);
    assert_eq!(eval("a=0; b=1e999; sigmoid(a,b)"), 0.5);
    assert_eq!(eval("a=1e999; b=0; sigmoid(a,b)"), 0.5);
    assert_eq!(
        eval("a=1e999; b=1e999; atan2(a,b)"),
        std::f64::consts::FRAC_PI_4
    );
    assert!(!eval("a=1e999; sin(a)").is_nan());
    assert!(!eval("a=-1e999; b=1e999; pow(a,b)").is_nan());
}

fn nested_sin(depth: usize) -> String {
    format!("{}x{}", "sin(".repeat(depth), ")".repeat(depth))
}

#[test]
fn deep_call_nesting() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let mut vt = VarTable::new();
            let p = compile(&nested_sin(120), &mut vt).unwrap();
            let mut vars = vec![0.5; vt.len()];
            let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(1));
            let mut env = ExecEnv {
                vars: &mut vars,
                megabuf: &mut mb,
                gmegabuf: &mut gm,
                rng: &mut rng,
            };
            assert!(p.run(&mut env).value.is_finite());
            assert!(compile(&nested_sin(2000), &mut VarTable::new()).is_err());
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn loop_budget_shared_across_runs() {
    let mut vt = VarTable::new();
    let p = compile("x = 0; loop(600000, x += 1); x", &mut vt).unwrap();
    let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(1));
    let mut vars = vec![0.0; vt.len()];
    let mut budget = LoopBudget::new(1_000_000);
    let mut values = Vec::new();
    for _ in 0..3 {
        let mut env = ExecEnv {
            vars: &mut vars,
            megabuf: &mut mb,
            gmegabuf: &mut gm,
            rng: &mut rng,
        };
        let r = p.run_with_budget(&mut env, &mut budget);
        assert!(!r.loop_cap_hit, "the per-run cap is not what stopped it");
        values.push(r.value);
    }
    // The first run fits; the second gets the rest; the third gets nothing.
    assert_eq!(values, vec![600000.0, 400000.0, 0.0]);
    assert!(budget.exhausted());
    assert_eq!(budget.remaining(), 0);
}

#[test]
fn loop_budget_exhausted_skips_while_body() {
    let mut budget = LoopBudget::new(0);
    let mut vt = VarTable::new();
    let p = compile("y = 0; while(y += 1; 1); y", &mut vt).unwrap();
    let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(1));
    let mut vars = vec![0.0; vt.len()];
    let mut env = ExecEnv {
        vars: &mut vars,
        megabuf: &mut mb,
        gmegabuf: &mut gm,
        rng: &mut rng,
    };
    assert_eq!(p.run_with_budget(&mut env, &mut budget).value, 0.0);
    assert!(budget.exhausted());
}

#[test]
fn loop_budget_share_and_refund() {
    let mut b = LoopBudget::new(1000);
    let s1 = b.share(1, 4);
    assert_eq!((s1.remaining(), b.remaining()), (250, 750));
    let mut s2 = b.share(3, 3);
    assert_eq!((s2.remaining(), b.remaining()), (750, 0));
    b.refund(s1);
    assert_eq!(b.remaining(), 250);
    assert!(!b.exhausted());
    // An exhausted share exhausts the parent.
    let mut s3 = LoopBudget::new(0);
    assert!(!s3.exhausted());
    let mut vt = VarTable::new();
    let p = compile("loop(5, 1)", &mut vt).unwrap();
    let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(1));
    p.run_with_budget(
        &mut ExecEnv {
            vars: &mut [],
            megabuf: &mut mb,
            gmegabuf: &mut gm,
            rng: &mut rng,
        },
        &mut s3,
    );
    assert!(s3.exhausted());
    s2 = LoopBudget::new(s2.remaining());
    b.refund(s2);
    b.refund(s3);
    assert!(b.exhausted());
    assert!(b.share(1, 1).exhausted());
    // Unlimited shares do not overflow.
    let mut u = LoopBudget::unlimited();
    assert_eq!(u.share(1, 2).remaining(), u64::MAX / 2);
}
