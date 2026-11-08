use ms_expr::Rng;
use ms_expr::*;
use proptest::prelude::*;
use std::fmt;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compile_with(src: &str, names: &[&str]) -> (Program, VarTable) {
    let mut vt = VarTable::new();
    for n in names {
        vt.get_or_insert(n);
    }
    let p = compile(src, &mut vt).unwrap();
    (p, vt)
}

fn rbw_names(src: &str) -> Vec<String> {
    let mut vt = VarTable::new();
    let p = compile(src, &mut vt).unwrap();
    let ids = p.reads_before_write();
    assert!(
        ids.windows(2).all(|w| w[0].0 < w[1].0),
        "not sorted/deduped: {ids:?}"
    );
    ids.iter().map(|&id| vt.name(id).to_string()).collect()
}

/// Runs the scalar VM once per lane, in lane order, on a copy of `cols`.
fn run_scalar(p: &Program, cols: &[Vec<f64>], n: usize, seed: u64) -> (Vec<Vec<f64>>, bool) {
    let mut out = cols.to_vec();
    let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(seed));
    let mut capped = false;
    let mut vars = vec![0.0; cols.len()];
    for lane in 0..n {
        for (v, c) in vars.iter_mut().zip(&out) {
            *v = c[lane];
        }
        let mut env = ExecEnv {
            vars: &mut vars,
            megabuf: &mut mb,
            gmegabuf: &mut gm,
            rng: &mut rng,
        };
        capped |= p.run(&mut env).loop_cap_hit;
        for (v, c) in vars.iter().zip(out.iter_mut()) {
            c[lane] = *v;
        }
    }
    (out, capped)
}

fn run_batch(p: &Program, cols: &[Vec<f64>], n: usize, seed: u64) -> (Vec<Vec<f64>>, bool) {
    let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(seed));
    let mut lanes = Lanes {
        n,
        cols: cols.to_vec(),
    };
    let mut shared = BatchShared {
        megabuf: &mut mb,
        gmegabuf: &mut gm,
        rng: &mut rng,
    };
    let capped = p.run_batched(&mut lanes, &mut shared);
    (lanes.cols, capped)
}

fn random_cols(nvars: usize, n: usize, seed: u64, lo: f64, hi: f64) -> Vec<Vec<f64>> {
    let mut rng = Rng::seeded(seed);
    (0..nvars)
        .map(|_| (0..n).map(|_| lo + (hi - lo) * rng.next_f64()).collect())
        .collect()
}

/// Bitwise column comparison (NaN equals NaN); returns the first mismatch.
fn first_mismatch(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<(usize, usize, f64, f64)> {
    for (v, (ca, cb)) in a.iter().zip(b).enumerate() {
        for (i, (&x, &y)) in ca.iter().zip(cb).enumerate() {
            let same = x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
            if !same {
                return Some((v, i, x, y));
            }
        }
    }
    None
}

fn assert_same(src: &str, names: &[&str], cols: Vec<Vec<f64>>, n: usize) -> (Vec<Vec<f64>>, bool) {
    let (p, vt) = compile_with(src, names);
    assert_eq!(vt.len(), cols.len(), "test must provide one column per var");
    assert!(p.batchable());
    let (s, sc) = run_scalar(&p, &cols, n, 1);
    let (b, bc) = run_batch(&p, &cols, n, 1);
    assert_eq!(first_mismatch(&s, &b), None, "{src}");
    assert_eq!(sc, bc, "{src}: cap flag");
    (b, bc)
}

// ---------------------------------------------------------------------------
// reads_before_write / batchable
// ---------------------------------------------------------------------------

#[test]
fn rbw_detects_carry() {
    assert_eq!(rbw_names("t = t + x"), ["t", "x"]);
}

#[test]
fn rbw_clean() {
    assert_eq!(rbw_names("t = x*2; zoom = t"), ["x"]);
}

#[test]
fn rbw_branch() {
    let r = rbw_names("if(x, t=1, 0); y=t");
    assert!(r.contains(&"t".to_string()), "{r:?}");
    assert!(!r.contains(&"y".to_string()), "{r:?}");
    // Written on both arms: definitely written afterwards.
    assert_eq!(rbw_names("if(x, t=1, t=2); y=t"), ["x"]);
    // Short-circuit: the right side may not run.
    assert_eq!(rbw_names("x && (t=1); y=t"), ["x", "t"]);
}

#[test]
fn rbw_loops() {
    // A zero-trip loop body may not run.
    assert_eq!(rbw_names("loop(n, s=1); s"), ["n", "s"]);
    assert_eq!(rbw_names("s=0; loop(3, s+=1)"), Vec::<String>::new());
    // Read in iteration k of a value written in iteration k-1 (back edge).
    assert_eq!(rbw_names("while(t = u; u = 1; 0)"), ["u"]);
    // A while body normally runs at least once, but is skipped once an
    // earlier loop in the run hit the cap, so the analysis stays conservative.
    assert_eq!(rbw_names("while(u = 1; 0); u"), ["u"]);
}

#[test]
fn not_batchable_with_rand() {
    let mut vt = VarTable::new();
    assert!(!compile("x=rand(1)", &mut vt).unwrap().batchable());
    assert!(!compile("x=megabuf(1)", &mut vt).unwrap().batchable());
    assert!(!compile("gmegabuf(1)=x", &mut vt).unwrap().batchable());
    assert!(!compile("if(x, rand(2), 0)", &mut vt).unwrap().batchable());
    assert!(compile("x=sin(y)+min(x,2)", &mut vt).unwrap().batchable());
}

// ---------------------------------------------------------------------------
// run_batched
// ---------------------------------------------------------------------------

#[test]
fn straight_line_1000_lanes() {
    let names = ["x", "y", "t", "zoom"];
    let n = 1000;
    let cols = random_cols(4, n, 7, -2.0, 2.0);
    let src = "t = x*2 + sin(y); y = t - x/3; zoom = pow(abs(t), 0.5) + min(t, y); x %= 0.7";
    let (b, capped) = assert_same(src, &names, cols.clone(), n);
    assert!(!capped);
    // Spot-check a lane against the formula.
    let (x, y) = (cols[0][123], cols[1][123]);
    assert_eq!(b[2][123], x * 2.0 + y.sin());
}

#[test]
fn per_lane_trip_count() {
    let n = 600;
    let cols = random_cols(2, n, 9, 0.0, 2.0);
    let (b, capped) = assert_same("a = 0; loop(x*4, a+=1)", &["x", "a"], cols.clone(), n);
    assert!(!capped);
    for i in 0..n {
        let want = (cols[0][i] * 4.0).floor();
        assert_eq!(b[1][i], if want >= 1.0 { want } else { 0.0 });
    }
    // Trip counts differ per lane and the loops nest.
    assert_same(
        "a = 0; loop(x*3, loop(a+1, a+=0.5); b = a)",
        &["x", "a", "b"],
        random_cols(3, n, 4, 0.0, 2.0),
        n,
    );
}

#[test]
fn branches_and_logic_per_lane() {
    let n = 513; // not a multiple of the block size
    let cols = random_cols(4, n, 11, -2.0, 2.0);
    assert_same(
        "if(x > 0, (a = x; b = 1), if(y < 0.5, a = y, b = -1)); (a > 1) && (b = 7); (b < 0) || (y = 3)",
        &["x", "y", "a", "b"],
        cols,
        n,
    );
}

#[test]
fn loop_cap_per_lane() {
    let n = 10;
    let mut cols = vec![vec![0.0; n], vec![0.0; n]];
    cols[0][3] = 1.0;
    cols[0][8] = 1.0;
    let (b, capped) = assert_same("c = 0; while(c += 1; x > 0); c += 1", &["x", "c"], cols, n);
    assert!(capped);
    for i in 0..n {
        let want = if i == 3 || i == 8 {
            LOOP_CAP as f64 + 1.0
        } else {
            2.0
        };
        assert_eq!(b[1][i], want, "lane {i}");
    }
}

#[test]
fn loop_cap_not_shared_between_lanes() {
    // Each lane alone stays under the cap; together they would exceed it.
    let n = 4;
    let cols = vec![vec![0.0; n]];
    let (b, capped) = assert_same("s = 0; loop(600000, s += 1)", &["s"], cols, n);
    assert!(!capped);
    assert!(b[0].iter().all(|&s| s == 600000.0));
}

#[test]
fn empty_and_zero_lanes() {
    let (p, _) = compile_with("x = x + 1", &["x"]);
    let (b, c) = run_batch(&p, &[vec![]], 0, 1);
    assert_eq!(b, vec![Vec::<f64>::new()]);
    assert!(!c);
    let (p, _) = compile_with("", &[]);
    assert!(!run_batch(&p, &[], 5, 1).1);
}

/// Columns missing (or too short) read as zero and drop writes. Debug builds
/// flag it with a debug_assert; release builds must not panic.
#[test]
#[cfg_attr(debug_assertions, should_panic(expected = "Lanes"))]
fn short_cols_do_not_panic_in_release() {
    let (p, vt) = compile_with("y = x + 1; z = y * 2; x = 5", &["x", "y", "z"]);
    assert_eq!(vt.len(), 3);
    // Only x and y columns; y is shorter than n.
    let cols = vec![vec![1.0; 4], vec![0.0; 2]];
    let (b, capped) = run_batch(&p, &cols, 4, 1);
    assert!(!capped);
    assert_eq!(b[0], vec![5.0; 4]);
    assert_eq!(b[1], vec![2.0; 2]);
}

/// Non-batchable programs fall back to the scalar VM lane by lane in release
/// (and debug_assert in debug builds).
#[test]
#[cfg_attr(debug_assertions, should_panic(expected = "batchable"))]
fn non_batchable_falls_back_to_scalar() {
    let (p, _) = compile_with("x = rand(10) + megabuf(0); megabuf(0) = x", &["x"]);
    assert!(!p.batchable());
    let cols = vec![vec![0.0; 50]];
    let (s, _) = run_scalar(&p, &cols, 50, 3);
    let (b, _) = run_batch(&p, &cols, 50, 3);
    assert_eq!(first_mismatch(&s, &b), None);
}

// ---------------------------------------------------------------------------
// Property: scalar == batched
// ---------------------------------------------------------------------------

const VARS: [&str; 4] = ["x", "y", "a", "b"];
const BINOPS: [&str; 16] = [
    "+", "-", "*", "/", "%", "^", "==", "!=", "<", "<=", ">", ">=", "&", "|", "&&", "||",
];
const ASSIGNS: [&str; 6] = ["=", "+=", "-=", "*=", "/=", "%="];
const NUMS: [f64; 8] = [0.0, 1.0, 2.0, 0.5, -1.5, 3.25, 1e-6, 100.0];

#[derive(Clone, Debug)]
enum E {
    Num(usize),
    Var(usize),
    Neg(Box<E>),
    Not(Box<E>),
    Bin(usize, Box<E>, Box<E>),
    Assign(usize, usize, Box<E>),
    If(Box<E>, Box<E>, Box<E>),
    Sin(Box<E>),
    Min(Box<E>, Box<E>),
    Pow(Box<E>, Box<E>),
    Loop(u8, Box<E>),
    Seq(Box<E>, Box<E>),
}

impl fmt::Display for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            E::Num(i) => write!(f, "({})", NUMS[*i]),
            E::Var(v) => write!(f, "{}", VARS[*v]),
            E::Neg(a) => write!(f, "(-{a})"),
            E::Not(a) => write!(f, "(!{a})"),
            E::Bin(op, a, b) => write!(f, "({a} {} {b})", BINOPS[*op]),
            E::Assign(v, op, a) => write!(f, "({} {} {a})", VARS[*v], ASSIGNS[*op]),
            E::If(c, a, b) => write!(f, "if({c}, {a}, {b})"),
            E::Sin(a) => write!(f, "sin({a})"),
            E::Min(a, b) => write!(f, "min({a}, {b})"),
            E::Pow(a, b) => write!(f, "pow({a}, {b})"),
            E::Loop(n, a) => write!(f, "loop({n}, {a})"),
            E::Seq(a, b) => write!(f, "({a}; {b})"),
        }
    }
}

/// Shows the generated source text in proptest failure reports.
#[derive(Clone)]
struct Src(E);

impl fmt::Debug for Src {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

fn expr() -> impl Strategy<Value = E> {
    let leaf = prop_oneof![
        (0..NUMS.len()).prop_map(E::Num),
        (0..VARS.len()).prop_map(E::Var)
    ];
    leaf.prop_recursive(6, 64, 3, |inner| {
        let b = || inner.clone().prop_map(Box::new);
        prop_oneof![
            b().prop_map(E::Neg),
            b().prop_map(E::Not),
            (0..BINOPS.len(), b(), b()).prop_map(|(o, x, y)| E::Bin(o, x, y)),
            (0..VARS.len(), 0..ASSIGNS.len(), b()).prop_map(|(v, o, x)| E::Assign(v, o, x)),
            (b(), b(), b()).prop_map(|(c, x, y)| E::If(c, x, y)),
            b().prop_map(E::Sin),
            (b(), b()).prop_map(|(x, y)| E::Min(x, y)),
            (b(), b()).prop_map(|(x, y)| E::Pow(x, y)),
            (0u8..=4, b()).prop_map(|(n, x)| E::Loop(n, x)),
            (b(), b()).prop_map(|(x, y)| E::Seq(x, y)),
        ]
    })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    #[test]
    fn proptest_scalar_equals_batched(
        stmts in prop::collection::vec(expr().prop_map(Src), 1..4),
        seed in any::<u64>(),
    ) {
        let src = stmts.iter().map(|s| s.0.to_string()).collect::<Vec<_>>().join("; ");
        let (p, vt) = compile_with(&src, &VARS);
        prop_assert_eq!(vt.len(), 4);
        prop_assert!(p.batchable());
        let n = 300;
        let cols = random_cols(4, n, seed, -2.0, 2.0);
        let (s, sc) = run_scalar(&p, &cols, n, 1);
        let (b, bc) = run_batch(&p, &cols, n, 1);
        prop_assert_eq!(first_mismatch(&s, &b), None, "src: {}", src);
        prop_assert_eq!(sc, bc);
    }
}

#[test]
fn program_is_clone_send_sync() {
    fn check<T: Clone + Send + Sync>() {}
    check::<Program>();
}

#[test]
fn scratch_reuse_matches_fresh_runs() {
    // Two different programs through one scratch, in both orders: the reused
    // buffers must not leak state between calls.
    let (a, _) = compile_with(
        "y = x*2; while(c += 1; c < x*10); z = c",
        &["x", "y", "z", "c"],
    );
    let (b, _) = compile_with("y = if(x > 0.5, sin(x), cos(x))", &["x", "y", "z", "c"]);
    let n = 600;
    let cols = random_cols(4, n, 9, 0.0, 1.0);
    let mut scratch = BatchScratch::default();
    for p in [&a, &b, &a, &b] {
        let (fresh, fresh_capped) = run_batch(p, &cols, n, 1);
        let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(1));
        let mut lanes = Lanes {
            n,
            cols: cols.clone(),
        };
        let mut shared = BatchShared {
            megabuf: &mut mb,
            gmegabuf: &mut gm,
            rng: &mut rng,
        };
        let capped = p.run_batched_with(&mut lanes, &mut shared, &mut scratch);
        assert_eq!(capped, fresh_capped);
        assert_eq!(first_mismatch(&lanes.cols, &fresh), None);
    }
}

#[test]
fn compact_vars_renumbers_densely() {
    let mut vt = VarTable::new();
    for n in ["unused0", "x", "unused2", "y", "unused4", "k"] {
        vt.get_or_insert(n);
    }
    let p = compile("y = x + k; k = k + 1; w = y", &mut vt).unwrap();
    let (c, map) = p.compact_vars();
    let names: Vec<&str> = map.iter().map(|&v| vt.name(v)).collect();
    assert_eq!(names, ["x", "y", "k", "w"]);
    let rbw: Vec<&str> = c
        .reads_before_write()
        .iter()
        .map(|&v| vt.name(map[v.0 as usize]))
        .collect();
    assert_eq!(rbw, ["x", "k"]);
    assert!(c.ops().iter().all(|op| match *op {
        Op::LoadVar { var, .. } | Op::StoreVar { var, .. } => (var.0 as usize) < map.len(),
        _ => true,
    }));
    // Same results on compact columns as on the full table.
    let n = 300;
    let full = random_cols(vt.len(), n, 4, -2.0, 2.0);
    let compact: Vec<Vec<f64>> = map.iter().map(|v| full[v.0 as usize].clone()).collect();
    let (full_out, _) = run_batch(&p, &full, n, 1);
    let (compact_out, _) = run_batch(&c, &compact, n, 1);
    let picked: Vec<Vec<f64>> = map.iter().map(|v| full_out[v.0 as usize].clone()).collect();
    assert_eq!(first_mismatch(&compact_out, &picked), None);
    let (scalar_out, _) = run_scalar(&c, &compact, n, 1);
    assert_eq!(first_mismatch(&compact_out, &scalar_out), None);
}
