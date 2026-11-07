//! Scalar vs batched VM on a MilkDrop-style per-vertex program over a 64x48
//! mesh (3,072 lanes).

use criterion::{criterion_group, criterion_main, Criterion};
use ms_expr::*;
use std::hint::black_box;

const VERTEX: &str = "
d = sqrt(x*x + y*y);
ang = atan2(y, x);
zoom = zoom + 0.05*sin(d*6 - time*1.3)*(1 - d);
rot = rot + 0.02*cos(ang*3 + time);
dx = dx + 0.003*sin(y*8 + time*0.7);
dy = dy - 0.003*cos(x*8 - time*0.5);
";

const W: usize = 64;
const H: usize = 48;

fn setup() -> (Program, Lanes) {
    let mut vt = VarTable::new();
    let p = compile(VERTEX, &mut vt).expect("vertex program compiles");
    let n = W * H;
    let mut cols = vec![vec![0.0; n]; vt.len()];
    let col = |name: &str| vt.lookup(name).map(|v| v.0 as usize);
    for j in 0..H {
        for i in 0..W {
            let k = j * W + i;
            let (x, y) = (
                i as f64 / (W - 1) as f64 * 2.0 - 1.0,
                j as f64 / (H - 1) as f64 * 2.0 - 1.0,
            );
            for (name, v) in [("x", x), ("y", y), ("time", 3.7), ("zoom", 1.0)] {
                if let Some(c) = col(name) {
                    cols[c][k] = v;
                }
            }
        }
    }
    (p, Lanes { n, cols })
}

fn bench(c: &mut Criterion) {
    let (p, mut lanes) = setup();
    assert!(p.batchable());
    eprintln!("vertex program: {} ops, {} lanes", p.ops().len(), lanes.n);
    let (mut mb, mut gm, mut rng) = (Megabuf::new(), Megabuf::new(), Rng::seeded(1));
    let mut g = c.benchmark_group("vm/vertex_64x48");
    g.bench_function("scalar", |b| {
        let mut vars = vec![0.0; lanes.cols.len()];
        b.iter(|| {
            for k in 0..lanes.n {
                for (v, col) in vars.iter_mut().zip(&lanes.cols) {
                    *v = col[k];
                }
                let mut env = ExecEnv {
                    vars: &mut vars,
                    megabuf: &mut mb,
                    gmegabuf: &mut gm,
                    rng: &mut rng,
                };
                black_box(p.run(&mut env));
                for (v, col) in vars.iter().zip(lanes.cols.iter_mut()) {
                    col[k] = *v;
                }
            }
        })
    });
    let (p, mut lanes) = setup();
    g.bench_function("batched", |b| {
        b.iter(|| {
            let mut shared = BatchShared {
                megabuf: &mut mb,
                gmegabuf: &mut gm,
                rng: &mut rng,
            };
            black_box(p.run_batched(&mut lanes, &mut shared))
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
