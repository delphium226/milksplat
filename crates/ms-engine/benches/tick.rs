//! One full preset tick (frame code, a ~60-op batched vertex program over the
//! mesh, the warp UVs, uniforms and draw list) at the default and the
//! largest mesh size.

use criterion::{criterion_group, criterion_main, Criterion};
use ms_analysis::AudioFrame;
use ms_engine::{Globals, PresetInstance, TickInput, VertexPath};
use ms_preset::{compile_preset, parse};
use std::hint::black_box;
use std::sync::Arc;

const CODE: &str = r#"
[code]
frame = "q1 = sin(time*0.7)*0.05; zoom = 1 + bass*0.02"
vertex = '''
d = sqrt(sqr(x - 0.5) + sqr(y - 0.5));
zoom = zoom + 0.05*sin(d*6 - time*1.3) + q1*rad;
rot = rot + 0.02*cos(ang*3 + time);
dx = dx + 0.003*sin(y*8 + time*0.7);
dy = dy - 0.003*cos(x*8 - time*0.5);
'''
"#;

fn instance(cols: u32, rows: u32) -> PresetInstance {
    let toml = format!("[params]\nmesh = [{cols}, {rows}]\n{CODE}");
    let (p, _) = parse(toml.as_bytes()).expect("bench preset parses");
    let p = Arc::new(compile_preset(p));
    let ops = p.vertex.as_ref().map_or(0, |v| v.ops().len());
    eprintln!(
        "vertex program: {ops} ops, {} vertices",
        (cols + 1) * (rows + 1)
    );
    PresetInstance::new(p, 1, [192, 144])
}

fn bench(c: &mut Criterion) {
    let audio = AudioFrame::default();
    let mut g = c.benchmark_group("engine");
    for (name, cols, rows) in [("tick_64x48", 64, 48), ("tick_192x144", 192, 144)] {
        let mut inst = instance(cols, rows);
        let mut globals = Globals::default();
        let mut frame = 0u64;
        g.bench_function(name, |b| {
            b.iter(|| {
                frame += 1;
                let input = TickInput {
                    time: frame as f64 / 60.0,
                    fps: 60.0,
                    frame,
                    progress: 0.0,
                    audio: &audio,
                    aspect: [1.0, 1.0],
                    pixels: [1920, 1080],
                };
                black_box(inst.tick(&input, &mut globals).mesh.uv[0])
            })
        });
        assert_eq!(inst.vertex_path(), VertexPath::Batched);
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
