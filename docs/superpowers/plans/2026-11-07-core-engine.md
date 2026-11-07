# Milksplat Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build sub-project 1 of Milksplat: a Windows MilkDrop-style visualizer that captures system audio, runs `.splat` presets in our EEL2-compatible expression language, renders them with wgpu, and ships as a portable zip.

**Architecture:**
- A Cargo workspace of seven crates (§3.1 of the spec): pure-logic crates first (`ms-expr`, `ms-analysis`, `ms-preset`), then the device-facing ones (`ms-audio`, `ms-render`), then `ms-engine`, which ties presets to the renderer, and finally the `milksplat` app.
- Every crate is tested on its own. Rendering is tested headlessly against golden images, with WARP as the fallback adapter.

**Tech Stack:** Rust stable (edition 2021), wgpu + naga (WGSL), cpal (WASAPI loopback), `windows` crate, realfft, winit, rtrb, serde + toml, notify, tracing + tracing-appender, image, rayon, font8x8, criterion, proptest, bytemuck.

**Spec:** `docs/superpowers/specs/2026-11-07-core-engine-design.md`. Section references below (§n) point into it, and executors read both documents.

## Prerequisites (human, once)

Install Rust:
1. Install Rust with rustup from https://rustup.rs (default `stable-x86_64-pc-windows-msvc`).
2. When asked, install the **Visual Studio Build Tools** with the "Desktop development with C++" workload.
3. Check that `cargo --version` prints a version in a new terminal.

## Global Constraints

- Platform: Windows 10/11 x64, target `x86_64-pc-windows-msvc`. Windows-only code lives in `#[cfg(windows)]` modules inside `ms-audio` and `milksplat` only.
- Licence: every crate's `Cargo.toml` has `license = "MIT OR Apache-2.0"`; repo root has `LICENSE-MIT` and `LICENSE-APACHE`.
- Dependencies: add with `cargo add` at the latest released version and declare them once in `[workspace.dependencies]`. Crates use `dep.workspace = true`.
- No `unwrap()`/`expect()` on any path reachable from user data (preset files, config, audio, GPU errors). Tests may unwrap.
- The audio callback never allocates, locks or blocks.
- Simulation tick: 60 Hz default; composite runs at display rate (§3.3).
- Expression values are `f64`; GPU-facing values are `f32`, converted at the engine boundary, where NaN/±Inf become `0.0`.
- Logging uses `tracing`. User-facing errors never panic.

## Review Focus

1. **Preset files saved by other editors** (UTF-8 BOM, CRLF line endings, tabs, non-UTF-8 bytes) must load or report a located error, never panic. Test: Task 11.
2. **Code that produces NaN/Inf or extreme values** (e.g. `zoom = 1/0.000001`, `x = sqrt(-1)`) must not poison the feedback buffer or crash; the frame renders. Tests: Task 20 (engine sanitising) and Task 14 (fp16 clamp).
3. **Empty or tiny preset libraries** (zero or one preset; every preset banned) must not panic. The app shows the built-in default preset and auto-advance does nothing. Tests: Task 12 and Task 22.
4. **Extreme window sizes** (1×1, minimized 0×0, larger than the GPU's max texture size) must not crash. Internal size is clamped to `[16, max_texture_dimension_2d]`. Tests: Task 14 and Task 18.
5. **Unusual audio formats** (mono, 3/6/8 channels, 22.05 kHz and 192 kHz) must give valid band values. Band edges above Nyquist are clamped. Tests: Task 9 and Task 7.

---

## File structure

```
Cargo.toml                      workspace + [workspace.dependencies] + release profile
.cargo/config.toml              static CRT rustflags
rust-toolchain.toml             channel = "stable"
LICENSE-MIT, LICENSE-APACHE, README.md
scripts/package.ps1             build + zip
presets/*.splat                 ~10 bundled original presets
textures/                       bundled textures (random/ subfolder)
crates/
  ms-expr/      src/{lib,lexer,parser,ast,compile,bytecode,vm,batch,vars,rng}.rs   tests/  benches/
  ms-analysis/  src/{lib,history,spectrum,bands,beat}.rs                           tests/
  ms-audio/     src/{lib,downmix,ring,backoff}.rs  src/win/{capture,notify,bluetooth}.rs   tests/
  ms-preset/    src/{lib,format,compile,library,select}.rs                         tests/
  ms-render/    src/{lib,gpu,feedback,warp,shaders,prelude.wgsl,composite,blur,
                     textures,noise,waves,shapes,borders,motion,text,golden}.rs    tests/
  ms-engine/    src/{lib,env,tick,clock,blend,director,loader,governor}.rs         tests/  benches/
  milksplat/    src/{main,app,render_thread,hotkeys,config,paths,logging,crash,keepawake,hotreload}.rs  tests/
```

---

### Task 1: Workspace skeleton and `ms-expr` lexer

**Files:**
- Create: `Cargo.toml`, `rust-toolchain.toml`, `.cargo/config.toml`, `LICENSE-MIT`, `LICENSE-APACHE`, all seven crate skeletons (`cargo new --lib`; `milksplat` with `--bin`), `crates/ms-expr/src/lexer.rs`
- Test: `crates/ms-expr/tests/lexer.rs`

**Interfaces:**
- Produces:
  - `pub enum Tok { Num(f64), Ident(String), Op(&'static str), LParen, RParen, Comma, Semi, Eof }`
  - `pub struct Token { pub tok: Tok, pub line: u32, pub col: u32 }`
  - `pub fn lex(src: &str) -> Result<Vec<Token>, CompileError>`
  - `pub struct CompileError { pub line: u32, pub col: u32, pub message: String }` (in `lib.rs`, `impl Display + Error`)

- [ ] **Step 1: Create the workspace.** `Cargo.toml` with `members = ["crates/*"]`, `resolver = "2"`, and `[profile.release] lto = "fat", codegen-units = 1`. `.cargo/config.toml` sets `[target.x86_64-pc-windows-msvc] rustflags = ["-C", "target-feature=+crt-static"]`. Run `cargo build` and expect success.
- [ ] **Step 2: Write failing tests**

```rust
#[test] fn numbers() { assert_eq!(nums("1 .5 2.5e3 0x1F $x10"), vec![1.0, 0.5, 2500.0, 31.0, 16.0]); }
#[test] fn constants() { assert_eq!(nums("$pi $e $phi"), vec![std::f64::consts::PI, std::f64::consts::E, 1.618033988749895]); }
#[test] fn idents_lowercased() { assert!(matches!(&lex("Zoom")[0].tok, Tok::Ident(s) if s == "zoom")); }
#[test] fn ops_longest_match() { assert_eq!(ops("+= -= *= /= %= == != <= >= && || = < > + - * / % ^ & | !"), 23); }
#[test] fn comments_skipped() { assert_eq!(lex("a // x\n/* y\n */ b").len(), 3); } // a, b, Eof
#[test] fn bom_and_crlf() { let t = lex("\u{feff}a;\r\nb"); assert_eq!((t[2].line, t[2].col), (2, 1)); }
#[test] fn bad_char_located() { let e = lex_err("a = 1 @"); assert_eq!((e.line, e.col), (1, 7)); }
```
(`nums`, `ops`, `lex`, `lex_err` are small helpers in the test file.)

- [ ] **Step 3: Run `cargo test -p ms-expr --test lexer`.** Expected: FAIL (does not compile).
- [ ] **Step 4: Implement `lex`.** Hand-written char iterator. Line and column are 1-based and count chars; `\r\n` counts as one newline; a leading BOM is skipped.
- [ ] **Step 5: Run the tests again.** Expected: all PASS.
- [ ] **Step 6: Commit:** `git add -A && git commit -m "feat(ms-expr): workspace skeleton and lexer"`

---

### Task 2: Parser → AST

**Files:**
- Create: `crates/ms-expr/src/{ast,parser}.rs`
- Test: `crates/ms-expr/tests/parser.rs`

**Interfaces:**
- Consumes: `lex`, `Token`, `CompileError` (Task 1)
- Produces:
  - `pub enum Expr { Num(f64), Var(String), Assign { op: AssignOp, name: String, value: Box<Expr> }, Unary(UnOp, Box<Expr>), Binary(BinOp, Box<Expr>, Box<Expr>), Call { name: String, args: Vec<Expr>, line: u32, col: u32 }, Block(Vec<Expr>) }`
  - `pub fn parse(src: &str) -> Result<Expr, CompileError>` (always returns `Block`)

Precedence, low → high:
1. `=` and the compound assignments (right-associative)
2. `||`
3. `&&`
4. `|`
5. `&`
6. `== != < <= > >=`
7. `+ -`
8. `* / %`
9. `^` (right-associative)
10. unary `- + !`

A parenthesised group may hold a `;`-separated sequence, which parses as a nested `Block`. The program's statements are separated by `;`, and empty statements are allowed.

- [ ] **Step 1: Write failing tests:**
  - `precedence`: `"1+2*3^2"` gives `Binary(Add, 1, Binary(Mul, 2, Binary(Pow, 3, 2)))`.
  - `pow_right_assoc`: `"2^3^2"` gives `Pow(2, Pow(3, 2))`.
  - `assign_chain`: `"a=b=3"`.
  - `compound`: `"a+=1"`.
  - `call_args`: `"if(a,b,c)"` gives a `Call` with 3 args.
  - `paren_block`: `"(a=1; b=2)"` gives a nested `Block`.
  - `empty_statements_ok`: `";;a;;"` gives a `Block` of length 1.
  - `assign_to_non_var_errors`: `"1=2"` gives an error at 1:2.
  - `unclosed_paren_located`: `"sin(1"` gives an error at 1:6.
- [ ] **Step 2: Run `cargo test -p ms-expr --test parser`.** Expected: FAIL.
- [ ] **Step 3: Implement `parse`** as a Pratt or recursive-descent parser over the precedence table above.
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-expr): parser`

---

### Task 3: Compiler and scalar VM — arithmetic, variables, EEL quirks

**Files:**
- Create: `crates/ms-expr/src/{vars,bytecode,compile,vm,rng}.rs`
- Test: `crates/ms-expr/tests/semantics.rs`

**Interfaces:**
- Consumes: `parse` (Task 2)
- Produces:
  - **Variables:**
    - `pub struct VarId(pub u32)`
    - `pub struct VarTable`, with `fn new() -> Self`, `fn get_or_insert(&mut self, name: &str) -> VarId` (case-insensitive), `fn lookup(&self, name: &str) -> Option<VarId>`, `fn len(&self) -> usize` and `fn name(&self, id: VarId) -> &str`
  - **Compiling:** `pub struct Program` (`Clone + Send + Sync`) and `pub fn compile(src: &str, vars: &mut VarTable) -> Result<Program, CompileError>`
  - **Memory and randomness:**
    - `pub struct Megabuf` (1,048,576 × f64, lazily allocated in 64K-entry pages), with `fn get(&self, i: f64) -> f64` and `fn set(&mut self, i: f64, v: f64)`
    - `pub struct Rng`, with `fn seeded(seed: u64) -> Self` and `fn next_f64(&mut self) -> f64` (xoshiro256++ in [0, 1))
  - **Execution:**
    - `pub struct ExecEnv<'a> { pub vars: &'a mut [f64], pub megabuf: &'a mut Megabuf, pub gmegabuf: &'a mut Megabuf, pub rng: &'a mut Rng }`
    - `impl Program { pub fn run(&self, env: &mut ExecEnv) -> RunResult }`
    - `pub struct RunResult { pub value: f64, pub loop_cap_hit: bool }`
  - A `vars` slice shorter than `VarTable::len()` is a caller bug; `run` panics in debug builds.

Semantics (§5.1):
- **Division and modulo:** `/` and `%` by 0 give 0. `%` truncates both operands to `i64` first.
- **Equality:** `==` and `!=` use `|a-b| < 1e-5`.
- **Logical operators:** `&&` and `||` short-circuit and return 1 or 0.
- **Bitwise operators:** `&` and `|` truncate to `i64`.
- **`^`:** `powf`.
- **Comparisons:** return 1 or 0.
- **`!x`:** 1 if `x == 0` (fuzzy), else 0.
- **Variables:** reading an unknown variable registers it with initial value 0.

- [ ] **Step 1: Write failing tests** using a helper `eval(src) -> f64` that compiles, runs with fresh state and returns `value`:
  - `div_zero`: `eval("1/0") == 0.0` and `eval("5%0") == 0.0`.
  - `modulo_truncates`: `eval("7.9%3.2") == 1.0` and `eval("-7%3") == -1.0`.
  - `fuzzy_eq`: `eval("1==1.000001") == 1.0` and `eval("1==1.0001") == 0.0`.
  - `logic_short_circuit`: `eval("a=0; 0&&(a=5); a") == 0.0`.
  - `bitwise`: `eval("6&3") == 2.0` and `eval("6|3") == 7.0`.
  - `case_insensitive`: `eval("Foo=2; foo*FOO") == 4.0`.
  - `block_value_is_last`: `eval("1;2;3") == 3.0`.
  - `assign_returns_value`: `eval("a=(b=4)+1; a+b") == 9.0`.
  - `compound_ops`: `eval("a=10; a-=3; a*=2; a/=7; a%=3") == 2.0`.
  - `vars_persist_between_runs`: `"x=x+1"` run 3 times on the same `vars` gives 3.0.
- [ ] **Step 2: Run `cargo test -p ms-expr --test semantics`.** Expected: FAIL.
- [ ] **Step 3: Implement the compiler and VM.**
  - The AST is constant-folded, then lowered to register bytecode (`enum Op` in `bytecode.rs`).
  - The VM is a `match` loop over `Vec<Op>` using a scratch register file sized at compile time.
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-expr): compiler and scalar VM`

---

### Task 4: Functions, control flow, loop cap, megabuf

**Files:**
- Modify: `crates/ms-expr/src/{compile,vm}.rs`
- Test: `crates/ms-expr/tests/functions.rs`

**Interfaces:**
- Consumes: Task 3 types.
- Produces: no new types. `CompileError` covers unknown functions and wrong argument counts, located at the call.

Built-in functions (§5.1):
- **Trigonometry:** `sin cos tan asin acos atan`, `atan2(y,x)`.
- **Powers and logs:** `sqrt pow exp log log10`.
- **Rounding and sign:** `abs sign min max floor ceil`, `int` (= floor, as in the EEL2 docs).
- **Other maths:** `sqr(x) = x*x`, `sigmoid(x,c) = 1/(1+exp(-x*c))`.
- **Logic and comparison:** `band(a,b) = a&&b`, `bor(a,b) = a||b`, `bnot(x) = !x`, `equal(a,b) = a==b` (fuzzy), `above(a,b) = a>b`, `below(a,b) = a<b`.
- **`rand(x)`:** `next_f64() * max(x, 1)`.
- **Control:**
  - `if(c,a,b)` evaluates only the chosen branch;
  - `loop(n, body)` runs `floor(n)` times (none if ≤ 0);
  - `while(body)` repeats while the body's value is nonzero.
- **Memory:** `megabuf(i)` and `gmegabuf(i)` are readable and assignable, e.g. `megabuf(3) = 5`. Index `floor(i)`; out of range `[0, 1048576)` reads 0, and writes are ignored.
- **Loop cap:** the total iterations of all loops in one `run` may not exceed 1,048,576. When the cap is hit, every loop in that run exits immediately and `loop_cap_hit = true`.

- [ ] **Step 1: Write failing tests:**
  - `trig`: `eval("atan2(1,1)") ≈ PI/4`.
  - `int_floors`: `eval("int(-1.5)") == -2.0`.
  - `sigmoid`: `eval("sigmoid(0,5)") == 0.5`.
  - `if_lazy`: `eval("a=0; if(1, 2, a=9); a") == 0.0`.
  - `loop_counts`: `eval("n=0; loop(5, n+=1); n") == 5.0`.
  - `loop_negative`: `eval("n=0; loop(-3, n+=1); n") == 0.0`.
  - `while_runs`: `eval("i=0; while(i+=1; i<10); i") == 10.0`.
  - `loop_cap`: `"while(1)"` returns with `loop_cap_hit == true` in under 1 s.
  - `megabuf_rw`: `eval("megabuf(10)=3; megabuf(10.7)") == 3.0`.
  - `megabuf_oob`: `eval("megabuf(-1)=3; megabuf(2000000)=3; megabuf(-1)+megabuf(2000000)") == 0.0`.
  - `gmegabuf_shared`: two programs sharing one `gmegabuf` see each other's writes.
  - `rand_range`: 10k samples of `rand(10)` all in [0, 10), and `rand(0.2)` stays in [0, 1).
  - `rand_deterministic`: the same seed gives the same sequence.
  - `unknown_fn_located`: `"a=foo(1)"` gives an error at 1:3.
  - `arity_checked`: `"sin(1,2)"` gives an error.
- [ ] **Step 2: Run `cargo test -p ms-expr --test functions`.** Expected: FAIL.
- [ ] **Step 3: Implement.** `if`, `loop`, `while`, `&&` and `||` compile to jumps. The loop counter is a field on the VM run state, checked on every back-edge.
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-expr): builtins, control flow, loop cap, megabuf`

---

### Task 5: Batched VM and carried-variable analysis

**Files:**
- Create: `crates/ms-expr/src/batch.rs`, `crates/ms-expr/benches/vm.rs`
- Test: `crates/ms-expr/tests/batch.rs`

**Interfaces:**
- Consumes: `Program`, `ExecEnv` (Task 3)
- Produces:
  - `impl Program { pub fn reads_before_write(&self) -> &[VarId] }`: variables that some execution path reads before writing.
  - `pub struct Lanes { pub n: usize, pub cols: Vec<Vec<f64>> }` (one column per `VarId`, each of length `n`)
  - `impl Program { pub fn run_batched(&self, lanes: &mut Lanes, shared: &mut BatchShared) -> bool /*loop_cap_hit*/ }`
  - `pub struct BatchShared<'a> { pub megabuf: &'a mut Megabuf, pub gmegabuf: &'a mut Megabuf, pub rng: &'a mut Rng }`

Rules (§5.4):
- Each opcode runs over blocks of 256 lanes.
- Branches (`if`, `&&`, `||`) use masks.
- Loops run per lane with a lane mask. Each lane has its own loop-cap counter of 1,048,576, exactly as if the scalar VM ran that lane alone.
- Any program that calls `megabuf`, `gmegabuf` or `rand` is reported as non-batchable via `pub fn batchable(&self) -> bool`, because sequential order is observable.
- The engine (Task 21) decides whether to batch based on `reads_before_write` together with `batchable`.

- [ ] **Step 1: Write failing tests:**
  - `rbw_detects_carry`: `"t = t + x"` gives `[t]`, plus `x`.
  - `rbw_clean`: `"t = x*2; zoom = t"` gives `[x]` only.
  - `rbw_branch`: `"if(x, t=1, 0); y=t"` includes `t`.
  - `not_batchable_with_rand`: `"x=rand(1)"` gives `batchable() == false`.
  - `proptest_scalar_equals_batched`:
    - generate random ASTs (depth ≤ 6) over vars `x,y,a,b`, all ops, and `if`/`sin`/`min`/`pow`/`loop` with constant counts ≤ 4;
    - for 300 lanes of random inputs in [-2, 2], run the scalar VM per lane and the batched VM;
    - assert bitwise-equal f64 results for every var (`to_bits` equality, with NaN treated as equal to NaN).
- [ ] **Step 2: Run `cargo test -p ms-expr --test batch`.** Expected: FAIL.
- [ ] **Step 3: Implement `run_batched` and the analysis.** `reads_before_write` is a forward dataflow over bytecode basic blocks: the set of vars "definitely written" at each point, where a read of a var not in that set records it.
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Add the criterion bench** `vm/vertex_64x48` running a ~60-op vertex program over 3,072 lanes, scalar and batched. Run `cargo bench -p ms-expr`; it must complete and print both timings.
- [ ] **Step 6: Commit:** `feat(ms-expr): batched VM, carried-var analysis, bench`

---

### Task 6: Audio history, gap filling, waveform window

**Files:**
- Create: `crates/ms-analysis/src/{lib,history}.rs`
- Test: `crates/ms-analysis/tests/history.rs`

**Interfaces:**
- Produces:
  - `pub type Frame = [f32; 2];`
  - `pub struct AudioHistory`, with:
    - `fn new(sample_rate: u32, seconds: f64) -> Self` (capacity ≥ 1 s)
    - `fn sample_rate(&self) -> u32`
    - `fn push(&mut self, frames: &[Frame], capture_time: f64)` (time of the *last* frame, in seconds on the app clock)
    - `fn fill_silence_until(&mut self, now: f64)` (pads zeros if `now - last_time > 0.050`)
    - `fn window(&self, end_time: f64, out: &mut [Frame])` (frames ending at `end_time`, zeros where there's no data)
    - `fn silent_for(&self, now: f64) -> f64` (seconds since the last non-silent frame, |s| > 1e-4)
    - `fn reset(&mut self, sample_rate: u32)`

- [ ] **Step 1: Write failing tests:**
  - `window_reads_latest`: push 1,000 ramp frames ending at t=1.0 at 48 kHz; `window(1.0, out[100])` equals the last 100.
  - `window_with_offset`: `window(1.0 - 0.001, ..)` ends 48 frames earlier.
  - `gap_filled`: push at t=1.0, `fill_silence_until(1.2)`, then `window(1.2, ..)` is all zero.
  - `no_fill_under_50ms`: `fill_silence_until(1.04)` adds nothing.
  - `silent_for`: tracks seconds since the last loud frame.
  - `wraps`: pushing 3 s into a 1 s history keeps only the latest second.
  - `rate_change_resets`: after `reset(44100)`, `sample_rate() == 44100` and the window is all zero.
- [ ] **Step 2: Run `cargo test -p ms-analysis --test history`.** Expected: FAIL.
- [ ] **Step 3: Implement** as a ring buffer of `Frame`, tracking the time of the newest frame.
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-analysis): audio history`

---

### Task 7: Spectrum, bands and smoothing

**Files:**
- Create: `crates/ms-analysis/src/{spectrum,bands}.rs`
- Test: `crates/ms-analysis/tests/bands.rs`

**Interfaces:**
- Consumes: `AudioHistory` (Task 6)
- Produces:
  - `pub struct AudioFrame { pub wave: [[f32; 576]; 2], pub spectrum: [f32; 512], pub bass: f32, pub mid: f32, pub treb: f32, pub bass_att: f32, pub mid_att: f32, pub treb_att: f32, pub beat: f32, pub beat_strength: f32, pub silent_for: f32 }` (`beat` fields are filled in Task 8; zero here)
  - `pub struct Analyzer`, with `fn new(sample_rate: u32) -> Self` and `fn tick(&mut self, hist: &AudioHistory, now: f64, av_offset: f64, dt: f64) -> &AudioFrame`

Constants (§8.3):
- **FFT:** 2048-sample Hann window on mono `(l+r)/2`, giving 512 bins of magnitude from the first 512 rfft bins.
- **Display scaling:** `spectrum[i] = ln(1 + mag*0.5)`, a fixed formula used for display.
- **Band edges:** bass [20, 250), mid [250, 4000), treb [4000, 16000) Hz. An edge above Nyquist is clamped to Nyquist. A band that becomes empty reads 1.0.
- **Long-term average:** an exponential average with time constant 5 s, floored at 1e-6.
- **Instant values:** `band = energy / long_avg`, smoothed with τ = 0.05 s.
- **`_att` values:** τ = 0.4 s.
- **Smoothing formula:** `y += (x - y) * (1 - exp(-dt/τ))`.
- **Initial state:** the long-term average is initialised from the first non-silent tick, so the first loud second reads ≈ 1.
- **Waveform:** 576 frames ending at `now - av_offset`.

- [ ] **Step 1: Write failing tests**, with helper `run(signal, seconds, rate)` feeding 60 Hz ticks:
  - `sine_60hz_is_bass`: after 3 s, `bass > 0.5` and `mid < 0.1*bass` and `treb < 0.1*bass` (energy check against the raw band energies, exposed through `#[cfg(test)] fn raw_energy()`).
  - `sine_1k_is_mid` and `sine_8k_is_treb`: the same pattern.
  - `steady_signal_normalises_to_one`: pink noise for 10 s gives `bass`, `mid` and `treb` within 0.8..1.25.
  - `loud_hit_above_one`: the noise level doubles at t=10 s, so `bass > 1.5` within 0.2 s.
  - `att_smoother_than_instant`: on a 2 Hz square wave, the variance of `bass_att` is below the variance of `bass`.
  - `silence_no_nan`: 5 s of zeros leaves every field finite.
  - `rates_consistent`: pink noise at 22050, 44100, 48000 and 192000 Hz gives all three bands within ±15% of the 48 kHz result after 10 s (22.05 kHz treb: band clamped to Nyquist, still finite and > 0).
- [ ] **Step 2: Run `cargo test -p ms-analysis --test bands`.** Expected: FAIL.
- [ ] **Step 3: Implement** with `realfft`; plan the FFT once in `new`.
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-analysis): spectrum and bands`

---

### Task 8: Beat detection

**Files:**
- Create: `crates/ms-analysis/src/beat.rs`
- Modify: `crates/ms-analysis/src/bands.rs` (call it from `tick`)
- Test: `crates/ms-analysis/tests/beat.rs`

**Interfaces:**
- Produces: `AudioFrame.beat` and `AudioFrame.beat_strength` are now filled.

Algorithm (§8.3):
- **Flux:** `flux = max(0, bass_energy - prev_bass_energy)`, computed on each tick.
- **Threshold:** the mean of the last 60 flux values plus `k*std`, with `k = 1.5`.
- **Onset:** `flux > threshold && flux > 1e-6 && since_last >= 0.150 s`.
- **On an onset:**
  - `beat = 1.0` for that tick only;
  - `beat_strength = flux / threshold`.
- **Between onsets:** `beat_strength` decays linearly to 0 over 0.2 s.

- [ ] **Step 1: Write failing tests:**
  - `click_track_120bpm`: 50 Hz kick bursts of 30 ms every 0.5 s over 10 s. Each onset is within ±1 tick of the kick times, and there are 19–21 beats in total.
  - `refractory`: kicks every 0.1 s give at most one beat per 0.15 s.
  - `steady_tone_no_beats`: a constant 60 Hz sine gives 0 beats after a 1 s warm-up.
  - `strength_decays`: 0.2 s after a beat, `beat_strength == 0`.
- [ ] **Step 2: Run `cargo test -p ms-analysis --test beat`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-analysis): beat detection`

---

### Task 9: Downmix, sample ring and backoff (pure parts of `ms-audio`)

**Files:**
- Create: `crates/ms-audio/src/{lib,downmix,ring,backoff}.rs`
- Test: `crates/ms-audio/tests/pure.rs`

**Interfaces:**
- Consumes: `ms_analysis::{Frame, AudioHistory}`
- Produces:
  - `pub fn layout_for(channels: u16) -> Vec<Channel>` where `pub enum Channel { FL, FR, FC, LFE, BL, BR, SL, SR, Other }`:
    - 1 → `[FC]`
    - 2 → `[FL, FR]`
    - 6 → `[FL, FR, FC, LFE, SL, SR]`
    - 8 → `[FL, FR, FC, LFE, BL, BR, SL, SR]`
    - anything else → `[FL, FR, Other…]`
  - `pub fn downmix(interleaved: &[f32], layout: &[Channel], out: &mut [Frame])`:
    - FC and LFE are added to both channels × `0.70710677`;
    - the left surrounds (BL, SL) are added to left × `0.70710677`, and the right surrounds (BR, SR) to right;
    - `Other` channels are ignored;
    - mono FC goes to both channels at ×1.0.
  - `pub fn ring(capacity_frames: usize) -> (RingProducer, RingConsumer)` (rtrb of `Frame` plus rtrb of `Stamp { time: f64, frames: u32 }`)
  - `RingProducer::push(&mut self, frames: &[Frame], time: f64)`, which drops excess frames when full and never blocks
  - `RingConsumer::drain_into(&mut self, hist: &mut AudioHistory)`
  - `pub fn backoff_delay(attempt: u32) -> Duration`: 0.5 s, 1 s, 2 s, 5 s, then 5 s from then on

- [ ] **Step 1: Write failing tests:**
  - `stereo_passthrough`
  - `mono_to_both`
  - `five_one`: unit impulses in each channel land as specified.
  - `seven_one`
  - `three_channels_third_ignored`
  - `drain_preserves_order_and_time`: push 3 packets, drain, then the history window equals the concatenation, and the history's last time equals the last stamp.
  - `full_ring_drops_not_blocks`
  - `backoff_sequence`: attempts 0..6 give `[0.5, 1, 2, 5, 5, 5]` s.
- [ ] **Step 2: Run `cargo test -p ms-audio --test pure`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-audio): downmix, ring, backoff`

---

### Task 10: WASAPI loopback capture and device notifications

**Files:**
- Create: `crates/ms-audio/src/win/{mod,capture,notify,bluetooth}.rs`, `crates/ms-audio/examples/meter.rs`
- Test: `crates/ms-audio/tests/capture_smoke.rs` (marked `#[ignore]`; run manually)

**Interfaces:**
- Consumes: Task 9.
- Produces:
  - `pub enum DeviceChoice { Default, Named(String) }`
  - `pub enum AudioEvent { Started { name: String, sample_rate: u32, channels: u16, bluetooth: bool }, Stopped { reason: String }, DefaultDeviceChanged }`
  - `pub struct AudioService`, with:
    - `fn start(choice: DeviceChoice, clock: Arc<dyn Fn() -> f64 + Send + Sync>) -> (Self, RingConsumer, Receiver<AudioEvent>)`
    - `fn list_output_devices() -> Vec<String>`
  - `AudioService` restarts its stream itself:
    - on `DefaultDeviceChanged` while the choice is `Default`;
    - on stream error or device removal, retrying with `backoff_delay`.
  - Each restart that changes the sample rate is reported through `Started`. The render thread then calls `hist.reset(rate)`.

Implementation notes:
- **Loopback:** cpal opens WASAPI loopback when an *input* stream is built on an *output* device.
- **Callback:** downmixes into a preallocated scratch buffer of `Frame`, then calls `RingProducer::push` with `clock()`.
- **Device-change notifications:** `notify.rs` implements `IMMNotificationClient` (crate `windows`, feature `Win32_Media_Audio`) and forwards `OnDefaultDeviceChanged` for `eRender`/`eConsole` over a channel.
- **Bluetooth detection:** `bluetooth.rs` reads the endpoint's `PKEY_Device_EnumeratorName`. A value of `BTHENUM` or `BTHLEDEVICE` means Bluetooth.
- **Service thread:** owns the cpal `Stream` (not `Send`) and handles restarts.

- [ ] **Step 1: Write the example `meter`.** It prints rolling RMS and device events once a second. Run `cargo run -p ms-audio --example meter` with music playing; RMS must be > 0, and it must be ≈ 0 when paused.
- [ ] **Step 2: Manual check.** Switch the default output device in Windows while `meter` runs; within 6 s it prints `Started` for the new device and RMS resumes.
- [ ] **Step 3: Write the ignored smoke test** `capture_starts`: `AudioService::start(Default)` yields `Started` within 3 s. Run `cargo test -p ms-audio -- --ignored`. Expected: PASS on a machine with an output device.
- [ ] **Step 4: Commit:** `feat(ms-audio): WASAPI loopback service with device-change restart`

---

### Task 11: `.splat` format

**Files:**
- Create: `crates/ms-preset/src/{lib,format}.rs`, `crates/ms-preset/tests/fixtures/*.splat`
- Test: `crates/ms-preset/tests/format.rs`

**Interfaces:**
- Produces:
  - `pub struct Preset { pub version: u32, pub meta: Meta, pub params: Params, pub code: Code, pub waves: Vec<WaveDef>, pub shapes: Vec<ShapeDef>, pub shader: ShaderDefs }`
  - `pub struct Meta { pub name: String, pub author: String, pub license: String }`
  - `pub struct Params { pub feedback_format: FeedbackFormat /* Rgba8 | Fp16 */, pub mesh: [u32; 2], pub sim_rate: f64, pub values: BTreeMap<String, f64> }`. Keys of `values` are lowercased writable variable names (§5.3).
  - `pub struct Code { pub init: String, pub frame: String, pub vertex: String }`
  - `pub struct WaveDef { pub enabled: bool, pub samples: u32, pub params: BTreeMap<String,f64>, pub init: String, pub frame: String, pub point: String }`
  - `pub struct ShapeDef { pub enabled: bool, pub sides: u32, pub instances: u32, pub params: BTreeMap<String,f64>, pub init: String, pub frame: String }`
  - `pub struct ShaderDefs { pub warp: Option<String>, pub comp: Option<String> }`
  - `pub fn parse(bytes: &[u8]) -> Result<(Preset, Vec<Warning>), PresetError>`
  - `pub struct Warning { pub message: String }`
  - `pub struct PresetError { pub line: u32, pub col: u32, pub message: String }`
  - `impl Preset { pub fn to_toml(&self) -> String }`

Defaults and limits:
- **Defaults:** `version` 1, `rgba8`, mesh `[64, 48]`, `sim_rate` 60, `samples` 512, `sides` 4, `instances` 1.
- **Writable variable defaults:** `pub const PARAM_DEFAULTS: &[(&str, f64)]` in `format.rs` lists every writable variable in §5.3 with its MilkDrop default. For example: `zoom=1, rot=0, warp=1, decay=0.98, cx=0.5, cy=0.5, sx=1, sy=1, zoomexp=1, wave_a=0.8, wave_r=wave_g=wave_b=1, gamma=1, echo_zoom=1, echo_alpha=0, b1x=b2x=b3x=1, b1ed=0.25`, everything else 0.
- **Clamps:**
  - more than 4 waves or 4 shapes: extras are dropped with a warning;
  - `mesh` is clamped to [8, 192] × [6, 144];
  - `samples` is clamped to 1..=512, `sides` to 3..=100 and `instances` to 1..=1024.
- **Text handling:** a BOM is stripped and CRLF accepted. Invalid UTF-8 is an error located at the first bad byte's line and column.
- **Unknown keys:** each one produces a `Warning`.
- **Version:** a `version` above 1 is an error: "made by a newer Milksplat".

- [ ] **Step 1: Write failing tests:**
  - `minimal_file_defaults`: `"version = 1"` parses, and `params.mesh == [64,48]`.
  - `full_fixture_roundtrip`: `parse(to_toml(parse(f)))` equals `parse(f)`.
  - `unknown_key_warns`
  - `extra_waves_dropped_with_warning`
  - `clamps`
  - `bom_crlf_tabs_ok`
  - `invalid_utf8_located`
  - `toml_syntax_error_located`: line and column match the toml error span.
  - `future_version_rejected`
- [ ] **Step 2: Run `cargo test -p ms-preset --test format`.** Expected: FAIL.
- [ ] **Step 3: Implement** with `serde` and raw `toml::Table` walking, so unknown keys can be collected.
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-preset): .splat format`

---

### Task 12: Compiled presets, library, selection

**Files:**
- Create: `crates/ms-preset/src/{compile,library,select}.rs`
- Test: `crates/ms-preset/tests/{compile,select}.rs`

**Interfaces:**
- Consumes: `Preset` (Task 11), `ms_expr::{compile, VarTable, Program}`
- Produces:
  - `pub const BUILTIN_VARS: &[&str]`: every read-only and writable name in §5.3 plus `q1..q32`, `t1..t8` and `reg00..reg99`. Registered first in this order, so their `VarId`s are stable.
  - `pub struct CompiledPreset { pub source: Preset, pub vars: VarTable, pub init: Option<Program>, pub frame: Option<Program>, pub vertex: Option<Program>, pub waves: Vec<CompiledWave>, pub shapes: Vec<CompiledShape>, pub errors: Vec<CodeError> }`
  - `pub struct CodeError { pub block: String /* "frame", "wave[1].point" … */, pub error: ms_expr::CompileError }`
  - `CompiledWave` and `CompiledShape` each have their own `vars: VarTable` (also seeded with `BUILTIN_VARS`) and `Option<Program>` for each block.
  - `pub fn compile_preset(p: Preset) -> CompiledPreset`. A block with an error becomes `None` and is recorded; the preset still compiles.
  - **Library:**
    - `pub struct Library { pub entries: Vec<Entry> }`
    - `pub struct Entry { pub id: String /* path relative to its root, '/'-separated, lowercased */, pub path: PathBuf }`
    - `Library::scan(roots: &[PathBuf]) -> Library` (recursive, `*.splat` only, sorted by id)
  - **Preferences:** `pub struct Prefs { pub favourites: HashSet<String>, pub bans: HashSet<String>, pub blocklist: HashSet<String> }`
  - **Selector:**
    - `pub struct Selector`, with `fn new(seed: u64) -> Self`
    - `fn next(&mut self, lib: &Library, prefs: &Prefs) -> Option<String>`
    - `fn prev(&mut self) -> Option<String>`
    - `fn push_manual(&mut self, id: String)`
  - `pub fn next_duration(base_s: f64, rng: &mut ms_expr::Rng) -> f64` (`base * (0.75 + 0.5*u)`)

Selection rules (§7.3):
- Candidates are presets that are neither banned nor blocklisted, and not among the most recent `ceil(0.3 × candidates)` history entries.
- If that leaves none, the recency rule is dropped.
- Favourites have weight 2, others weight 1.
- The history keeps 50 entries.
- `prev` walks back through the history.

- [ ] **Step 1: Write failing tests:**
  - **`compile.rs`:**
    - `builtin_ids_stable`: `vars.lookup("zoom")` is identical across two presets.
    - `bad_block_disabled_others_kept`: frame has a syntax error and vertex is fine, so `frame.is_none()`, `vertex.is_some()` and `errors.len() == 1` with `block == "frame"`.
  - **`select.rs`:**
    - `empty_library_none`
    - `single_preset_repeats`
    - `all_banned_none`
    - `no_recent_repeat`: in a 10-preset library, no preset repeats within 3 consecutive picks over 1,000 picks.
    - `favourites_twice_as_often`: over 20k picks with 1 favourite among 10, its share is 2/11 ± 0.02.
    - `prev_walks_history`
    - `history_capped_50`
    - `duration_range`: 10k samples of `next_duration(20, ..)` all lie in [15, 25].
- [ ] **Step 2: Run `cargo test -p ms-preset`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-preset): compile, library, selector`

---

### Task 13: GPU setup, readback and golden-image harness

**Files:**
- Create: `crates/ms-render/src/{lib,gpu,golden}.rs`
- Test: `crates/ms-render/tests/gpu.rs`

**Interfaces:**
- Produces:
  - `pub struct Gpu { pub instance: wgpu::Instance, pub adapter: wgpu::Adapter, pub device: wgpu::Device, pub queue: wgpu::Queue, pub lost: Arc<AtomicBool> }`
  - `Gpu::new(compatible_surface: Option<&wgpu::Surface>) -> Result<Gpu, GpuError>`:
    - backends DX12 then Vulkan;
    - `PowerPreference::HighPerformance`;
    - requests `TIMESTAMP_QUERY` if available;
    - sets `device.on_uncaptured_error` to log via `tracing::error!`, never panicking;
    - sets the device-lost callback to `lost = true`.
  - `Gpu::headless() -> Result<Gpu, GpuError>`: a real adapter, or WARP (`force_fallback_adapter: true` on DX12) if none.
  - `pub fn read_texture(gpu: &Gpu, tex: &wgpu::Texture) -> image::RgbaImage` (handles row padding; rgba16float is converted)
  - `pub fn assert_golden(actual: &image::RgbaImage, name: &str)`:
    - compares against `crates/<crate>/tests/golden/<name>.png`;
    - passes if ≤ 0.5% of pixels differ by more than 8 in any channel;
    - `MS_BLESS=1` writes the reference instead;
    - on failure, writes `<name>.actual.png` next to it.

- [ ] **Step 1: Write failing tests:**
  - `headless_device`: `Gpu::headless()` is `Ok`.
  - `clear_and_read`: clear a 64×64 rgba8 texture to (255, 0, 0, 255); every pixel reads back red.
  - `odd_width_readback`: a 37×5 texture reads back correctly (row padding).
  - `uncaptured_error_does_not_panic`: create a texture with an invalid descriptor inside no error scope; the test completes.
  - `golden_roundtrip`: bless, then compare the same image, which passes; change 1% of pixels and it fails.
- [ ] **Step 2: Run `cargo test -p ms-render --test gpu`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-render): gpu setup and golden harness`

---

### Task 14: Feedback chain and built-in warp pass

**Files:**
- Create: `crates/ms-render/src/{feedback,warp}.rs`, `crates/ms-render/src/shaders/warp_builtin.wgsl`
- Test: `crates/ms-render/tests/warp.rs`

**Interfaces:**
- Consumes: `Gpu` (Task 13)
- Produces:
  - `pub struct RenderConfig { pub internal_size: [u32; 2], pub format: FeedbackFormat }` (re-exported enum, `Rgba8 | Fp16`)
  - `pub struct Renderer`, with:
    - `fn new(gpu: &Gpu, cfg: RenderConfig) -> Self`
    - `fn resize(&mut self, gpu: &Gpu, internal_size: [u32; 2])` (clamped to `[16, adapter.limits().max_texture_dimension_2d]`, contents kept by a bilinear copy)
    - `fn set_format(&mut self, gpu: &Gpu, f: FeedbackFormat)`
    - `fn feedback_texture(&self) -> &wgpu::Texture` (the latest buffer)
  - `pub struct MeshData { pub cols: u32, pub rows: u32, pub uv: Vec<[f32; 2]>, pub color: Vec<[f32; 4]> }` (`(cols+1)*(rows+1)` vertices, row-major, x left→right, y top→bottom)
  - `pub struct WarpLayer<'a> { pub mesh: &'a MeshData, pub weight: &'a [f32] /* per vertex */, pub shader: &'a WarpShader, pub uniforms: &'a Uniforms }`. `WarpShader` and `Uniforms` are defined in Task 15; until then use the builtin only.
  - `Renderer::warp(&mut self, gpu: &Gpu, layers: &[WarpLayer])`:
    - swaps the ping-pong buffers;
    - clears the target to 0;
    - draws each layer with additive blending, with vertex alpha = `weight`;
    - the builtin computes `out = sample(prev, uv).rgb * decay`, where `decay` comes from `uniforms`;
    - in `Fp16`, the fragment output is `clamp(select(v, 0, isnan(v) || isinf(v)), 0, 64)`.

- [ ] **Step 1: Write failing tests:**
  - `identity_warp_decays`: seed the buffer with mid-grey 128; one warp with identity mesh and `decay = 0.5` gives ≈ 64 ± 1 everywhere.
  - `zoom_mesh_golden`: seed a test pattern; 30 warps with UVs scaled 0.98 about the centre; `assert_golden("warp_zoom")`.
  - `two_layers_weighted`: layer A writes decay 1 (from white), layer B decay 0, weights 0.25/0.75 gives ≈ 64.
  - `fp16_nan_sanitised`: seed with NaN via a write-texture of `f16::NAN`; after one warp, no NaN in the readback.
  - `resize_clamps`: a request for `[1, 1]` gives `[16, 16]`; a request for `[100000, 10]` gives `[max_dim, 16]`.
- [ ] **Step 2: Run `cargo test -p ms-render --test warp`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests again.** Expected: PASS (bless the golden once with `MS_BLESS=1`, inspect the PNG by eye, then run normally).
- [ ] **Step 5: Commit:** `feat(ms-render): feedback chain and warp pass`

---

### Task 15: Shader prelude, uniforms, user shaders, composite

**Files:**
- Create: `crates/ms-render/src/{shaders,composite}.rs`, `crates/ms-render/src/shaders/{prelude.wgsl,comp_builtin.wgsl}`
- Test: `crates/ms-render/tests/shaders.rs`

**Interfaces:**
- Consumes: Tasks 13–14.
- Produces:
  - `#[repr(C)] #[derive(Pod, Zeroable)] pub struct Uniforms`, with these fields in order, std140-padded:
    - `time, fps, frame, progress: f32`
    - `bass, mid, treb, beat: f32`
    - `bass_att, mid_att, treb_att, beat_strength: f32`
    - `decay, gamma, echo_zoom, echo_alpha: f32`
    - `echo_orient, invert, solarize, brighten: f32`
    - `darken, darken_center, aspect_x, aspect_y: f32`
    - `texsize: [f32; 4]`
    - `rand_frame: [f32; 4]`, `rand_preset: [f32; 4]`
    - `roam_cos: [f32; 4]`, `roam_sin: [f32; 4]`
    - `blur_min: [f32; 4]`, `blur_max: [f32; 4]` (xyz used)
    - `q: [[f32; 4]; 8]`
  - **Shader compilation:**
    - `pub struct ShaderSet { pub warp: WarpShader, pub comp: CompShader, pub errors: Vec<String>, pub blur_levels: u8, pub user_textures: Vec<String> }`
    - `Renderer::compile_shaders(&self, gpu: &Gpu, warp: Option<&str>, comp: Option<&str>, tex: &dyn Fn(&str) -> Option<image::RgbaImage>) -> ShaderSet`. This is safe to call from any thread.
  - `Renderer::composite(&mut self, gpu: &Gpu, target: &wgpu::TextureView, target_format: wgpu::TextureFormat, layers: &[CompositeLayer])`, where `pub struct CompositeLayer<'a> { pub shader: &'a CompShader, pub uniforms: &'a Uniforms, pub alpha: f32 }`

**Prelude contract (§6.5):**
- **Group 0:**
  - binding 0 `u: Uniforms`
  - 1 `sampler_main`
  - 2–4 `sampler_blur1..3`
  - 5–7 `sampler_noise_lq/mq/hq`
  - 8–9 `sampler_noisevol_lq/hq` (texture_3d)
  - 10 `sampler_spectrum`
  - samplers 11–14 `s_fw, s_fc, s_pw, s_pc`
- **Group 1:** user textures. For a name `X`, `sampler_X: texture_2d<f32>` is at binding `2i` and `sampler_X_s: sampler` at `2i+1`, for i = 0..16.
- **Name prefixes:** the prefixes `fw_`, `fc_`, `pw_` and `pc_` on `X` choose wrap/clamp and linear/point filtering, and are stripped to find the file.
- **Helper functions:** `GetMain(uv)`, `GetPixel(uv)` (point-sampled main), and `GetBlur1/2/3(uv)`. The blur helpers return `blur_min + tex * (blur_max - blur_min)`.
- **User code:**
  - a user warp shader is the *body* of `fn user_warp(uv: vec2f, uv_orig: vec2f, rad: f32, ang: f32, hue_shader: vec3f) -> vec3f`;
  - a user comp shader is the body of `fn user_comp(uv: vec2f, rad: f32, ang: f32, hue_shader: vec3f) -> vec3f`.
- **Detection:**
  - user textures are found by scanning the source for `sampler_[a-z0-9_]+` that are not prelude names;
  - `blur_levels` is the highest `GetBlurN` or `sampler_blurN` referenced.
- **Failure:** naga parse/validate errors and any error-scope error from pipeline creation make that stage fall back to the builtin; the message goes into `errors`.
- **Built-in composite**, applied in this order:
  1. video echo (`echo_alpha`, `echo_zoom`, `echo_orient` 0–3 flips);
  2. `gamma`;
  3. `brighten`, `darken`, `solarize`;
  4. `invert`;
  5. `darken_center`.

  The formulas follow MilkDrop's fixed-function behaviour and are documented in a comment in `comp_builtin.wgsl`.

- [ ] **Step 1: Write failing tests:**
  - `uniforms_layout`: `size_of::<Uniforms>()` is a multiple of 16, and the offset of `q` equals the prelude's offset. This is checked by compiling the prelude with naga and comparing the struct member offsets reported by naga's layouter.
  - `builtin_comp_identity_golden`: seed the feedback with the test pattern and default uniforms; composite to rgba8; `assert_golden("comp_identity")`.
  - `user_comp_applies`: body `return vec3f(1.0, 0.0, 0.0);` gives a red output.
  - `syntax_error_falls_back`: body `return vec3f(;` gives a non-empty `errors` and the output equals the builtin golden.
  - `validation_error_falls_back`: body `return 1.0;` (wrong type) does the same.
  - `user_texture_bound`: a comp shader sampling `sampler_fc_checker` with a supplied 2×2 checker image produces the checker; `user_textures == ["fc_checker"]`.
  - `blur_levels_detected`: a source using `GetBlur2` gives `blur_levels == 2`.
  - `compile_off_thread`: `compile_shaders` called from `std::thread::spawn` succeeds.
- [ ] **Step 2: Run `cargo test -p ms-render --test shaders`.** Expected: FAIL.
- [ ] **Step 3: Implement.** User warp shaders plug into the warp pass of Task 14: the `WarpShader` pipeline replaces the builtin per layer.
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-render): shader prelude, user shaders, composite`

---

### Task 16: Blur chain, noise, user and random textures, spectrum texture

**Files:**
- Create: `crates/ms-render/src/{blur,noise,textures}.rs`
- Test: `crates/ms-render/tests/textures.rs`

**Interfaces:**
- Consumes: Tasks 13–15.
- Produces:
  - `Renderer::blur(&mut self, gpu: &Gpu, levels: u8)`. Level n is at half the resolution of level n−1, made by a separable 9-tap Gaussian, and only levels 1..=`levels` are computed.
  - `pub fn noise_textures(gpu: &Gpu) -> NoiseSet`, generated once from seed `0x4D53_504C`:
    - 2D `lq` 32², `mq` 64² and `hq` 256² (rgba8);
    - 3D `noisevol_lq` 32³ and `noisevol_hq` 64³.
  - `pub struct TextureResolver { roots: Vec<PathBuf> }` with `fn resolve(&self, name_without_prefix: &str) -> Option<RgbaImage>`:
    - tries `.png`, then `.jpg`, in each root in order (§4);
    - `rand00`–`rand15` pick a random file from `<root>/random/` using a per-preset `Rng`.
  - `Renderer::set_spectrum(&mut self, gpu: &Gpu, spectrum: &[f32; 512])` uploads a 512×1 r32float texture.

- [ ] **Step 1: Write failing tests:**
  - `blur_levels_sizes`: internal 512² gives blur textures 256², 128² and 64².
  - `blur_only_requested`: `blur(1)` leaves levels 2–3 untouched (a sentinel colour stays).
  - `blur_golden`: a white dot blurred at level 3 gives `assert_golden("blur3")`.
  - `noise_deterministic`: two calls give identical bytes.
  - `resolver_order`: the same name in the preset folder and in bundled textures resolves to the preset folder.
  - `resolver_missing_none`
  - `rand_texture_from_folder`: `rand03` with 3 files in `random/` resolves to one of them.
  - `spectrum_uploaded`: a shader reading `sampler_spectrum` at x = 10/512 returns the uploaded value.
- [ ] **Step 2: Run `cargo test -p ms-render --test textures`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-render): blur, noise, textures, spectrum`

---

### Task 17: Waves, shapes, borders, motion vectors

**Files:**
- Create: `crates/ms-render/src/{waves,shapes,borders,motion}.rs` + matching `.wgsl`
- Test: `crates/ms-render/tests/draw.rs`

**Interfaces:**
- Consumes: Tasks 13–16.
- Produces:
  - `pub struct DrawList { pub motion: Option<MotionVectors>, pub shapes: Vec<ShapeDraw>, pub waves: Vec<WaveDraw>, pub main_wave: Option<MainWave>, pub borders: Borders }`
  - `pub struct WaveDraw { pub points: Vec<[f32; 2]>, pub colors: Vec<[f32; 4]>, pub dots: bool, pub thick: bool, pub additive: bool }`
  - `pub struct MainWave { pub mode: u8 /*0..=7*/, pub wave: [[f32; 576]; 2], pub spectrum: [f32; 512], pub color: [f32; 4], pub x: f32, pub y: f32, pub mystery: f32, pub dots: bool, pub thick: bool, pub additive: bool, pub brighten: bool }`
  - `pub struct ShapeDraw { pub x: f32, pub y: f32, pub rad: f32, pub ang: f32, pub sides: u32, pub inner: [f32; 4], pub outer: [f32; 4], pub border: [f32; 4], pub additive: bool, pub thick_outline: bool, pub textured: bool, pub tex_zoom: f32, pub tex_ang: f32 }` (one per instance)
  - `pub struct Borders { pub outer: (f32 /*size*/, [f32; 4]), pub inner: (f32, [f32; 4]) }`
  - `pub struct MotionVectors { pub nx: f32, pub ny: f32, pub dx: f32, pub dy: f32, pub len: f32, pub color: [f32; 4] }`
  - `Renderer::draw(&mut self, gpu: &Gpu, list: &DrawList, alpha: f32)` draws into the current feedback buffer in §6.2 order, with every colour's alpha × `alpha`.
  - `pub fn main_wave_points(w: &MainWave, aspect: [f32; 2]) -> Vec<[f32; 2]>`: the CPU geometry for the 8 modes, following MilkDrop's documented mode shapes:
    - 0: circle
    - 1: X-Y oscilloscope
    - 2: centred line, left channel
    - 3: line with volume-driven alpha
    - 4: horizontal derivative line
    - 5: X-Y "explosive"
    - 6: angled line
    - 7: dual lines

    Each mode is a separate function with a doc comment.
- **Coordinates:** all are in [0, 1] with the origin at the bottom-left, as in MilkDrop. The y flip happens in the vertex shader.

- [ ] **Step 1: Write failing tests:**
  - `main_wave_modes_in_bounds`: for each mode, with a sine wave input, every point is finite and within [−0.5, 1.5].
  - `mode0_is_closed`: the first point equals the last.
  - `wave_golden`: mode 2, a sine, white, thick gives `assert_golden("wave_mode2")`.
  - `shape_hexagon_golden`: 6 sides, gradient red → blue, border white gives `assert_golden("shape_hex")`.
  - `shape_textured`: a textured shape samples the feedback; with the feedback pre-filled green, the shape's centre is green.
  - `instances_1024`: 1,024 tiny shapes draw without error.
  - `borders_golden`: `assert_golden("borders")`.
  - `alpha_scales`: the same draw at `alpha = 0.5` has half the brightness of `1.0` (±2).
  - `order_shapes_before_waves`: an opaque shape and an opaque wave at the same spot; the wave's colour wins.
- [ ] **Step 2: Run `cargo test -p ms-render --test draw`.** Expected: FAIL.
- [ ] **Step 3: Implement.** Lines are drawn as quads; thick lines are 2 px wide at 1080p-equivalent scale; dots are 2×2 px quads.
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-render): waves, shapes, borders, motion vectors`

---

### Task 18: Text overlay and device-loss recovery

**Files:**
- Create: `crates/ms-render/src/text.rs`
- Modify: `crates/ms-render/src/lib.rs` (`recreate`)
- Test: `crates/ms-render/tests/{text,recovery}.rs`

**Interfaces:**
- Consumes: Tasks 13–17.
- Produces:
  - `pub struct TextItem { pub text: String, pub pos: [f32; 2] /* pixels, top-left */, pub scale: u32, pub color: [f32; 4] }`
  - `Renderer::draw_text(&mut self, gpu: &Gpu, target: &wgpu::TextureView, target_size: [u32; 2], items: &[TextItem])`: `font8x8` glyphs (basic Latin), with unknown chars drawn as `?` and a 1-px black shadow.
  - `Renderer::recreate(&mut self, gpu: &Gpu)` rebuilds every GPU resource on a new `Gpu`. The feedback contents are lost and cleared to black.
  - `ShaderSet` keeps its WGSL sources so `Renderer::recompile(&self, gpu, &ShaderSet) -> ShaderSet` can rebuild it.

- [ ] **Step 1: Write failing tests:**
  - `text_golden`: `"Ink Bloom — 60 fps"` at scale 2 gives `assert_golden("text")`. The em dash renders as `?`.
  - `recreate_on_new_device`: render the warp-zoom sequence for 10 ticks, `recreate` on a fresh `Gpu::headless()`, render 10 more ticks; no error, and the output is not all black (seeded again by drawing a shape).
  - `zero_size_target_skipped`: `draw_text` with `target_size [0, 0]` returns without error.
- [ ] **Step 2: Run `cargo test -p ms-render --test text --test recovery`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-render): text overlay, device recreation`

---

### Task 19: Fixed-tick clock and quality governor

**Files:**
- Create: `crates/ms-engine/src/{lib,clock,governor}.rs`
- Test: `crates/ms-engine/tests/{clock,governor}.rs`

**Interfaces:**
- Produces:
  - `pub struct FixedClock` with:
    - `fn new(rate_hz: f64) -> Self`
    - `fn set_rate(&mut self, hz: f64)`
    - `fn advance(&mut self, dt: f64) -> Ticks`
    - `fn alpha(&self) -> f64` (fraction into the next tick)
  - `pub struct Ticks { pub run: u32, pub dropped: u32 }`. At most 4 run; extra due ticks are dropped and not carried.
  - `pub struct Governor` with:
    - `fn new() -> Self`, where the scale starts at 1.0
    - `fn record(&mut self, now: f64, frame_ms: f64, budget_ms: f64) -> Option<f32>`, which returns the new scale when it changes
  - Governor rules:
    - steps are `[1.0, 0.875, 0.75, 0.625, 0.5]`;
    - the moving average covers 2 s;
    - step down when avg > 0.9 × budget, and up when avg < 0.6 × budget;
    - at least 3 s between steps.

- [ ] **Step 1: Write failing tests:**
  - `sixty_hz_from_144hz_frames`: 144 frames of dt = 1/144 run 60 ticks ± 1.
  - `catch_up_capped`: `advance(1.0)` gives `run 4`, `dropped 56`.
  - `alpha_range`: `alpha` stays within [0, 1).
  - `rate_change`: `set_rate(30)` halves the tick count.
  - `governor_steps_down`: a constant 15 ms against a 16.7 ms budget steps down after 2 s and again ≥ 3 s later.
  - `governor_floor`: the scale never goes below 0.5.
  - `governor_recovers`: a constant 5 ms steps back up to 1.0.
  - `governor_hysteresis`: alternating 15 ms and 5 ms every 1 s produces no change.
- [ ] **Step 2: Run `cargo test -p ms-engine --test clock --test governor`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-engine): fixed clock and quality governor`

---

### Task 20: Preset instance and per-tick execution

**Files:**
- Create: `crates/ms-engine/src/{env,tick}.rs`
- Test: `crates/ms-engine/tests/tick.rs`

**Interfaces:**
- Consumes: `CompiledPreset` (Task 12), `AudioFrame` (Task 7), `MeshData`/`DrawList`/`Uniforms` (Tasks 14–17), `Program::{run, run_batched, reads_before_write, batchable}` (Task 5).
- Produces:
  - `pub struct Globals { pub regs: [f64; 100], pub gmegabuf: Megabuf }`
  - `pub struct PresetInstance`, with:
    - `fn new(p: Arc<CompiledPreset>, seed: u64, mesh_cap: [u32; 2]) -> Self`, which runs `init` (§5.2)
    - `fn tick(&mut self, input: &TickInput, g: &mut Globals) -> TickOutput`
  - `pub struct TickInput<'a> { pub time: f64, pub fps: f64, pub frame: u64, pub progress: f64, pub audio: &'a AudioFrame, pub aspect: [f64; 2], pub pixels: [u32; 2] }`
  - `pub struct TickOutput { pub mesh: MeshData, pub draw: DrawList, pub uniforms: Uniforms, pub warnings: Vec<String> }`

**Order** (§5.2):
1. Write the read-only inputs.
2. Reset the writable variables to `params`.
3. Copy `regs` into `reg00..reg99`.
4. Run `frame`.
5. Snapshot `q1..q32`.
6. Run `vertex` on every mesh point:
   - `x`, `y` are in [0, 1], with y from the bottom;
   - `rad = dist from centre / sqrt(0.5)`;
   - `ang = atan2`;
   - each point starts from the post-frame writables and the q snapshot;
   - user variables carry over in row-major order.
7. Run waves and shapes (their `frame` from the q snapshot and their own `t1..t8`; then `point` per sample, with `sample` in [0, 1] and `value1`/`value2` from the L/R waveform resampled to `samples`).
8. Copy `reg00..reg99` back into `regs`.

**Batching rule:** the `vertex` code is batched with rayon over row chunks when `program.batchable()` holds and no element of `reads_before_write()` is a user variable (`VarId` ≥ `BUILTIN_VARS.len()`). Otherwise it runs on the scalar VM in row-major order.

**Mesh UV per vertex** (MilkDrop's warp formula):
1. Start from `(x - cx)` and `(y - cy)`, scaled by `1/sx` and `1/sy`.
2. Apply zoom: `zoom^(zoomexp^(rad*2-1))`.
3. Add the warp term: the sum of sin/cos terms of `time` and `warp`, as in MilkDrop. The exact four-term formula is written in a doc comment in `tick.rs`.
4. Rotate by `rot`.
5. Translate by `dx`, `dy` and `cx`, `cy`.

**Sanitising:** every `f64 → f32` conversion maps NaN/Inf to 0. UVs are clamped to [−10, 10].

- [ ] **Step 1: Write failing tests**, all with fixture presets written inline as TOML strings:
  - `identity_preset_identity_mesh`: default params give `uv == (x, y)` within 1e-6.
  - `zoom_param_shrinks`: `zoom = 1.1` gives corner UVs closer to the centre.
  - `frame_code_sees_bass`: `"zoom = 1 + bass"` with `bass = 0.5` makes `uniforms`/mesh reflect zoom 1.5.
  - `q_passes_to_vertex`: frame `"q1 = 0.25"`, vertex `"dx = q1"` shifts every u by 0.25.
  - `vertex_starts_from_frame_values`: frame `"rot=1"`, vertex reads `rot` and writes `zoom = rot + 1`; each vertex has zoom 2.
  - `user_var_carries_rowmajor`: vertex `"n = n + 1; dx = n*0.0001"`; the last vertex's dx = `(cols+1)*(rows+1)*0.0001`, and the program ran on the scalar path (assert via `#[cfg(test)]` counter).
  - `batched_used_when_safe`: vertex `"zoom = 1 + x*0.1"` takes the batched path, and the results equal the scalar path.
  - `user_vars_persist_ticks`: frame `"t = t + 1; q1 = t"`; on the third tick `q1 == 3`.
  - `writables_reset_each_tick`: frame `"zoom = zoom * 2"` gives 2 on every tick.
  - `regs_global`: preset A writes `reg05 = 7`; preset B reads 7.
  - `nan_sanitised`: vertex `"dx = sqrt(-1); zoom = 1/0.0000001"` leaves every uv finite and within [−10, 10].
  - `wave_point_code`: wave point `"x = sample; y = 0.5"` gives `samples` points along a horizontal line.
  - `shape_instances`: shape frame `"x = instance/4"` with `instances = 4` gives 4 draws at x 0, 0.25, 0.5 and 0.75.
  - `code_error_block_skipped`: a preset whose vertex code failed to compile still ticks and uses the frame values.
- [ ] **Step 2: Run `cargo test -p ms-engine --test tick`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Add the criterion bench** `engine/tick_64x48` and `engine/tick_192x144` with a ~60-op vertex program. Run `cargo bench -p ms-engine`; the 64×48 tick must take < 1 ms on the dev machine (record the number in the commit message).
- [ ] **Step 6: Commit:** `feat(ms-engine): preset instance and tick execution`

---

### Task 21: Blending

**Files:**
- Create: `crates/ms-engine/src/blend.rs`
- Test: `crates/ms-engine/tests/blend.rs`

**Interfaces:**
- Consumes: `PresetInstance`, `TickOutput` (Task 20), `Renderer::{warp, draw, composite}` (Tasks 14–17).
- Produces:
  - `pub enum BlendPattern { Uniform, RadialIn, RadialOut, Sweep { angle: f32 }, Plasma { seed: u32 } }`
  - `impl BlendPattern { pub fn random(rng: &mut Rng) -> Self; pub fn weight(&self, x: f32, y: f32, p: f32) -> f32 }`
  - `pub fn blend_weights(pattern: &BlendPattern, cols: u32, rows: u32, p: f32) -> Vec<f32>` (weight of the *incoming* preset per vertex)
  - `pub fn mix_meshes(a: &MeshData, b: &MeshData, w: &[f32]) -> MeshData`. Both meshes are resampled to the larger grid first if their sizes differ.

**Weight formula:** Uniform is `w = p`. Every other pattern uses `w = smoothstep(0, 1, p*2 - f)`:
- `f ∈ [0, 1]` comes from the pattern:
  - RadialIn: `f = 1 - d`;
  - RadialOut: `f = d`, where `d = dist from centre / sqrt(0.5)`;
  - Sweep: the projection of `(x, y)` onto the direction `angle`, normalised to [0, 1];
  - Plasma: 2-octave value noise from `seed`, in [0, 1].
- `p` is the blend progress in [0, 1].

**Renderer use during a blend:**
- `warp` gets two `WarpLayer`s: outgoing with weights `1-w`, incoming with `w`. Each uses its own shader and uniforms, over the *mixed* mesh.
- `draw` is called twice, with `alpha` `1-p` and then `p`.
- `composite` gets two layers, with alpha `1-p` and then `p`.

- [ ] **Step 1: Write failing tests:**
  - `weights_endpoints`: for every pattern, `p = 0` gives all 0 and `p = 1` gives all 1.
  - `weights_monotonic_in_p`: for every vertex, w is non-decreasing over p ∈ {0, 0.1, …, 1}.
  - `uniform_equals_p`
  - `radial_out_centre_first`: at `p = 0.5`, w(centre) > w(corner).
  - `mix_meshes_halfway`: at w = 0.5, the UV equals the average.
  - `mix_different_sizes`: meshes of 32×24 and 64×48 mix to 64×48.
  - `blend_golden`: two fixture presets (red-zoom-in and blue-rotate) blended over 60 ticks with RadialOut give `assert_golden("blend_radial_mid")` at tick 30.
  - `blend_continuity`: the first frame of a blend equals the outgoing preset's frame alone (p = 0) within the golden tolerance.
- [ ] **Step 2: Run `cargo test -p ms-engine --test blend`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-engine): preset blending`

---

### Task 22: Loader and director (transitions, hard cuts, blocklist)

**Files:**
- Create: `crates/ms-engine/src/{loader,director}.rs`
- Test: `crates/ms-engine/tests/director.rs`

**Interfaces:**
- Consumes: Tasks 12, 15, 16, 19–21.
- Produces:
  - **Loader:**
    - `pub struct Loader`, with `fn spawn(gpu: Arc<Gpu>, renderer_shared: RendererShared, resolver: TextureResolver) -> Self`, `fn request(&self, id: String, path: PathBuf)` and `fn poll(&self) -> Option<LoadResult>`
    - `pub enum LoadResult { Ready { id: String, preset: Arc<CompiledPreset>, shaders: ShaderSet, warnings: Vec<String> }, Failed { id: String, error: String } }`
    - `RendererShared` (in `ms-render`) holds `Arc`s of the bind-group layouts and pipeline layouts that `compile_shaders` needs. `Renderer::shared(&self) -> RendererShared`.
    - **Loader work:** the thread reads the file, runs `parse` and `compile_preset`, then `compile_shaders`, then a warm-up draw:
      - one warp pass and one composite into a private 64×64 target, with `device.poll(Wait)`;
      - the warm-up runs inside error scopes.
  - **Director:**
    - `pub enum Command { Next, Prev, Random, ToggleLock, HardCut, Load(String) }`
    - `pub struct DirectorConfig { pub auto_advance_s: f64 /*20*/, pub blend_s: f64 /*2.7*/, pub hard_cuts: bool /*true*/, pub hard_cut_strength: f32 /*2.0*/, pub hard_cut_min_gap_s: f64 /*10*/ }`
    - `pub struct Director`, with:
      - `fn new(cfg, lib: Library, prefs: Prefs, seed: u64) -> Self` (starts on the built-in default preset, embedded via `include_str!("../../../presets/default.splat")`)
      - `fn command(&mut self, c: Command)`
      - `fn update(&mut self, now: f64, audio: &AudioFrame, loader: &Loader)`
      - `fn current(&self) -> State`
      - `fn on_device_lost(&mut self, now: f64) -> Option<String>`, which returns the id to blocklist when the active preset became current less than 2 s ago
    - `pub enum State { Showing { cur }, Blending { from, to, start, dur, pattern } }`
    - **Behaviour:**
      - the next preset is requested from the loader as soon as the current one starts, and a switch happens only when it is `Ready`;
      - a manual `Load`/`Next` keeps the current preset until the requested one is ready;
      - `Failed` logs the error, adds the id to a session-skip set and requests another;
      - a hard cut swaps instantly, with no blend, when `beat_strength > hard_cut_strength`, at least `hard_cut_min_gap_s` after the last change, and when not locked;
      - `ToggleLock` stops both auto-advance and hard cuts.

- [ ] **Step 1: Write failing tests**, with a test loader built on `Gpu::headless()` and fixture presets in a temp dir:
  - `waits_for_ready`: a `Next` with the loader blocked stays on the current preset.
  - `auto_advance_starts_blend`: simulate 25 s at 60 Hz with a 20 s base duration, which reaches `Blending` once, then `Showing` 2.7 s later.
  - `lock_prevents_advance`
  - `hard_cut_on_strong_beat`: no blend, gap respected.
  - `prev_returns_previous`
  - `failed_preset_skipped`: a fixture with invalid TOML is never shown, and the director moves on.
  - `bad_shader_preset_still_shows`: its shaders fall back and it is shown.
  - `device_lost_blocklists_recent`: 1 s after activation it returns `Some(id)`; 5 s after, `None`.
  - `empty_library_stays_on_default`: 60 s with no library gives no requests and stays on the default.
  - `warmup_runs_off_thread`: `Ready` arrives with no render-thread involvement (the loader thread name is recorded).
- [ ] **Step 2: Run `cargo test -p ms-engine --test director`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat(ms-engine): loader and director`

---

### Task 23: App shell — window, render thread, hotkeys

**Files:**
- Create: `crates/milksplat/src/{main,app,render_thread,hotkeys,paths}.rs`
- Test: `crates/milksplat/tests/hotkeys.rs`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `pub enum Action { Next, Prev, Random, ToggleLock, Favourite, Ban, HardCut, AvOffset(i32 /*ms*/), CycleQuality, ShowName, ToggleFps, ToggleHelp, ToggleFullscreen, NextMonitor, EscapeKey }`
  - `pub fn map_key(key: &winit::keyboard::Key, mods: ModifiersState) -> Option<Action>`, mapping the §9.2 table. `Alt+Enter` maps to `ToggleFullscreen`; `[` and `]` give `AvOffset(-10)` and `AvOffset(10)`.
  - `pub enum ToRender { Resize { size: [u32; 2], scale: f64 }, Action(Action), Suspend(bool), Quit }` sent over a crossbeam channel.
  - `pub fn run_render_thread(surface: wgpu::Surface<'static>, window: Arc<Window>, rx: Receiver<ToRender>, cfg: Config) -> JoinHandle<()>`
- **Main thread:**
  - creates the window (`Arc<Window>`, title "Milksplat", 1280×720 default) and the surface;
  - spawns the render thread;
  - forwards events;
  - handles fullscreen (`Fullscreen::Borderless(Some(monitor))`), `NextMonitor`, double-click (two left clicks within 400 ms) and `Esc` (leave fullscreen, or quit when windowed).
- **Render thread loop:**
  1. Drain the channel.
  2. If the size is zero, sleep 50 ms and `continue`.
  3. `ring.drain_into(hist)`, then `hist.fill_silence_until(now)`.
  4. `clock.advance(dt)`.
  5. For each tick, run `analyzer.tick`, then `director.update`, then the instance tick(s) and the renderer `warp`/`draw`/`blur`.
  6. Composite into the surface texture.
  7. Draw text.
  8. Present.
  9. Feed the governor.
- **Presenting:**
  - `present_mode` is Mailbox if supported, else Fifo;
  - `desired_maximum_frame_latency = 1`;
  - `SurfaceError::Lost`/`Outdated` lead to a reconfigure;
  - `gpu.lost` triggers `Gpu::new` plus `renderer.recreate` plus `director.on_device_lost`. The `Loader` is then re-spawned with the new `Gpu`, and the current preset's shaders are rebuilt with `recompile`.

- [ ] **Step 1: Write failing tests:** `hotkeys.rs` asserts every row of the §9.2 table, plus `Alt+Enter`, maps to the expected `Action`, and unmapped keys give `None`.
- [ ] **Step 2: Run `cargo test -p milksplat --test hotkeys`.** Expected: FAIL.
- [ ] **Step 3: Implement the hotkeys, the app and the render thread.**
- [ ] **Step 4: Run `cargo test -p milksplat`.** Expected: PASS.
- [ ] **Step 5: Manual check with `cargo run -p milksplat --release`** while music plays:
  - the default preset reacts to the music;
  - `→` blends to the next preset (once Task 26 presets exist; until then, the fixture dir via `--presets <dir>`);
  - `F` toggles fullscreen;
  - dragging the window does not freeze the animation;
  - minimize/restore works.
- [ ] **Step 6: Commit:** `feat(milksplat): window, render thread, hotkeys`

---

### Task 24: Config, logging, crash handling, safe mode, keep-awake, hot reload, hints

**Files:**
- Create: `crates/milksplat/src/{config,logging,crash,keepawake,hotreload}.rs`
- Test: `crates/milksplat/tests/{config,crash}.rs`

**Interfaces:**
- Produces:
  - `pub struct Config { pub audio_device: Option<String>, pub av_offsets_ms: BTreeMap<String, i32>, pub auto_advance_s: f64, pub blend_s: f64, pub hard_cuts: bool, pub hard_cut_strength: f32, pub quality: Quality /*Auto|High|Medium|Low*/, pub favourites: BTreeSet<String>, pub bans: BTreeSet<String>, pub blocklist: BTreeSet<String>, pub window: WindowState, pub fullscreen_monitor: Option<String> }` (serde, every field `#[serde(default)]`)
  - `pub fn load_config(path: &Path) -> (Config, Vec<String> /*warnings*/)`:
    - a missing file gives defaults and is written out;
    - unreadable or invalid TOML gives defaults plus a warning, and the bad file is renamed to `config.toml.bad`;
    - an invalid individual field keeps the defaults for that field only (deserialise via `toml::Table`, field by field).
  - `pub fn save_config(path: &Path, c: &Config) -> std::io::Result<()>` (writes to a temp file, then renames)
  - **Paths:** `pub struct Paths { config, logs, crash_dir, user_presets, bundled_presets, bundled_textures }` from `%APPDATA%`, `%LOCALAPPDATA%`, `%USERPROFILE%\Documents` and the exe dir (§9.3).
  - **Logging:** `pub fn init_logging(dir: &Path) -> WorkerGuard` (tracing-appender daily rotation, 7 files kept)
  - **Crash handling:**
    - `pub fn install_panic_hook(paths: &Paths, current_preset: Arc<Mutex<Option<String>>>)`. It writes `crash-<timestamp>.txt` with the last 200 log lines (kept in an in-memory ring layer), the adapter info, the preset and `CARGO_PKG_VERSION`, then shows a `MessageBoxW`.
    - `pub struct RunMarker`, with `fn begin(dir: &Path) -> (Self, Option<String> /*preset from a crashed run*/)`, `fn set_preset(&self, id: &str)` and `fn end(self)` (deletes the marker)
  - **Safe mode:** `--safe` gives the built-in default preset only, user shaders disabled (`compile_shaders(None, None)`) and the default audio device.
  - **Keep-awake:** `keepawake::set(bool)` calls `SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED)`, true only while fullscreen and `silent_for < 10`.
  - **Hot reload:** `hotreload::watch(dirs) -> Receiver<PathBuf>` (notify, debounced 300 ms). A changed current preset is reloaded and swapped without a blend.
  - **No-audio hint:** while `silent_for >= 10`, show "No audio detected - is something playing?" at the bottom centre (render text). It uses an ASCII hyphen, because font8x8 has no em dash.
  - **A/V offset:** looked up per device name; Bluetooth defaults to 200 ms (§8.4). Adjusting it saves the config.

- [ ] **Step 1: Write failing tests:**
  - `missing_config_defaults_written`
  - `garbage_config_renamed_bad`
  - `one_bad_field_others_kept`: `auto_advance_s = "x"` with `blend_s = 5` gives `auto_advance_s == 20` and `blend_s == 5`.
  - `save_load_roundtrip`
  - `marker_detects_crash`: `begin`, `set_preset("a")`, drop without `end`; the next `begin` returns `Some("a")`.
  - `marker_clean_exit`: `begin` then `end`, and the next `begin` returns `None`.
- [ ] **Step 2: Run `cargo test -p milksplat --test config --test crash`.** Expected: FAIL.
- [ ] **Step 3: Implement**, and wire everything into `app.rs` and `render_thread.rs`.
- [ ] **Step 4: Run `cargo test -p milksplat`.** Expected: PASS.
- [ ] **Step 5: Manual check:**
  - edit a preset file while it is showing, and it updates within 0.5 s;
  - `--safe` starts on the default preset;
  - after a forced panic (debug-only hotkey `Ctrl+Shift+F12` under `#[cfg(debug_assertions)]`), a crash file exists, a message box appears, and the next start skips that preset.
- [ ] **Step 6: Commit:** `feat(milksplat): config, logging, crash handling, safe mode, hot reload`

---

### Task 25: Bundled presets and end-to-end golden tests

**Files:**
- Create: `presets/default.splat` plus 10 original presets in `presets/`:
  - `ink-bloom.splat`
  - `bass-tunnel.splat`
  - `prism-drift.splat`
  - `pulse-grid.splat`
  - `aurora-sweep.splat`
  - `kaleido-spin.splat`
  - `ember-rain.splat`
  - `liquid-chrome.splat`
  - `starfield-warp.splat`
  - `neon-orbit.splat`
- Create: `textures/clouds.png` (an original procedurally generated texture), `textures/random/` with 3 original generated PNGs, and `crates/ms-engine/tests/presets_e2e.rs`

**Interfaces:**
- Consumes: Tasks 20–22.

Requirements:
- **Coverage across the set:** at least 3 presets use a custom `warp` shader, 3 use a `comp` shader, 2 use blur, 2 use shapes with instances, 2 use custom waves, 1 uses `fp16`, 1 uses a user texture, and 1 uses `megabuf`.
- **Audio:** every preset reacts visibly to `bass`/`mid`/`treb`.
- **Metadata:** every preset has `meta.author = "Milksplat"` and `license = "MIT OR Apache-2.0"`.

- [ ] **Step 1: Write the test `presets_e2e`.**
  - For every file in `presets/`:
    1. Parse it, with no warnings and no `errors`.
    2. Compile its shaders, with no `errors`.
    3. Run 120 ticks with deterministic synthetic audio.
       - The audio is a mix of a 55 Hz kick on every 0.5 s, a 440 Hz pad and noise hats, made by `fn synth_audio(t) -> Frame` in the test file.
       - Seeds are fixed.
    4. `assert_golden("preset_<name>")` on the composite.
  - **Reaction check:** the tick-120 image with audio differs from the image with silence (> 5% of pixels).
- [ ] **Step 2: Run `cargo test -p ms-engine --test presets_e2e`.** Expected: FAIL (no presets).
- [ ] **Step 3: Write the presets.** Tune each live with hot reload (`cargo run -p milksplat --release -- --presets presets`), then bless the goldens with `MS_BLESS=1` and inspect every PNG by eye.
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit:** `feat: bundled presets and end-to-end golden tests`

---

### Task 26: Release packaging and manual checklist

**Files:**
- Create: `scripts/package.ps1`, `README.md`, `docs/release-checklist.md`
- Modify: `crates/milksplat/Cargo.toml` (`[package.metadata]` version `0.1.0`)

Requirements:
- **`package.ps1`:**
  - runs `cargo test --workspace` and stops on failure;
  - runs `cargo build --release -p milksplat`;
  - builds `dist/milksplat-<version>-win64/` containing `milksplat.exe`, `presets/` (excluding `default.splat`, which is embedded), `textures/`, `LICENSE-MIT`, `LICENSE-APACHE` and `README.md`;
  - zips it to `dist/milksplat-<version>-win64.zip`.
- **`README.md`:**
  - what it is, how to run it, the hotkey table (§9.2) and where presets go;
  - writing presets: a short `.splat` example plus a link to the spec §4–§5;
  - the SmartScreen note: "Windows may warn that the app is unrecognised; click More info → Run anyway";
  - the licences.
- **`release-checklist.md`:** the manual checklist from spec §10, one checkbox each, plus "the zip runs on a clean Windows VM without installing anything".

- [ ] **Step 1: Run `pwsh scripts/package.ps1`** (or `powershell -File`). Expected: the tests pass and the zip exists.
- [ ] **Step 2: Check static linking.** Run `dumpbin /dependents dist\...\milksplat.exe`; the output must list no `VCRUNTIME140.dll`.
- [ ] **Step 3: Work through `docs/release-checklist.md` on real hardware and tick every item.** Any failure is fixed in the owning crate, with a regression test, before continuing.
- [ ] **Step 4: Commit:** `chore: release packaging, README, checklist`

---

## Spec coverage

| Spec § | Tasks |
|---|---|
| 1.2 success criteria | 20 (bench), 22 (no-hitch loader), 19 (clock), 26 (checklist) |
| 2 stack | 1 |
| 3.1 crates | 1, all |
| 3.2 threads | 10 (audio), 22 (loader), 23 (main and render), 20 (workers) |
| 3.3 clock | 19, 23 |
| 4 format | 11, 16 (texture lookup) |
| 5.1–5.4 language | 1–5, 20 |
| 6.1 resources | 14, 16 |
| 6.2–6.4 tick, composite, waves and shapes | 14, 15, 17 |
| 6.5 shader interface | 15 |
| 6.6 governor and adapter | 19, 13 |
| 6.7 robustness | 13, 15, 18, 22, 23 |
| 7 transitions and selection | 12, 21, 22 |
| 8 audio | 6–10, 24 (offset and hint) |
| 9 app shell | 23, 24 |
| 10 testing | every task; benches in 5 and 20; e2e in 25 |
| 11 packaging | 1 (profile and CRT), 26 |
