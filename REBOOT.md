# MetricsQNM.jl — Reboot Plan (2026-06)

## Why a reboot
The `main` branch (this branch) carries a proven GR core but a **broken, over-engineered
sGB extension** driven by a now-falsified dogma (see `CLAUDE.md` header). Empirically, its
headline algorithm (per-equation matched clearing + full Newton on (v,ω)) **does not
converge**: at a=0.3, ζ=0.01, seeded at Leaver, N=16/18 hit max-iter with residual stuck at
~1e-8, ω drifts off the fundamental, and ω₁≈|10| vs the paper's O(0.1) — and it gets *worse*
with N. Nothing in the test suite asserts correctness, so this rotted silently.

## Decisions (2026-06)
- **H_i background:** STAGE IT. Phase-1 export from notebook → committed Julia data
  (Mathematica-free at runtime); Phase-2 recompute natively in Julia to surpass.
- **"Outperform":** speed + reproduce paper Tables I & II first (with a real regression
  harness), then exceed (more digits/spin/modes).
- **Validation:** wolframscript generates golden masters at dev time only; every stage is
  asserted against an oracle in `test/runtests.jl`.

## Target architecture (post-reboot src/)
```
core/    kerr, perturbation_ansatz, spectral, leaver          (proven, keep)
cas/     sparse_poly                                          (keep; demarcate; N_VARS revisit)
gr/      linearize, symbolic_pipeline(GR only), assembly, rectangular_qep   (keep, slim)
sgb/     sgb_background(data-loader), sgb_equations, sgb_assembly, sgb_perturbation
goldens/ exported H_i data + Wolfram-generated reference values + the .wls exporters
```
Two public entry points over shared internals: `build_qnm_gr`, `build_qnm_sgb`.
Export the real solver (`solve_qep_svd`, `solve_qep_full_newton`, `solve_sgb_perturbation`).

## Phases & gates

### Phase 0 — Foundation (in progress)
- [x] Rewrite `CLAUDE.md` (dogma → paper algorithm + Mathematica-free + golden-master discipline).
- [x] Write this plan.
- [ ] `goldens/` infra + wolframscript exporter for H_i (and ϑ) → committed JSON data.
- [ ] Pure-Julia loader replacing `.nb` parsing in `sgb_background.jl`; validate vs golden master.
- [ ] Safe dead-code deletion (failed symbolic-a scaffolding; dead `sgb_linearize` half).
- **Gate:** `using MetricsQNM` loads with no `.nb` dependency; GR Table I still 219/220; H_i
  Julia values match golden master to ~1e-12 at sample (r,χ,a).

### Phase 1 — Make the sGB result correct (the actual blocker)
- [ ] Implement the paper's **step 2: divide out each equation's common Σ^p Δ^q (1−χ²)^s
      prefactor** after clearing (prime suspect for the broken ω₁).
- [ ] Validate the combined system at ζ=0 reproduces GR ω₀ to machine precision (gauge sanity).
- [ ] Validate D̃⁰ and D̃¹ coefficients against Wolfram golden masters at sample points.
- [ ] Re-run a=0.3 N-sweep; confirm full Newton converges and ω₁ → O(0.1), stable in N.
- **Gate:** ω₁(a=0.3, nlm=022) matches paper Table II to its stated accuracy; converges in N.

### Phase 2 — Reproduce + harden
- [ ] N-sweep with backward-modulus-difference N-selection; spin sweep; Tables I & II.
- [ ] Regression harness in `runtests.jl` (hardcoded Table I & II expected values + golden masters).
- [ ] Performance pass (kill the ~6 min/N build cost; profile; cache extraction).
- **Gate:** Tables I & II reproduced in CI-runnable tests; pipeline materially faster than original.

### Phase 3 — Exceed the original
- [ ] Native-Julia H_i background (replace exported data), validated vs golden master.
- [ ] sGB Sources 2–4 + scalar sector (7th unknown h₇, 11th equation) — the real physics gap.
- [ ] More modes (033, 021), higher spin, more digits; fit ω⁽¹⁾(a) (Eqs. 125–130).

## Memory-safety rule (WSL2, OOM history)
All Julia runs capped: `ulimit -v ~24-28GB`, `--threads=2`, small N first, no unbounded sweeps.
wolframscript segfaults on shutdown *after* output — capture stdout, ignore exit code.
