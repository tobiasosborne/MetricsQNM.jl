# Handoff — MetricsQNM.jl sGB Extension

## Project in one sentence

Compute quasinormal mode (QNM) frequency corrections in scalar-Gauss-Bonnet (sGB) gravity by eigenvalue perturbation theory, reproducing Tables I-III of arxiv:2406.11986.

## What works (GR foundation — COMPLETE)

The GR METRICS pipeline reproduces **219/220 digits** of Table I (arxiv:2312.08435) for the fundamental (n=0, l=2, m=2) Kerr QNM across 11 spins a=0.005..0.95.

Pipeline: `compute_field_equations(2)` → symbolic linearized Ricci → `extract_G_bespoke` (bespoke SparsePoly CAS) → `build_system_bespoke(a, N, m)` → `METRICSSystem(D0, D1, D2)` → `solve_qep_svd` (SVD compression + companion QZ) → eigenvalue at machine precision.

Key insight: the QEP solver (not Newton-Raphson) is the production method. It finds ALL eigenvalues at once, verified to 1e-14 on synthetic benchmarks. The QEP audition is at `../af-tests/examples13/REPORT_rectangular_qep.md`.

## sGB extension — current state (Phase 2 of 4 in progress)

**Goal**: Eq. 111 of 2406.11986: `x⁽¹⁾ = -J⁻¹ · [D̃⁽¹⁾(ω⁰) · v⁰]`, where ω⁽¹⁾ = last component of x⁽¹⁾.

| Component | Status | File |
|-----------|--------|------|
| **GR solution** (ω⁰, v⁰, J) | **DONE** | `newton.jl` — `solve_qnm` returns J, free_idx, pinned_idx |
| **sGB background** (H₁-H₄, ϑ) | **DONE** | `sgb_background.jl` — parses Mathematica notebook, verified 12 digits |
| **sGB linearization** (compiled correction evaluator) | **DONE** | `sgb_linearize.jl` — 10 compiled equations, 1.7s/point after JIT |
| **Eigenvalue perturbation solver** | **DONE** | `sgb_perturbation.jl` — `solve_sgb_perturbation(sys_corr, ω0, v0, J)` |
| **D̃⁽¹⁾ collocation assembly** | **TODO** | Wire compiled evaluator into spectral collocation assembly |
| **End-to-end test** | **TODO** | Compare ω⁽¹⁾ to Tables I-III of 2406.11986 |

## EXACTLY what needs to be done next

### Step 1: Build `build_sgb_Dtilde_collocation` (~150 lines)

At each collocation point (r_j, χ_k) on the spectral grid:

1. Compute `hp = sgb_H_params(bg, r_val, χ_val)` — 24 numerical values (H_i + derivatives via finite diff)
2. Call `separate_sgb_omega(csc, r, χ, a, hp)` → `(C0, C1, C2)` — 10×40 complex correction coefficient matrices
3. These matrices tell you: correction equation k gets contribution `C0[k,d] + C1[k,d]·ω + C2[k,d]·ω²` from h-derivative d

The 40 h-derivative terms map to spectral coefficients via the same derivative/multiplication matrices used in the GR assembly (`spectral.jl`). Look at how `build_Dtilde` in `dtilde.jl` or `build_factored_system` in `factored_assembly.jl` does this for the GR case.

The output is `METRICSSystem(D0_corr, D1_corr, D2_corr, N, m, a)` — the correction matrices.

**Performance**: After JIT warmup (~355s one-time), each point takes ~1.7s. For N=8 (81 points) that's ~138s. Acceptable.

### Step 2: End-to-end test (~50 lines)

```julia
a = 0.3; N = 8; m = 2
bg = sgb_background(a)
sys_gr = build_system_bespoke(a, N, m)
gr = solve_qnm(sys_gr, ω_guess; parity=:polar)  # → ω⁰, v⁰, J

sys_corr = build_sgb_Dtilde_collocation(a, N, m, bg, csc)
ω1 = solve_sgb_perturbation(sys_corr, ComplexF64(gr.ω), gr.v, gr.J)
# Compare to paper: ω⁽¹⁾_polar(a=0.3) ≈ -0.36419 - 0.04236i
```

### Step 3: Sweep Table I (all spins, both parities)

Ground truth from paper's Table I (nlm=022):
```
a=0.005: axial  0.05984+0.00719i  polar -0.22664-0.07525i
a=0.1:   axial  0.07282+0.01353i  polar -0.26228-0.07146i
a=0.3:   axial  0.09087+0.00782i  polar -0.36419-0.04236i
```

## Critical technical knowledge

### The symbolic blowup and how we solved it

The sGB metric corrections H₁-H₄ are 27-term rational functions of (r, χ) at fixed spin. Naively inserting them into the symbolic linearization produces expressions of **100-155 million characters** because Symbolics.jl doesn't auto-simplify rational functions (unlike Mathematica which does GCD at every step).

**Solution**: Use 24 ABSTRACT symbolic variables (`H1, H1_r, H1_chi, H1_rr, H1_chichi, H1_rchi, H2, ...`) instead of the actual polynomial expressions. The correction equations are LINEAR in these, so expressions stay at ~200K chars (comparable to GR). At evaluation time, substitute numerical values from `sgb_background`.

This is implemented in `sgb_linearize.jl`. The compiled evaluator takes 70 arguments: 6 base (r, χ, ω_re, ω_im, iu, a) + 24 H params + 40 h-derivative placeholders.

### The iu-polynomial trick

Complex arithmetic is handled by treating the imaginary unit as a symbolic variable `iu`. Evaluate at iu=0, 1, -1, extract polynomial coefficients, substitute iu→im. This avoids complex symbolic algebra. See `extract_coefficients_complex` in `galerkin.jl`.

### ω-polynomial separation

The D̃ matrices are quadratic in ω: `D̃(ω) = D̃₀ + ω·D̃₁ + ω²·D̃₂`. Extract by evaluating at ω=0, 1, -1 and solving the 3-point system. See `separate_omega_dependence` in `galerkin.jl`.

### What D̃⁽¹⁾ is missing (known limitation)

The current `sgb_linearize.jl` computes D̃⁽¹⁾ from **source 1 only**: the O(ζ) correction to the linearized Ricci from the modified background metric g = g_Kerr + ζ·H. There is also **source 2**: the linearized sGB source tensor A_μ^ν (Eq. 12 of the paper), which involves the Gauss-Bonnet invariant contracted with ∇∇ϑ. This is NOT yet implemented. If results from source 1 alone don't match the paper, source 2 must be added. TensorGR.jl at `../TensorGR.jl` has `euler_density()` and `generalized_delta()` that could help.

### The bespoke SparsePoly CAS (`sparse_poly.jl`)

For the GR pipeline, we built a custom CAS to replace `Symbolics.simplify_fractions` (which hangs). It represents polynomials as `Dict{NTuple{5,Int}, Float64}` with structural denominator tracking. This was essential for the GR G-coefficient extraction. A similar approach could be built for the sGB tensor algebra (option 3 from the plan) to eliminate the JIT warmup cost.

## Conventions

- `M = 1` everywhere. Table values are `ωM` (dimensionless).
- SpinWeightedSpheroidalHarmonics.jl returns Teukolsky λ, NOT Leaver A_slm. Convert: `A_slm = λ - c² + 2mc`.
- Use `--threads=8` for Julia. More threads (32+) cause OpenBLAS contention.
- TensorGR.jl at `../TensorGR.jl` — actively developed, DO NOT edit.
- The user prefers QEP over Newton-Raphson, rich diagnostic output with flush after every print, and fail-fast behavior.

## File map

```
src/
  MetricsQNM.jl              — module root
  kerr.jl                    — KerrParams, Σ, Δ, r±
  perturbation_ansatz.jl     — A_k(r), z↔r transforms
  spectral.jl                — ChebyshevBasis, LegendreBasis
  leaver.jl                  — Leaver CF QNM solver (reference values)
  linearize.jl               — Symbolic: Kerr → δΓ → δR → 10 field equations
  coefficients.jl            — compile_field_equations → fast callables
  galerkin.jl                — extract_coefficients_complex, separate_omega_dependence
  sparse_poly.jl             — Bespoke SparsePoly CAS (replaces simplify_fractions)
  symbolic_pipeline.jl       — extract_G_bespoke, build_system_bespoke, sweep_N
  assembly.jl                — PDECoefficients → D̃ via spectral inner products
  rectangular_qep.jl         — SVD compression QEP solver (the winner)
  newton.jl                  — solve_qnm (returns J for perturbation theory)
  sgb_background.jl          — Parse H₁-H₄, ϑ from Mathematica + derivative evaluator
  sgb_linearize.jl           — Perturbative sGB correction: compile + extract coefficients
  sgb_perturbation.jl        — Eigenvalue perturbation solver (Eq. 111)
  [dtilde.jl, factored_assembly.jl, collocation.jl] — earlier D̃ approaches (reference)
  [poly_extract.jl, zspace_extract.jl, pipeline.jl] — dead ends (kept for reference)
test/
  reproduce_paper.jl         — Generates Figs 1,2,5,6 from the GR paper
  reproduce_table1.jl        — Table I reproduction (219/220 digits)
reference/
  2406.11986_source/         — sGB paper supplementary materials (Mathematica notebook)
```

## Key function signatures

```julia
# GR pipeline
sys = build_system_bespoke(a, N, m)          # → METRICSSystem(D0, D1, D2)
eigs = solve_qep_svd(sys; ω₀=ω_L, refine=1) # → all eigenvalues
gr = solve_qnm(sys, ω_guess; parity=:polar)  # → (ω, v, J, free_idx, ...)

# sGB background
bg = sgb_background(a; verbose=true)         # → SGBBackground with H[1..4](r,χ)
hp = sgb_H_params(bg, r, χ)                  # → 24-element Float64 vector

# sGB correction (compiled evaluator)
csc = compile_sgb_correction(2; verbose=true) # → CompiledSGBCorrection (one-time)
C0, C1, C2 = separate_sgb_omega(csc, r, χ, a, hp)  # → 10×40 complex matrices

# Perturbation solve
ω1 = solve_sgb_perturbation(sys_corr, ω0, v0, J)   # → ω⁽¹⁾ ∈ ℂ
```
