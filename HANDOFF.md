# Handoff — MetricsQNM.jl sGB Extension

## Project in one sentence

Compute quasinormal mode (QNM) frequency corrections in scalar-Gauss-Bonnet (sGB) gravity by eigenvalue perturbation theory, reproducing Tables I-III of arxiv:2406.11986.

## What works (GR foundation — COMPLETE)

The GR METRICS pipeline reproduces **219/220 digits** of Table I (arxiv:2312.08435) for the fundamental (n=0, l=2, m=2) Kerr QNM across 11 spins a=0.005..0.95.

Pipeline: `compute_field_equations(2)` → symbolic linearized Ricci → `extract_G_bespoke` (bespoke SparsePoly CAS) → `build_system_bespoke(a, N, m)` → `METRICSSystem(D0, D1, D2)` → `solve_qep_svd` (SVD compression + companion QZ) → eigenvalue at machine precision.

Key insight: the QEP solver (not Newton-Raphson) is the production method. It finds ALL eigenvalues at once, verified to 1e-14 on synthetic benchmarks.

## sGB extension — current state

**Goal**: Eq. 111 of 2406.11986: `x⁽¹⁾ = -J⁻¹ · [D̃⁽¹⁾(ω⁰) · v⁰]`, where ω⁽¹⁾ = last component of x⁽¹⁾.

| Component | Status | File |
|-----------|--------|------|
| **GR solution** (ω⁰, v⁰, J) | **DONE** | `rectangular_qep.jl` → `solve_qep_with_vectors` + `compute_jacobian` |
| **sGB background** (H₁-H₄, ϑ) | **DONE** | `sgb_background.jl` — parses Mathematica notebook, verified 12 digits |
| **sGB linearization** (compiled correction evaluator) | **DONE** | `sgb_linearize.jl` — 10 compiled equations, ~6s/point after JIT |
| **Eigenvalue perturbation solver** | **DONE** | `sgb_perturbation.jl` — `solve_sgb_perturbation(sys_corr, ω0, v0, J)` |
| **β=3 assembly extension** | **DONE** | `assembly.jl` — `_leg_d3` for 3rd χ-derivative |
| **D̃⁽¹⁾ assembly (numerical)** | **ABANDONED** | `sgb_galerkin.jl` — polynomial fitting failed (C[k,d] not polynomial) |
| **D̃⁽¹⁾ assembly (exact SparsePoly)** | **IN PROGRESS** | `sgb_symbolic_pipeline.jl` — Phase 2-3 work, awaiting full run completion |
| **End-to-end test** | **PARTIAL** | `test/test_sgb_e2e.jl` runs, ω₁ = O(1) but not converged |

## What the paper actually does (deep review 2026-03-20)

A thorough review of arxiv:2406.11986 revealed the paper uses **the exact same Galerkin assembly** for D̃⁽¹⁾ as for D̃⁽⁰⁾:

1. Symbolically linearize the FULL sGB field equations about g = g_Kerr + ζH
2. The H_i(r,χ) are rational functions → after clearing, everything is polynomial in r, χ
3. Extract K^(η=1) coefficients (numbers, not functions!) via symbolic pipeline
4. Assemble via same spectral inner products as GR

The K^(η=1) coefficients are just numbers because the entire symbolic pipeline stays rational. This is NOT possible in our code because inserting H_i content into Symbolics.jl causes 100-155M char expression blowup on WSL2.

### Critical divergences from paper

1. **Assembly method**: Paper uses symbolic K-extraction; we use numerical evaluation
2. **Missing sources**: We only have Source 1 (modified Ricci from H). Paper needs all 4:
   - Source 1: O(ζ) Ricci correction from modified background ✓
   - Source 2: linearized sGB source tensor A_μ^ν ✗
   - Source 3: linearized scalar stress-energy T_μ^ν ✗
   - Source 4: Ω_H¹, κ¹ corrections in A_k ✗
3. **System size**: Paper is 11 eqs × 7 unknowns (includes scalar h₇); we are 10×6

## THE CURRENT PROBLEM: C[k,d] is NOT polynomial

### What was discovered (2026-03-20)

The correction coefficients C[k,d](r, χ) from `extract_sgb_coefficients_complex` are:
- **O(1) everywhere** — raw values range from ~0.5 to ~6.0
- **Smooth** on [r₊, ∞) × [-1, 1]
- **Asymptote to a nonzero constant** (~1.2) as r → ∞

This means C[k,d] is NOT polynomial in r, NOT polynomial in z, and NOT rational with a finite denominator. It's a genuinely smooth function that requires spectral representation.

### What was tried and failed

1. **r-space clearing + Vandermonde fit** (P=4, Q=2, S=2): Clearing factor Σ^P Δ^Q (1-χ²)^S makes values HUGE at large r (up to 1e25). Polynomial fit gives rel. error ~1e16.

2. **z-space monomial Vandermonde fit**: C[k,d](z, χ) is smooth on [-1,1] but NOT polynomial in z (the rational structure of H_i introduces 1/(1+z)^n terms). Fit error ~1e8, ω₁ does not converge with increasing grid size.

3. **z-space Chebyshev interpolation → monomial**: Same issue — the Chebyshev expansion converges but the monomial conversion amplifies high-degree coefficients. ω₁ unstable.

## Exact SparsePoly extraction — STATUS (2026-03-20)

**Path B runs end-to-end.** First ω₁ computed 2026-03-21.

### What's done:
1. **DenomSig extended** with 4th field `t` for r-power tracking (backward-compat, GR still 11.9 digits at N=8)
2. **SparsePoly differentiation** — exact `differentiate(p, var_idx)` function
3. **H_i → 24 RatPoly** — `load_H_ratpolys(a)` parses Mathematica notebook, converts to SparsePoly, differentiates symbolically. Values match numerical to ~1e-16, derivatives are now EXACT (vs O(ε²) finite diff before)
4. **c_{k,d,p} probing** — `extract_sgb_coefficients_symbolic(a)` double-probes sGB equations: 203 non-zero (k,d) pairs, 1165 non-zero (k,d,p) triples (~63s)
5. **Multiply + accumulate** — `combine_sgb_K` multiplies c_{k,d,p} × H_p in RatPoly land, clears denominators, ω-decomposes. 203 pairs: ~975-4417 poly terms, clearing P≤11,Q≤4,S=1,T≤57
6. **Full pipeline** — `build_sgb_system_bespoke(a, N, m)` wires everything through _r_to_z + assemble_system
7. **test_sgb_e2e.jl updated** to use `build_sgb_system_bespoke`
8. **test/run_sgb_bespoke.jl** — lean runner script for quick validation

### First ω₁ result (2026-03-21, a=0.3, N=4, Source 1 only):
```
ω⁽⁰⁾ = 0.4195266818 - 0.0877292719i  (Leaver match: 3.18e-14)
ω⁽¹⁾ = 4.678124 - 3.375967i           (Source 1 only)
Paper: ω⁽¹⁾ = -0.364190 - 0.042360i   (all 4 sources)
```
ω₁ is O(1), consistent with earlier numerical Galerkin attempts. The paper's much smaller value suggests near-cancellation between Sources 1-4.

### Bugs fixed in this session:
1. **`binomial` Int64 overflow** in `_r_to_z`: sGB has d_max=80, binomial(77,22) > typemax(Int64). Fixed via BigInt precomputation.
2. **`spectral_basis` hardcoded max_delta=25**: sGB needs z-degree up to 80 and chi-degree up to 57. Fixed by parameterizing `chebyshev_basis(N; max_delta)` and `legendre_basis(N, m; max_sigma)`.
3. **`reduce_ratpoly` performance**: r-power stripping via O(n) exponent subtraction instead of O(24×T×n²) polynomial division. Phase 3 speedup: 2405s → 1709s (29%).

### Critical finding: d_max incompatibility (2026-03-21)

The `_r_to_z` transform multiplies each equation by `(1-z)^d_max` to make r-polynomials into z-polynomials. GR uses d_max=9, sGB uses d_max=80. These are **fundamentally incompatible** in the spectral basis:

1. **Can't use shared d_max=80**: (1-z)^80 creates degree-89 polynomials that can't be resolved by N+1=5 Chebyshev modes. The GR eigenvalue shifts by 0.06 (catastrophic truncation error).
2. **Can't use different d_max**: The perturbation formula J·x₁ = -D̃⁽¹⁾·v₀ requires J and D̃⁽¹⁾ in the same "units", but different (1-z)^d_max factors make them incomparable.
3. **Independent normalization doesn't fix it**: The normalization mismatch is a secondary effect. The primary issue is that D̃⁽⁰⁾ and D̃⁽¹⁾ represent different polynomial spaces.

**N-convergence test results (Source 1 only, a=0.3):**
- With independent normalization: ω₁ = 4.68, 1.93, 0.80 (NOT converging — changes sign)
- With GR normalization on D̃⁽¹⁾: ω₁ = O(10^40) (raw D̃⁽¹⁾ is O(10^42))
- With shared d_max=80: ω₀ shifts by 0.06, cond(J) = 3.2e+17, ω₁ = O(10^26)

### ROOT CAUSE FOUND (2026-03-21, late session)

**The paper keeps `a` (spin) SYMBOLIC and works per-a-order.** The Mathematica notebook stores H_i as power series in `a^{2k}` (k=0..19, up to a^38). Each a-order has moderate r-denominators (r^{2k+5} ≈ r^{5} to r^{43}). Our pipeline evaluates `a` numerically FIRST (in `load_H_ratpolys`), summing all 20 a-orders into one rational function with LCD den.t=66. This inflates d_max from ~15 to ~80.

**The fix**: Load H_i as per-a-order RatPolys: H_i = Σ_k a^{2k} H_i^{(2k)}(r,χ). Each H_i^{(2k)} has moderate den.t ≈ 2k+5. Then `combine_sgb_K` processes each a-order separately with d_max ≈ 15-20. After assembly: D̃⁽¹⁾ = Σ_k a^{2k} D̃^{(2k)}.

**Evidence**: H₁ in notebook has a^{2k} for k=0..19, r-powers up to r^47 (TOTAL across all orders). Individual orders have much lower r-powers. The paper converges at N=20-25, confirming effective d_max ≈ 20-30 (not 80).

### Per-a-order pipeline — IMPLEMENTED (2026-03-21)

1. ✅ `load_H_ratpolys_per_order()` — parses H_i with `a` as SparsePoly variable (slot 3), splits by a-exponent via `_split_ratpoly_by_var`, reduces per-order r-denominators
2. ✅ `combine_sgb_K` called once per a-order with that order's 24 H_ratpolys
3. ✅ Assembly per-a-order with shared d_max across all retained orders
4. ✅ D̃⁽¹⁾ = Σ_k a^{2k} D̃^{(2k)} — matrix sum with numerical a weights
5. ✅ GR system rebuilt with `d_max_override` to match sGB shared d_max
6. ✅ Truncation: a-orders with `a^{2k} < epsilon` are dropped (default epsilon=1e-14)
7. ✅ Verification: `verify_H_ratpolys_per_order` checks per-order sum matches original

**Key design decisions:**
- Variable slot 3 reused: `a` during H_i parsing → stripped to 0 after splitting → `ω_re` during combine
- `SymToPolyCtx(var_list, 1.0)` dummy for H_i conversion (Σ/Δ polys don't match r-only denoms)
- Shared d_max across all retained orders AND GR system (required for consistent Galerkin weighting)
- `c_{k,d,p}` stays numerical in `a` — only H_p needs per-order splitting

### First end-to-end results (2026-03-21, a=0.3, epsilon=1e-14, N=8):

Per-order d_max values: 24, 26, 30, 33, 35, 39, 42, 46, 50, 53, 56, 60, 63, 66.
Pattern: d_max ≈ 24 + 3k (k = a-order index). The base d_max=24 comes from
the sGB correction c_{k,d,p} having high r-degrees from Kerr background.

**Shared d_max=66 with N=8 is catastrophically under-resolved:**
- GR ω⁰ shifted by 0.046 (should match Leaver to 1e-14)
- ‖D‖ = O(10^26-38), cond(J) = 1.7e+18
- ω₁ = O(10^9) (garbage)

**Practical parameter choices** (a=0.3):
| epsilon | n_a_max | shared d_max | min N needed |
|---------|---------|-------------|-------------|
| 1e-14   | 14      | ~66         | ~65         |
| 1e-4    | 4       | ~33         | ~30         |
| 1e-3    | 3       | ~30         | ~25         |
| 1e-2    | 2       | ~26         | ~22         |

The paper converges at N=20-25, suggesting they use effective truncation of ~3-5 a-orders.

**1.5% verification error** in per-order H_i splitting (at r=5, χ=0.5):
Likely from different Symbolics simplification paths when `a` is symbolic vs.
numerical. Need investigation — may require using the same Symbolics path
for both, or tighter cleanup tolerances.

### Next steps:
1. **Investigate 1.5% verification error** — check if cleanup tolerance or Symbolics simplification
2. **Run with epsilon=1e-3, N=25** — should give manageable d_max ≈ 30
3. **Implement Sources 2-4** (see roadmap below)
4. **Parallelize Phase 3**: gather/merge pattern from GR pipeline could give ~4x speedup

### The correct path forward

**The assembly needs coefficients as z^δ monomials** (because the spectral operators are `Z[δ+1] * D` etc.). But C[k,d] is smooth, not polynomial. Two viable paths:

**Path A: Spectral projection (recommended)**
Instead of fitting C as polynomial, project it directly onto the spectral basis:
```
D̃⁽¹⁾[(k,n,l), (j,n',l')] = ∫∫ C[k,d](z,χ) × T_n(z) × P_l^m(χ) × T_{n'}(z) × P_{l'}^m(χ) w(z,χ) dz dχ
```
Compute this integral via Gauss-Chebyshev × Gauss-Legendre quadrature. This avoids polynomial fitting entirely — the matrix elements are computed directly as numerical integrals. The C[k,d] values at quadrature points are O(1), no blowup.

**Subtlety**: This computes D̃⁽¹⁾ in h-derivative space (the C[k,d] multiply h-derivatives). The GR assembly uses the same convention (r-derivative indices passed through as z-derivative operators). For consistency, the spectral projection should use the SAME operator convention as the GR assembly.

**Path B: Extend SparsePoly to handle sGB**
Extend the bespoke SparsePoly CAS to extract K^(η=1) symbolically by:
1. Probing the Symbolics expressions for H-parameter coefficients (24 probes per (k,d))
2. Representing each c_{k,d,p}(r,χ) as RatPoly (with Σ, Δ denominators)
3. Multiplying by H_p(r,χ) in SparsePoly land
4. This avoids the 155M char blowup because H_i content enters as SparsePoly, not Symbolics

This gives exact K coefficients but requires more implementation work and careful memory management for the ~2000 SparsePoly products.

## What the compiled evaluator produces

`compile_sgb_correction(2)` gives a `CompiledSGBCorrection` with:
- 10 compiled functions (one per equation component)
- 40 h-derivative terms (verified, names printed in test output)
- Derivatives up to 3rd order: ∂³h₅/∂χ³, ∂³h₆/∂χ³, ∂²r∂χ h₅, ∂r∂²χ h₅/h₆
- 24 H-parameters, 6 base variables (r, χ, ω_re, ω_im, iu, a)
- JIT compile time: ~30s. Evaluation: ~6s/point first run, ~0.07s/point after JIT warm-up

`extract_sgb_coefficients_complex(csc, r, χ, ω₀, a, hp)` returns a 10×40 complex matrix C where `correction_k = Σ_d C[k,d] × h_deriv_d`. These coefficients are O(1) and well-behaved everywhere.

## Critical technical knowledge

### The derivative convention mystery

The GR pipeline uses r-derivative indices (α_r from `_parse_h_terms`) but applies z-derivative operators in assembly (`_z_operator` uses ∂_z, (1-z²)∂²_z`). This does NOT include the chain-rule factor dz/dr. Yet it produces 219/220 correct digits. The sGB code must use the SAME convention for consistency.

### The symbolic blowup and how we solved it

Use 24 ABSTRACT symbolic variables (`H1, H1_r, H1_chi, ...`) instead of actual polynomial expressions. Equations stay at ~200K chars. Numerical values substituted at evaluation time.

### The iu-polynomial trick

Complex arithmetic via symbolic `iu` variable. Evaluate at iu=0, 1, -1 → reconstruct complex result.

### Missing sources (4 total for full paper reproduction)

Source 1 (DONE): O(ζ) Ricci correction from H background
Source 2 (TODO): linearized sGB source tensor A_μ^ν (involves ∂²ϑ, curvature)
Source 3 (TODO): linearized scalar stress-energy T_μ^ν
Source 4 (TODO): Ω_H¹, κ¹ corrections in A_k asymptotic factor
Plus: 7th unknown h₇ (scalar field perturbation) + 11th equation (scalar wave eq)

### Performance note

After JIT warm-up, `extract_sgb_coefficients_complex` runs at ~15 pts/s with 8 threads. First call is ~550s for 96 points due to `Base.invokelatest` world-age overhead.

## Conventions

- `M = 1` everywhere. Table values are `ωM` (dimensionless).
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
  assembly.jl                — PDECoefficients → D̃ via spectral inner products (Galerkin)
  collocation.jl             — Collocation assembly + parse_h_term_map (fixed)
  rectangular_qep.jl         — SVD compression QEP solver (the winner)
  newton.jl                  — solve_qnm (returns J for perturbation theory)
  sgb_background.jl          — Parse H₁-H₄, ϑ from Mathematica + derivative evaluator
  sgb_linearize.jl           — sGB correction: compile + extract + build_sgb_correction_system
  sgb_perturbation.jl        — Eigenvalue perturbation solver (Eq. 111)
  sgb_galerkin.jl            — Numerical Galerkin assembly (abandoned — C[k,d] not polynomial)
  sgb_symbolic_pipeline.jl   — Exact SparsePoly K^(η=1) extraction (IN PROGRESS)
  [dtilde.jl, factored_assembly.jl] — earlier D̃ approaches (reference)
  [poly_extract.jl, zspace_extract.jl, pipeline.jl] — dead ends (kept for reference)
test/
  test_sgb_e2e.jl            — End-to-end sGB test (uses sgb_galerkin.jl)
  reproduce_paper.jl         — Generates Figs 1,2,5,6 from the GR paper
  reproduce_table1.jl        — Table I reproduction (219/220 digits)
reference/
  2406.11986_source/         — sGB paper supplementary materials (Mathematica notebook)
```

## Key function signatures

```julia
# GR pipeline
sys, nf = build_system_bespoke(a, N, m)      # → (METRICSSystem, norm_factors)
sys, nf = build_system_bespoke(a, N, m; d_max_override=20)  # for sGB compat
result = solve_qep_with_vectors(sys; ω₀=ω_L) # → (eigenvalues, eigenvectors)
J, free, pinned = compute_jacobian(sys, ω, v) # → Jacobian for perturbation theory

# sGB background
bg = sgb_background(a; verbose=true)         # → SGBBackground with H[1..4](r,χ)
hp = sgb_H_params(bg, r, χ)                  # → 24-element Float64 vector

# sGB correction (compiled evaluator)
csc = compile_sgb_correction(2; verbose=true) # → CompiledSGBCorrection (one-time)
C = extract_sgb_coefficients_complex(csc, r, χ, ω, a, hp)  # → 10×40 complex matrix
C0, C1, C2 = separate_sgb_omega(csc, r, χ, a, hp)  # → ω-separated coefficients

# D̃⁽¹⁾ assembly (per-a-order exact SparsePoly)
sys_corr, d_max = build_sgb_system_bespoke(a, N, m; epsilon=1e-14, verbose=true)
# H_i per-a-order loading
H_per_order, a_powers = load_H_ratpolys_per_order(; verbose=true)

# D̃⁽¹⁾ assembly (OLD — numerical Galerkin, abandoned: C[k,d] not polynomial)
sys_corr = build_sgb_galerkin(a, N, m, bg, ω0; csc=csc, N_z=20, N_χ=12)

# Perturbation solve
ω1 = solve_sgb_perturbation(sys_corr, ω0, v0, J)   # → ω⁽¹⁾ ∈ ℂ
```
