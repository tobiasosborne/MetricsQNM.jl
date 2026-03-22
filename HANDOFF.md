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
| **D̃⁽¹⁾ assembly (exact SparsePoly)** | **BLOCKED** | `sgb_symbolic_pipeline.jl` — d_max too high, see below |
| **End-to-end test** | **FAILS** | `test/convergence_sgb.jl` — ω₁ = O(10^10), not converging |

## THE BLOCKING PROBLEM: d_max from c_{k,d,p} (2026-03-22)

### Summary

The per-a-order pipeline runs end-to-end, verification is now at machine precision
(2.24e-14), but **ω₁ does not converge because the shared d_max is too high for the
spectral basis to resolve**. The d_max problem is NOT from H_i (fixed by per-a-order
splitting) — it's from the c_{k,d,p} coefficients themselves.

### Why d_max is high

The shared d_max has TWO contributions:
1. **c_{k,d,p}**: r-degree ~15-24 from Kerr metric components (Σ, Δ, etc.) at numerical a
2. **H_p per order**: r-degree ~2k+5 (low per order)

The COMBINED d_max = c_contribution + H_contribution ≈ 24 + (2k+5).

Per-a-order H_i splitting reduced H_p's contribution from den.t=66 (all orders summed)
to den.t=9-11 (individual orders). But c_{k,d,p}'s contribution of ~24 is FIXED because
it is evaluated at numerical a=0.3 in Phase 2. The base d_max=24 is the floor, not ~9.

### N-convergence results (2026-03-22, a=0.3, epsilon=1e-2, Source 1 only)

With epsilon=1e-2: 2 a-orders retained (a^0, a^2), shared d_max=26.

```
N     |Δω⁰|      ω₁ Re          ω₁ Im           |ω₁|       cond(J)
10    6.74e-05   -5.84e+09      -1.17e+11        1.17e+11   3.6e+14
12    1.54e-03   +7.93e+10      -1.83e+12        1.83e+12   3.4e+15
15    3.54e-02   -2.47e+11      +3.61e+11        4.38e+11   5.2e+16
18    7.40e-03   -2.53e+10      +3.82e+10        4.58e+10   5.5e+17
```

**Diagnosis**: The GR eigenvalue error |Δω⁰| is NOT decreasing monotonically — the
d_max=26 shared basis makes the QEP ill-conditioned. The condition number of J grows
from 3.6e+14 (N=10) to 5.5e+17 (N=18). ω₁ is O(10^10-12) (garbage). The N=20-25
results were still computing but the trend is catastrophic.

### Why the paper doesn't have this problem

The paper computes K^(η=1) **fully symbolically** — keeping `a` symbolic throughout
the entire pipeline (not just in H_i). This means:
1. c_{k,d,p} never has numerical a baked in
2. The full product c_{k,d,p} × H_p stays rational in (r, χ, a)
3. After a-order decomposition, each order has moderate r-degree (~2k+5)
4. The c_{k,d,p} contribution to d_max is ZERO because it factors through the a-expansion

Our pipeline evaluates c_{k,d,p} at numerical a=0.3 in Phase 2 (via probing). This
bakes the Kerr metric's r-structure into the coefficients as ~15-24 powers of r that
CANNOT be decomposed per-a-order afterward.

### What needs to change

**The c_{k,d,p} must also be decomposed per-a-order.** Two approaches:

**Approach 1: Symbolic-a probing**
Keep `a` symbolic in Phase 2 probing. This requires `compute_sgb_correction_equations`
to accept symbolic `a` and `extract_sgb_coefficients_symbolic` to return RatPolys in
(r, χ, a, ω). Then split c_{k,d,p} by a-exponent just like H_i. This would reduce the
combined d_max to ~(2k+5) per order (same as H_i alone).

Challenge: The Symbolics.jl expression tree may blow up when `a` is symbolic, since
the field equations contain Σ⁻¹, Δ⁻¹ etc. with symbolic a.

**Approach 2: Per-a-order c_{k,d,p} by multi-point probing**
Evaluate c_{k,d,p} at multiple a-values (e.g., a=0.1, 0.2, ..., 0.9) and fit the
a-dependence as a polynomial in a². This gives per-a-order c_{k,d,p} without symbolic a.

Challenge: Requires enough a-points to resolve all a-orders, and the fitting may be
ill-conditioned for high a-orders.

**Approach 3: Rethink the z-transform**
The (1-z)^{d_max} multiplication is the root cause of the resolution requirement.
An alternative coordinate transform or rational spectral basis could absorb the
r-denominator into the weight function, eliminating the need for (1-z)^{d_max}.

## What was fixed this session (2026-03-22)

### 1.5% verification error — FIXED

**Root cause**: `cleanup!(rp.num; tol=1e-15, relative=true)` on H4's numerator.

LCD clearing to r^66 inflated H4's max coefficient to **5.94e+13**. The cleanup
threshold became `1e-15 × 5.94e+13 = 0.059`, which dropped **8 legitimate terms**
with coefficients 0.012–0.058. These were real polynomial terms, not noise.

**Diagnosis method** (in `test/diagnose_per_order.jl`, `test/diagnose_H4_terms.jl`,
`test/diagnose_H4_cleanup.jl`):
1. Stage 1: `symexpr_to_poly` with symbolic vs numerical a → error only in **H4**
2. Stage 2: `_split_ratpoly_by_var` → exact (1e-16), not the culprit
3. Term-by-term comparison: all 967 collapsed monomials match to 1e-16 **before cleanup**
4. After cleanup: 8 terms dropped from H4, causing the entire 7.26e-4 error
5. Error only in χ-independent terms (H4_χ perfect, confirming dropped terms had χ^0)

**Fix**: Changed all sGB `cleanup!` calls from `relative=true` to absolute (`tol=1e-15`):
- `sgb_background.jl`: lines 529, 541, 588, 599, 729, 752, 761
- `sgb_symbolic_pipeline.jl`: line 155

**Verification**: max relative error dropped from 1.5e-2 to **2.24e-14** at all test points.

## Exact SparsePoly extraction — STATUS

**Path B runs end-to-end but does not converge due to d_max.** First ω₁ computed 2026-03-21.

### What's done:
1. **DenomSig extended** with 4th field `t` for r-power tracking (backward-compat, GR still 11.9 digits at N=8)
2. **SparsePoly differentiation** — exact `differentiate(p, var_idx)` function
3. **H_i → 24 RatPoly** — `load_H_ratpolys(a)` parses Mathematica notebook, converts to SparsePoly, differentiates symbolically. Values match numerical to ~1e-16, derivatives are now EXACT (vs O(ε²) finite diff before)
4. **c_{k,d,p} probing** — `extract_sgb_coefficients_symbolic(a)` double-probes sGB equations: 203 non-zero (k,d) pairs, 1165 non-zero (k,d,p) triples (~95s)
5. **Multiply + accumulate** — `combine_sgb_K` multiplies c_{k,d,p} × H_p in RatPoly land, clears denominators, ω-decomposes. 203 pairs: ~20K-26K terms per a-order
6. **Full pipeline** — `build_sgb_system_bespoke(a, N, m)` wires everything through _r_to_z + assemble_system
7. **Per-a-order H_i loading** — `load_H_ratpolys_per_order()` with exact verification (2.24e-14)
8. **Convergence test** — `test/convergence_sgb.jl` sweeps N with cached Phase 2-3

### Per-a-order pipeline timing (a=0.3, epsilon=1e-2):
- Phase 1 (H_i loading): ~25s
- Phase 2 (c_{k,d,p} probing): ~95s
- Phase 3 (combine, 2 a-orders): ~62s
- Phase 4-5 per N (r→z + assembly + GR + perturbation): 53s (N=10) to 1020s (N=18)

### Critical findings:
- d_max = 24 even at a^0 order (from c_{k,d,p}, not H_p)
- Shared d_max = 26 with 2 a-orders requires N ≥ 26 for spectral resolution
- GR eigenvalue with d_max_override=26 does NOT converge: erratic |Δω⁰| and cond(J) ∝ N^5
- The per-a-order approach solved the H_i d_max inflation (66→9) but NOT the c_{k,d,p} inflation (24→24)

## What the paper actually does (deep review 2026-03-20)

A thorough review of arxiv:2406.11986 revealed the paper uses **the exact same Galerkin assembly** for D̃⁽¹⁾ as for D̃⁽⁰⁾:

1. Symbolically linearize the FULL sGB field equations about g = g_Kerr + ζH
2. The H_i(r,χ) are rational functions → after clearing, everything is polynomial in r, χ
3. Extract K^(η=1) coefficients (numbers, not functions!) via symbolic pipeline
4. Assemble via same spectral inner products as GR

The K^(η=1) coefficients are just numbers because the entire symbolic pipeline stays rational. This is NOT possible in our code because inserting H_i content into Symbolics.jl causes 100-155M char expression blowup on WSL2.

### Critical divergences from paper

1. **Assembly method**: Paper uses symbolic K-extraction; we use numerical evaluation
2. **a-handling**: Paper keeps a symbolic throughout; we evaluate a numerically in c_{k,d,p}
3. **Missing sources**: We only have Source 1 (modified Ricci from H). Paper needs all 4:
   - Source 1: O(ζ) Ricci correction from modified background ✓
   - Source 2: linearized sGB source tensor A_μ^ν ✗
   - Source 3: linearized scalar stress-energy T_μ^ν ✗
   - Source 4: Ω_H¹, κ¹ corrections in A_k ✗
4. **System size**: Paper is 11 eqs × 7 unknowns (includes scalar h₇); we are 10×6

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
  sgb_symbolic_pipeline.jl   — Exact SparsePoly K^(η=1) extraction (BLOCKED on d_max)
  [dtilde.jl, factored_assembly.jl] — earlier D̃ approaches (reference)
  [poly_extract.jl, zspace_extract.jl, pipeline.jl] — dead ends (kept for reference)
test/
  test_sgb_e2e.jl            — End-to-end sGB test (uses sgb_galerkin.jl)
  convergence_sgb.jl         — N-convergence study (Source 1, per-a-order)
  diagnose_per_order.jl      — Multi-stage diagnostic for per-order splitting error
  diagnose_H4_terms.jl       — Term-by-term H4 comparison (numerical vs symbolic a)
  diagnose_H4_cleanup.jl     — Identifies cleanup!-dropped terms causing 1.5% error
  reproduce_paper.jl         — Generates Figs 1,2,5,6 from the GR paper
  reproduce_table1.jl        — Table I reproduction (219/220 digits)
  run_sgb_bespoke.jl         — Lean sGB runner script
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
