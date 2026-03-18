# Handoff Notes — MetricsQNM.jl

## Project Goal

Reproduce **Table I of arxiv:2312.08435** — fundamental (n=0, l=2, m=2) quasi-normal mode frequencies of Kerr black holes for spins a = 0.005 to 0.95 — using Julia + TensorGR.jl.

## Current Status (2026-03-18, session 3)

**SVD compression QEP solver added. Symbolic G extraction runs but has a denominator-clearing bug.**

### What happened in this session
1. **SVD compression QEP solver written** — `src/rectangular_qep.jl` (149 lines), adapted from `af-tests/examples13` benchmarks. SVD compression + companion QZ for rectangular QEPs. **Verified**: 72/72 synthetic eigenvalues recovered at 5.4e-14 accuracy.
2. **`reproduce_table1()` rewritten** in `newton.jl` to use the full pipeline: `build_system_symbolic` → `solve_qep_svd` → `solve_qep_newton`.
3. **Tree-walker `_walk_expanded_poly` wired up** (replacing slow `polynomial_coeffs`). Extraction runs at 104 tasks/s (3.8s total for 400 tasks) instead of 0.9 tasks/s.
4. **BLOCKER: Denominator clearing is insufficient.** The field equations contain rational subexpressions (Christoffel symbols with Σ, Δ denominators). `Symbolics.expand(coeff * denom)` does NOT distribute `denom` through rational subterms. Result: hundreds of `BasicSymbolicImpl{SymReal}` warnings = terms silently dropped. K₂ = 0 terms (no ω² contributions), making the QEP degenerate.

### The denominator bug in detail
- Field equations δR^μ_ν are **rational** in (r, χ) due to Christoffel symbol denominators (Σ², ΣΔ, etc.)
- Extracting coefficient of h_i via substitution gives a rational function, not a polynomial
- Multiplying by Σ^P Δ^Q (1-χ²)^S and calling `Symbolics.expand` doesn't cancel inner denominators
- The "problematic terms" are full rational expressions like `(poly_num * r^6) / (Σ^5 * Δ * ...)` sitting in the Add.dict of the "expanded" expression
- Both `polynomial_coeffs` and `_walk_expanded_poly` fail on these — they're not polynomials!

### Fix attempted (partially)
- Added `Symbolics.simplify_fractions(raw)` before `expand` — this should cancel the inner Σ/Δ factors against the clearing denom. **Not yet tested** — session ended before the run completed.

### Dead ends documented
- **Numerical polynomial fitting** (Vandermonde, `poly_extract.jl`): unstable, ~2-6% error
- **Numerical z-space extraction** (`zspace_extract.jl`): clearing factor destroys boundary conditions, caps at |Δω|≈0.01
- **P=3, Q=1, S=1 clearing alone**: insufficient for these rational field equations

## What Works (16/23 issues closed)

### 1. Symbolic Pipeline (COMPLETE)
- `src/linearize.jl`: Kerr metric → background Christoffel Γ → Regge-Wheeler perturbation h_μν → linearized Christoffel δΓ → linearized Ricci δR_μν → mixed form δR_μ^ν
- Mode derivatives: ∂_t → -iω, ∂_φ → im properly implemented via symbolic (ω_re, ω_im, iu_sym) split
- Output: 10 symbolic field equations, 20K-60K characters each, involving 40 h-derivative terms
- `compute_field_equations(2)` runs the full pipeline

### 2. Compiled Evaluator (COMPLETE)
- `src/coefficients.jl`: `compile_field_equations(m_mode=2)` compiles the 10 equations into fast Julia callables via `Symbolics.build_function`
- 43μs per equation evaluation
- Complex arithmetic via **iu-polynomial trick**: evaluate at iu=0, iu=1, iu=-1, extract coefficients a₀, a₁, a₂ of iu-polynomial, reconstruct with iu→im since im²=-1: `result = (a₀ - a₂) + im·a₁`
- The 40 h-derivative terms are auto-discovered from the symbolic equations via `Symbolics.get_variables`

### 3. Coefficient Extraction (COMPLETE)
- `src/galerkin.jl`: `extract_coefficients_complex(cfe, r, χ, ω, a)` returns 10×40 coefficient matrix at any point
- `separate_omega_dependence(cfe, r, χ, a)` returns (C₀, C₁, C₂) where C(ω) = C₀ + C₁ω + C₂ω²
- **Verified to 1e-16** at 4 complex ω values

### 4. A_k Factorization (COMPLETE)
- `src/factored_assembly.jl`: `_transform_h_to_u` converts h-derivative coefficients to u-derivative coefficients
- Uses `_Ak_ratios(j, r, ω, params)` → [1, dlogA/dr, d²logA/dr² + (dlogA/dr)²]
- Product rule: ∂_r^α(A u) = Σ binom(α,a)(∂^a A)(∂^{α-a} u), divide by A
- **Verified**: factored coefficients are EXACTLY degree-2 polynomial in ω (tested at 4 ω values, error < 1e-16)

### 5. Leaver Oracle (COMPLETE)
- `src/leaver.jl`: Matches Table I to 1e-9 (low spin) / 1e-5 (high spin)
- Cook-Zalutskiy (2014) recurrence, SpinWeightedSpheroidalHarmonics.jl angular eigenvalue
- **Critical**: Julia SWSH package returns λ (Teukolsky constant), not A_slm. Convert: `A_slm = λ - c² + 2mc`

### 6. Spectral Bases (COMPLETE)
- `src/spectral.jl`: ChebyshevBasis (derivative, z-multiplication matrices), LegendreBasis (derivative, χ-multiplication)
- Second-derivative elimination identities (Eq. 29)
- `src/assembly.jl`: full Galerkin assembly from PDECoefficients → D̃ matrices

### 7. D̃ Assembly (PARTIALLY WORKING)
- `src/dtilde.jl`: RadialGrid, AngularGrid (with Gauss-Legendre nodes), AsymptoticGrid
- `build_Dtilde`: collocation with/without A_k
- `build_factored_system`: Galerkin with A_k factorization + quadrature
- **Cross-check**: collocation vs Galerkin agree to 1.5e-16
- QEP companion linearization eigenvalue solver: finds approximate QNMs at |Δω| ≈ 0.02-0.06

### 8. SVD Compression QEP Solver (NEW, VERIFIED)
- `src/rectangular_qep.jl`: SVD compression + companion QZ for rectangular QEPs
- `solve_qep_svd(D0, D1, D2; ω₀, refine)`: finds ALL eigenvalues
- `solve_qep_newton(D0, D1, D2, ω; max_iter)`: refines single eigenvalue via σ_min Newton
- `qep_residual`, `validate_qep` for diagnostics
- **Verified on synthetic 60×36 rectangular QEP**: 72/72 eigenvalues at 5.4e-14

### 9. Newton-Raphson (STRUCTURE COMPLETE, NOT CONVERGING)
- `src/newton.jl`: standard overdetermined pinv Newton + updated `reproduce_table1()`
- `src/solve.jl`: per-step D̃ building variant
- Diverges because D̃ matrices are not accurate enough (see blocker)

## THE BLOCKER: Symbolic G Coefficient Extraction

### Current pipeline (`src/symbolic_pipeline.jl`)
1. `compute_field_equations(2)` → 10 symbolic equations with ~1870 terms each
2. For each (k, d): substitute h-terms to extract individual coefficient (~0.25s)
3. Multiply by `Σ^P Δ^Q (1-χ²)^S`, call `simplify_fractions`, then `expand`
4. Extract all monomials via `_walk_expanded_poly` (fast tree walker, 104 tasks/s)
5. Decompose ω via the iu→i trick → G₀, G₁, G₂
6. Transform r → z via binomial expansion
7. Feed to `assembly.jl` → D̃₀, D̃₁, D̃₂

### The problem
Step 3 currently uses `Symbolics.simplify_fractions` to cancel inner Σ/Δ denominators. **This has not been tested yet.** If `simplify_fractions` is too slow or doesn't fully clear, alternatives:
- Increase P, Q, S (try P=6, Q=3, S=2) — but `expand` alone won't distribute through divisions
- Use `Symbolics.simplify_fractions` + `expand` (current approach, untested)
- Switch strategy: use the **compiled evaluator** (galerkin.jl, 43μs/eval) + numerical Galerkin quadrature instead of symbolic extraction

### Alternative path: numerical Galerkin with compiled evaluator
The compiled evaluator (`coefficients.jl`) + `separate_omega_dependence` already work perfectly (verified to 1e-16). The existing `build_factored_system` in `factored_assembly.jl` uses this path with Gauss-Legendre quadrature. This numerical approach avoids the symbolic denominator-clearing issue entirely. The question is whether quadrature accuracy is sufficient for Newton convergence.

## Next Steps

1. **Test `simplify_fractions` fix**: `julia --project=. --threads=8 test/test_svd_qep.jl`
   - Check: are warnings gone? Is K₂ nonzero? Do QEP eigenvalues converge?
2. **If `simplify_fractions` is too slow**: try the compiled evaluator + numerical Galerkin path
3. **If either works**: Newton-Raphson refinement → Table I sweep

## Key Technical Details

### Convention: M = 1
All formulas use M = 1. The paper's Table I values are ωM (dimensionless).

### SpinWeightedSpheroidalHarmonics.jl
Returns Teukolsky λ, NOT Leaver A_slm. Convert: `A_slm = λ - c² + 2mc` where `c = aω`.

### iu-polynomial trick
Compiled functions take real args `(r, χ, ω_re, ω_im, iu_sym, a, p₁...p₄₀)`. To get complex results: evaluate at iu=0, 1, -1, extract polynomial in iu, substitute iu→im. See `extract_coefficients_complex()` in galerkin.jl.

### TensorGR.jl
At `../TensorGR.jl`, actively developed. DO NOT edit. Use `symbolic_metric()`, `symbolic_christoffel()`, `sym_deriv()` from the Symbolics extension.

### Compilation time
First `compile_field_equations()` call: ~97 seconds. Subsequent D̃ builds: 2-60 seconds depending on N. Compiled function evaluation: ~43μs.

### Threading
Use `--threads=8` for Julia. More threads (e.g. 64) cause OpenBLAS contention and 37% slowdown.

## File Map

```
src/
  MetricsQNM.jl          — module root
  kerr.jl                — KerrParams, Σ, Δ, r±, Ω_H, κ
  perturbation_ansatz.jl — A_k(r), ρ_H, ρ_∞, z↔r transforms
  spectral.jl            — ChebyshevBasis, LegendreBasis, nl_index
  leaver.jl              — Leaver CF QNM solver
  linearize.jl           — Symbolic: Kerr → δΓ → δR → 10 field equations
  coefficients.jl        — compile_field_equations, CompiledFieldEquations
  galerkin.jl            — extract_coefficients_complex, separate_omega_dependence
  factored_assembly.jl   — _transform_h_to_u (A_k factorization), Galerkin D̃
  assembly.jl            — PDECoefficients → D̃ via spectral inner products
  dtilde.jl              — RadialGrid, AngularGrid, collocation D̃
  pipeline.jl            — numerical pipeline (FLAWED — use symbolic instead)
  poly_extract.jl        — numerical polynomial extraction (FLAWED)
  collocation.jl         — scaffolding
  newton.jl              — solve_qnm + reproduce_table1 (uses SVD QEP)
  solve.jl               — solve_qnm_direct (per-step D̃ variant)
  symbolic_decompose.jl  — early symbolic extraction scaffold (superseded)
  zspace_extract.jl      — numerical z-space extraction (DEAD END)
  symbolic_pipeline.jl   — symbolic G extraction + tree walker + r→z transform
  rectangular_qep.jl     — SVD compression QEP solver (NEW, VERIFIED)
test/
  runtests.jl, test_kerr.jl, test_spectral.jl, test_leaver.jl — 605+ tests, all passing
  test_symbolic_pipeline.jl — full pipeline test with SVD convergence + QEP
  test_svd_qep.jl           — SVD QEP end-to-end test (NEW)
  test_zspace.jl            — z-space tests (dead end)
```

## Issue Tracker

`bd ready` shows unblocked work. `bd list` for full status. `bd graph --all` for dependencies. 16/23 closed.

Critical path: **Fix denominator clearing** → G extraction → assembly → QEP → Newton → Table I.
