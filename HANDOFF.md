# Handoff Notes — MetricsQNM.jl

## Project Goal

Reproduce **Table I of arxiv:2312.08435** — fundamental (n=0, l=2, m=2) quasi-normal mode frequencies of Kerr black holes for spins a = 0.005 to 0.95 — using Julia + TensorGR.jl.

## Current Status (2026-03-18)

**Symbolic G extraction pipeline is WRITTEN but UNTESTED.** The previous session crashed while waiting for the first full pipeline run to complete.

### What happened in the crashed session
1. **Numerical z-space extraction tried and abandoned** — `src/zspace_extract.jl` achieves |Δω|≈0.01 but stalls at N=12 (non-monotonic). Root cause: the (1+z)^{d_z} clearing factor destroys the outgoing-wave boundary condition at z=-1.
2. **Symbolic pipeline written** — `src/symbolic_pipeline.jl` (392 lines) implements the full pipeline: substitute h-terms → clear denominators → expand → extract monomials → ω-decompose → r→z transform → assemble D̃.
3. **Symbolics.jl performance researched deeply:**
   - Direct `Add.dict`/`Mul.dict` tree walking: ~110ms per 1870-term polynomial (fastest)
   - `Symbolics.polynomial_coeffs`: ~640ms (what the code currently uses)
   - `Symbolics.coeff` per monomial: ~22.5s (unusable)
4. **`_walk_expanded_poly` is written** in symbolic_pipeline.jl (lines 37-143) but `extract_G_exact` currently uses the slower `polynomial_coeffs` API instead. Switch to tree walking if speed is needed.
5. **Session crashed** while waiting for `julia --threads=16 test/test_symbolic_pipeline.jl` to complete.

### Dead ends documented
- **Numerical polynomial fitting** (Vandermonde, `poly_extract.jl`): unstable, ~2-6% error
- **Numerical z-space extraction** (`zspace_extract.jl`): clearing factor destroys boundary conditions, caps at |Δω|≈0.01

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

### 8. Newton-Raphson (STRUCTURE COMPLETE, NOT CONVERGING)
- `src/newton.jl`: standard overdetermined pinv Newton
- `src/solve.jl`: per-step D̃ building variant
- Diverges because D̃ matrices are not accurate enough (see blocker below)

## THE PIPELINE: Symbolic G Coefficient Extraction

### What the code does (`src/symbolic_pipeline.jl`)
1. `compute_field_equations(2)` → 10 symbolic equations with ~1870 terms each
2. Substitute h-terms to extract individual coefficients (~0.25s each)
3. Multiply by `Σ^P Δ^Q (1-χ²)^S` denominator, expand → exact polynomial
4. Substitute `a_s = a` (numeric)
5. Extract all monomials via `polynomial_coeffs` (or tree walking)
6. Decompose ω via the iu→i trick → G₀, G₁, G₂
7. Transform r → z via binomial expansion
8. Feed to `assembly.jl` → D̃₀, D̃₁, D̃₂

### Denominator powers
- `Σ³Δ¹`: most entries are exactly polynomial
- Safe choice: `P=3, Q=1, S=1` handles everything

### After G extraction: the rest is straightforward
1. **r → z transform**: `_r_to_z()` in symbolic_pipeline.jl, uses binomial expansion
2. **Spectral projection**: `assembly.jl` already handles `PDECoefficients → D̃₀, D̃₁, D̃₂`
3. **QEP eigenvalue**: companion linearization in `test_symbolic_pipeline.jl`
4. **Newton-Raphson**: `newton.jl` implements the paper's algorithm with pinv

## Next Steps

1. **Run the test**: `julia --project=. --threads=auto test/test_symbolic_pipeline.jl`
2. **Check convergence**: SVD study should show monotonic improvement with N
3. **If too slow**: switch `extract_G_exact` from `polynomial_coeffs` to `_walk_expanded_poly` (already written)
4. **If results correct**: Newton-Raphson refinement → Table I sweep

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
  newton.jl              — solve_qnm (Newton-Raphson with pinv)
  solve.jl               — solve_qnm_direct (per-step D̃ variant)
  symbolic_decompose.jl  — early symbolic extraction scaffold (superseded)
  zspace_extract.jl      — numerical z-space extraction (DEAD END)
  symbolic_pipeline.jl   — symbolic G extraction pipeline (UNTESTED — run this next)
test/
  runtests.jl, test_kerr.jl, test_spectral.jl, test_leaver.jl — 605+ tests, all passing
  test_symbolic_pipeline.jl — full pipeline test with SVD convergence + QEP
  test_zspace.jl            — z-space tests (dead end)
```

## Issue Tracker

`bd ready` shows unblocked work. `bd list` for full status. `bd graph --all` for dependencies. 16/23 closed.

Critical path: **G extraction (symbolic)** → assembly → Newton → Table I.
