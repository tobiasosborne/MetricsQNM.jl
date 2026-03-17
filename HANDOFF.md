# Handoff Notes — MetricsQNM.jl

## Project Goal

Reproduce **Table I of arxiv:2312.08435** — fundamental (n=0, l=2, m=2) quasi-normal mode frequencies of Kerr black holes for spins a = 0.005 to 0.95 — using a 100% Julia implementation leveraging TensorGR.jl for symbolic tensor algebra.

## Current State (16/23 issues closed)

The full symbolic pipeline is operational: Kerr metric → linearized Einstein equations → compiled numerical evaluators → D̃ matrix assembly. The Leaver continued-fraction oracle works. The Newton-Raphson solver exists but doesn't yet converge to QNM frequencies.

### What Works

1. **Package structure** (`src/MetricsQNM.jl`): Clean module with Kerr background, spectral bases, Leaver solver, symbolic linearization, coefficient extraction, D̃ assembly, Newton solver.

2. **Kerr background** (`src/kerr.jl`): KerrParams, b, r_plus, r_minus, Omega_H, kappa, Sigma, Delta. All tested.

3. **Leaver QNM solver** (`src/leaver.jl`): Matches Table I to 1e-9 (low spin) and 1e-5 (high spin). Uses Cook-Zalutskiy (2014) recurrence coefficients with SpinWeightedSpheroidalHarmonics.jl for angular eigenvalues. **Critical convention**: the Julia SWSH package returns λ (Teukolsky constant), not A_slm; conversion: `A_slm = λ - c² + 2mc`.

4. **Spectral bases** (`src/spectral.jl`): ChebyshevBasis and LegendreBasis with derivative matrices, multiplication-by-power matrices, second-derivative elimination identities (Eq. 29). All tested.

5. **Symbolic linearization** (`src/linearize.jl`):
   - `kerr_symbolic_metric()` → SymbolicMetric with Kerr g_μν in (t, r, χ, φ) coords
   - `regge_wheeler_perturbation()` → 4×4 h_μν ansatz with 6 functions h₁...h₆
   - `linearized_christoffel()` → δΓ via Palatini identity with **mode derivatives** (∂_t→-iω, ∂_φ→im)
   - `linearized_ricci()` → δR_μν from background Γ and perturbed δΓ
   - `linearized_ricci_mixed()` → δR_μ^ν (the 10 field equations)
   - `compute_field_equations()` → full pipeline, returns 10 symbolic equations

6. **Compiled evaluator** (`src/coefficients.jl`):
   - `compile_field_equations(m_mode=2)` → CompiledFieldEquations (10 fast callables, 43μs/eval)
   - 40 h-derivative terms auto-discovered from symbolic equations
   - Complex arithmetic via **iu-polynomial trick**: evaluate at iu=0,1,-1 to extract iu⁰, iu¹, iu² coefficients, then reconstruct with iu→im (since im²=-1)
   - `extract_coefficients_complex()` → 10×40 coefficient matrix at any (r,χ,ω,a) point
   - ω-polynomial decomposition: C₀ + C₁ω + C₂ω² verified to 1e-16 at 4 complex ω values

7. **D̃ assembly** (`src/dtilde.jl`):
   - `build_Dtilde(cfe, a, ω, N, m)` → D̃(ω) via collocation, with/without A_k factor
   - `build_Dtilde_constant(cfe, a, N, m)` → D̃₀, D̃₁, D̃₂ constant matrices
   - Cross-check: collocation vs Galerkin agree to **1.5e-16** (machine precision)
   - RadialGrid, AngularGrid, AsymptoticGrid for precomputed basis values
   - ~2-14 seconds per D̃ build (N=5 to N=12), after JIT compilation

8. **Newton-Raphson** (`src/newton.jl`): `solve_qnm(sys, ω_guess)` with Moore-Penrose pseudoinverse, polar/axial-led normalization. Structure correct but not yet producing correct QNM frequencies.

9. **Spectral projection assembly** (`src/assembly.jl`): Full Galerkin machinery (PDECoefficients → D̃) implemented but not yet connected (needs G→K coefficient transform with A_k factorization).

### What Doesn't Work Yet (The Blocker)

**The Newton-Raphson solver diverges.** The root cause is the **asymptotic factor A_k**:

- **Without A_k**: The spectral expansion h_k = Σ v T_n P_l doesn't converge because h_k has singular behavior at the horizon (Δ=0) and infinity. The D̃ matrix is well-formed but doesn't capture the physics at moderate N (tested up to N=12).

- **With A_k**: h_k = A_k(r,ω) × Σ v T_n P_l. The A_k factor captures the singularities, making u_k bounded and spectrally convergent. BUT A_k depends on ω through exp(iωr), so D̃(ω) is NOT polynomial in ω — the D̃₀+D̃₁ω+D̃₂ω² decomposition breaks. Also, A_k grows as exp(iωr)×r^(2iω+2) at large r, creating huge matrix entries (~10⁶ at moderate grid points).

### The Path Forward

The **correct** implementation (matching the paper) requires:

1. **G coefficient extraction** (symbolic/numerical): Decompose each field equation into the polynomial form G_{k,γ,δ,σ,α,β,j} × ω^γ × r^δ × χ^σ × ∂_r^α ∂_χ^β h_j. The coefficient extraction infrastructure exists (`extract_coefficients_complex`), but the polynomial-in-(r,χ) decomposition after clearing Σ,Δ,(1-χ²) denominators is not yet implemented.

2. **r → z coordinate transform**: Substitute r = 2r₊/(1+z) into the G coefficients to get K coefficients. The chain rule and polynomial re-expression machinery is in `src/perturbation_ansatz.jl` but not yet applied to the coefficients.

3. **A_k factorization of the K equations**: Substitute h_k = A_k u_k into the K equations, compute derivatives via product rule, divide out A_k and common factors. This produces equations for the BOUNDED u_k functions. After this step, the equations are polynomial in (ω, z, χ) and the D̃₀+D̃₁ω+D̃₂ω² decomposition is valid.

4. **Spectral projection** (Galerkin): Use the assembly.jl machinery to project K coefficients onto the Chebyshev-Legendre basis. This produces the final D̃₀, D̃₁, D̃₂.

**Alternative faster path**: Instead of the full symbolic G→K pipeline, use the compiled evaluator numerically:
- At each (r,χ) grid point, extract the 10×40 coefficient matrix C(r,χ,ω,a)
- Clear denominators (multiply by Σ^p × Δ^q × (1-χ²)^s)
- Fit the result as a polynomial in (r,χ) using interpolation at many (r,χ) points
- Transform r→z, apply A_k factorization numerically
- This avoids symbolic polynomial decomposition entirely

### Key Technical Details

**Convention**: M=1 throughout. The paper's Table I values are in Mω units.

**SpinWeightedSpheroidalHarmonics.jl convention**: Returns Teukolsky λ, not Leaver A_slm. Convert: `A_slm = λ - c² + 2mc` where `c = aω`.

**iu-polynomial trick**: The compiled functions take real arguments (ω_re, ω_im, iu_sym). To get complex results, evaluate at iu=0, iu=1, iu=-1, extract polynomial coefficients in iu, reconstruct with iu→im. See `extract_coefficients_complex()` in `src/galerkin.jl`.

**TensorGR.jl**: Located at `../TensorGR.jl`, actively developed by other agents. DO NOT edit. The Symbolics extension provides `symbolic_metric()`, `symbolic_christoffel()`, `sym_deriv()`. Full Riemann/Ricci computation crashes (Symbolics.jl DivideError on large Kerr expressions) — we compute δΓ and δR via our own component-level code instead.

**Compilation**: First `compile_field_equations()` call takes ~97 seconds (symbolic pipeline + Symbolics.build_function). Subsequent `build_Dtilde` calls take 2-30 seconds depending on N. The compiled functions are ~43μs per evaluation.

### File Map

```
src/
  MetricsQNM.jl     — module root, includes & exports
  kerr.jl            — Kerr background: KerrParams, Σ, Δ, r±, Ω_H, κ
  perturbation_ansatz.jl — A_k(r), ρ_H, ρ_∞, z↔r coordinate transforms
  spectral.jl        — ChebyshevBasis, LegendreBasis, SpectralBasis, nl_index
  leaver.jl          — Leaver CF QNM solver (Cook-Zalutskiy recurrence)
  linearize.jl       — Symbolic: Kerr metric → δΓ → δR → 10 field equations
  coefficients.jl    — compile_field_equations, CompiledFieldEquations
  galerkin.jl        — extract_coefficients_complex, separate_omega_dependence
  collocation.jl     — (scaffolding, superseded by dtilde.jl)
  assembly.jl        — Galerkin D̃ from PDECoefficients (not yet connected)
  dtilde.jl          — build_Dtilde (collocation), build_Dtilde_constant
  newton.jl          — solve_qnm (Newton-Raphson with Moore-Penrose pinv)
test/
  runtests.jl        — test harness
  test_kerr.jl       — Kerr background tests (118 tests)
  test_spectral.jl   — Chebyshev + Legendre basis tests (477 tests)
  test_leaver.jl     — Leaver solver vs Table I (10 tests)
```

### Issue Tracker (bd)

16/23 issues closed. Run `bd ready` to see unblocked work. Run `bd list` for full status. Run `bd graph --all` for dependency visualization. The critical path is:

```
MetricsQNM-bdm (Assemble D̃) → MetricsQNM-nv0 (Newton solver) → MetricsQNM-uf0 (Table I)
```

### Test Suite

All 605+ tests pass. Run `julia --project=. -e 'using Pkg; Pkg.test()'`.
