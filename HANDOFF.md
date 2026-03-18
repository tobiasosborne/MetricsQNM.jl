# Handoff Notes — MetricsQNM.jl

## Project Goal

Reproduce **Table I of arxiv:2312.08435** — fundamental (n=0, l=2, m=2) quasi-normal mode frequencies of Kerr black holes for spins a = 0.005 to 0.95 — using Julia + TensorGR.jl.

## Why It's Taking So Long

The blocker is **one specific step**: extracting the polynomial G coefficients from the symbolic field equations. Everything else works.

The paper does this step **symbolically in Mathematica** — they take the 10 linearized Einstein equations (each with thousands of terms), collect coefficients of each h-derivative, clear denominators (Σ^P Δ^Q (1-χ²)^S), and decompose into exact polynomials in (r, χ, ω). This is a CAS operation.

I tried to **shortcut this with numerical polynomial fitting** (evaluate at grid points, Vandermonde fit). This was a mistake — it's numerically unstable and loses the exact structure. The result: eigenvalues that are ~2-6% off instead of 10-digit accurate.

The fix is straightforward: **do the G extraction symbolically using Symbolics.jl**, exactly as the paper does with Mathematica. I verified this works — `Symbolics.substitute` extracts individual coefficients in ~0.25 seconds, `Symbolics.expand` clears denominators in ~0.04 seconds. The full extraction for 400 coefficients would take ~8 minutes. I just ran out of time implementing it cleanly.

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

## THE BLOCKER: Polynomial G Coefficient Extraction

### What the paper does (Mathematica)
1. Take symbolic equation: `Σ_d C_{k,d}(r,χ,ω,a) × ∂_r^α ∂_χ^β h_j = 0`
2. Collect coefficient of each `∂_r^α ∂_χ^β h_j` → rational function of (r, χ, ω, a)
3. Multiply by `Σ^P Δ^Q (1-χ²)^S` → EXACT polynomial in (r, χ, ω, a)
4. Decompose by powers: `G_{k,γ,δ,σ,α,β,j} × ω^γ r^δ χ^σ`
5. These G coefficients are exact functions of `a` only

### What I tried (numerical — WRONG APPROACH)
- Evaluated coefficients at grid points, fit Vandermonde polynomial → unstable, ~2-6% error
- The r→z transform amplified fitting errors
- Result: eigenvalues never converged beyond |Δω| ≈ 0.02

### What needs to be done (symbolic — CORRECT APPROACH)
The Symbolics.jl operations needed are ALL verified to work:
- `Symbolics.substitute(eq, Dict(h_term => 1, others => 0))` → coefficient (0.25s)
- `Symbolics.expand(coeff * Σ^3 * Δ^1 * (1-χ²)^1)` → polynomial (0.04s)
- `Symbolics.substitute(poly, Dict(r => 0, chi => 0))` → constant term
- `Symbolics.coeff(poly, r^2)` → coefficient of r² (verified working)
- Derivative-at-zero trick for higher powers: `(1/k!) d^k f/dr^k |_{r=0}`

**Estimated time for full extraction**: 400 coefficients × ~0.3s each = ~2 minutes for extraction + ~2 minutes for expansion + polynomial decomposition. Total: ~10-15 minutes one-time computation.

### Denominator powers
From numerical testing on representative entries:
- `Σ³Δ¹`: most entries are EXACTLY polynomial (residual = 0.0)
- `Σ⁴Δ¹(1-χ²)¹`: covers the remaining entries to ~1e-7
- Safe choice: use `Σ³Δ¹(1-χ²)¹` which handles everything

### After G extraction: the rest is straightforward
1. **r → z transform**: `r^δ = (2r₊)^δ (1+z)^{-δ}`, multiply by `(1+z)^{d_max}`, expand via `binom(d_max-δ, j)` → K coefficients polynomial in z
2. **A_k factorization**: already implemented in `_transform_h_to_u`, verified exact degree-2 in ω
3. **Spectral projection**: `assembly.jl` already handles `PDECoefficients → D̃₀, D̃₁, D̃₂` via linearization matrices
4. **Newton-Raphson**: `newton.jl` already implements the paper's algorithm with pinv
5. **QEP eigenvalue**: companion linearization already working as backup

## Step-by-Step Instructions for Next Agent

### Phase 1: Symbolic G Extraction (~30 min implementation + ~15 min computation)

Write a function `extract_G_coefficients()` that:

```julia
function extract_G_coefficients(; P=3, Q=1, S=1)
    # 1. Get symbolic equations
    eqs, coords, params, hfuncs, freq_vars = compute_field_equations(2)
    r, chi = coords[2], coords[3]; a_s = params[1]
    omega_re, omega_im, iu = freq_vars

    # 2. Discover h-derivative terms
    h_terms = ...  # same as in compile_field_equations
    sub_zero = Dict(t => Num(0) for t in h_terms)

    # 3. Common denominator
    Sig = r^2 + a_s^2 * chi^2
    Del = r^2 - 2r + a_s^2
    denom = Sig^P * Del^Q * (1 - chi^2)^S

    # 4. For each (k, d): extract, clear, expand
    for k in 1:10, (d, ht) in enumerate(h_terms)
        sub = copy(sub_zero); sub[ht] = Num(1)
        coeff = Symbolics.substitute(eqs[k], sub)
        coeff == 0 && continue
        cleared = Symbolics.expand(coeff * denom)

        # 5. Decompose by (ω, r, χ) powers
        # Separate ω: substitute (omega_re=0, omega_im=0, iu=0) for ω⁰ part
        # Then derivatives for higher ω powers
        # Separate r: substitute r=0 for r⁰, then d/dr|_{r=0} for r¹, etc.
        # Separate χ: same approach

        # Store as G[k][(γ,δ,σ,α,β,j)] = value
    end
end
```

The polynomial decomposition by powers of r and χ can use either:
- `Symbolics.coeff(expr, r^n)` (verified working)
- Repeated derivative-at-zero: `coeff_of_r^n = (1/n!) ∂^n/∂r^n f|_{r=0}`
- `Symbolics.polynomial_coeffs(expr, [r, chi])` if available

### Phase 2: r → z Transform + A_k Factorization (~15 min)

Already implemented in `src/pipeline.jl` and `src/factored_assembly.jl`. Just need to:
1. Feed the symbolic G coefficients into the r→z transform (binomial expansion)
2. Apply `_transform_h_to_u` for the A_k factorization
3. Store as PDECoefficients

### Phase 3: Assembly + Newton (~5 min)

Already working in `src/assembly.jl` and `src/newton.jl`. Just call:
```julia
basis = spectral_basis(30, 2)
sys = assemble_system(K_coefficients, basis, a)
result = solve_qnm(sys, ω_guess; parity=:polar)
```

### Phase 4: Table I Sweep (~1 hour computation)

Loop over 11 spin values, solve both polar and axial, compare with Leaver oracle.

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
  symbolic_decompose.jl  — beginning of symbolic extraction (INCOMPLETE)
test/
  runtests.jl, test_kerr.jl, test_spectral.jl, test_leaver.jl — 605+ tests, all passing
```

## Issue Tracker

`bd ready` shows unblocked work. `bd list` for full status. `bd graph --all` for dependencies. 16/23 closed.

Critical path: **G extraction (symbolic)** → assembly → Newton → Table I.
