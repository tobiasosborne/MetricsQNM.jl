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
| **D̃⁽¹⁾ assembly** | **BLOCKED** | Collocation approach implemented but numerically unstable — see below |
| **End-to-end test** | **BLOCKED** | `test/test_sgb_e2e.jl` runs but gives wrong answer due to assembly issue |

## THE BLOCKING PROBLEM: Asymptotic factor blowup in collocation

### What was attempted

`build_sgb_correction_system` in `sgb_linearize.jl` builds D̃⁽¹⁾(ω₀) via:
1. At each Chebyshev collocation point (z_i, χ_j), evaluate correction coefficients C (10×40 complex, via compiled evaluator with iu trick)
2. For each spectral test function (j,n,l), compute h_j = A_j(r,ω₀) × T_n(z) × P_l^m(χ) and all derivatives
3. Accumulate D_coll[row, col] = C · hvals
4. Vandermonde transform V⁻¹ to convert collocation rows → Galerkin spectral basis

### What happens

The test function h_j = A_j × u_j involves the asymptotic factor:
```
A_j = exp(iωr) × r^(2iω + ρ∞) × ((r - r₊)/r)^σ₊
```
At the outer collocation nodes (z → -1, r → ∞):
- |A_j| ~ exp(Im(iω) × r) = exp(0.0877 × r)
- At z = -0.9999 (r ≈ 39000): exp(3427) → **Inf** → NaN
- At z = -0.99 (r ≈ 390): exp(34) ≈ 7e14 → finite but D₀ norm ≈ 1e25

Even with z clamped to -0.99, the matrix entries are O(1e25) and the perturbation result is garbage (ω⁽¹⁾ ≈ 6e23 instead of ~0.36).

### Why the GR Galerkin assembly doesn't have this problem

The GR `assemble_system` in `assembly.jl` works entirely in spectral coefficient space. The A_j factor is absorbed into the symbolic G-coefficient extraction pipeline — the spectral operators (z^δ ∂_z^α) act on Chebyshev/Legendre coefficients algebraically, never evaluating A_j numerically at any point. So no exponential blowup occurs.

### The correct fix: Galerkin assembly with interpolated coefficients

**Do NOT use collocation.** Instead, adapt the Galerkin assembly for the sGB correction:

1. **Evaluate the correction PDE coefficients on a grid.** At each (r_i, χ_j), call `extract_sgb_coefficients_complex(csc, r, χ, ω₀, a, hp)` to get C[k,d] — the coefficient of h-derivative d in equation k. These are O(1) everywhere (no A_j involved).

2. **Fit each coefficient as a polynomial in z and χ.** For each (k, d) pair, C_k,d(z, χ) is a smooth function (depends on H_i which are rational). Interpolate on a Chebyshev grid to get polynomial coefficients: C̃_k,d = Σ c_{δ,σ} z^δ χ^σ.

3. **Build `PDECoefficients` from the polynomial fit.** For each h-derivative term d = (j, α_r, β_χ), the polynomial coefficient c_{δ,σ} maps to a G-coefficient entry `(γ=0, δ, σ, α, β, j) → c_{δ,σ}` in the `PDECoefficients` format.

4. **Call `assemble_system(K_corr, basis, a)`** — the existing Galerkin assembly handles everything: spectral operators, A_j absorption, weighted derivative identities. The output is `METRICSSystem(D0_corr, 0, 0)`.

**Key subtlety**: the h-derivatives in the correction evaluator are ∂_r^α ∂_χ^β h_j (physical coordinates), but the Galerkin assembly expects ∂_z^α ∂_χ^β u_j (compactified coordinates with weighted operators). You must convert the derivative basis:
- ∂_r h = (dz/dr) ∂_z h = (dz/dr) [A'u + A ∂_z u]
- ∂_r² h = ... (chain rule with d²z/dr²)
- The assembly operators already encode: ∂_z^0 = I, ∂_z^1 = d/dz, ∂_z^2 = (1-z²)d²/dz²
- The angular operators encode: ∂_χ^1 = (1-χ²)d/dχ, etc.

The conversion from r-derivatives to z-derivatives + A_j factoring can be done either:
- **(a) Symbolically**: expand ∂_r^α ∂_χ^β (A_j × u_j) via product rule, collect by (u_j, ∂_z u_j, ∂_z² u_j, ...), absorb A_j factors into the coefficient. This gives new coefficients that multiply u-derivatives, which map directly to the Galerkin assembly format.
- **(b) Numerically**: at each grid point, build a local matrix that maps u-derivative values to h-derivative values (using A_j and dz/dr at that point). Invert this to express the correction equation in u-derivatives. Then fit the resulting coefficients as polynomials.

**Approach (a) is strongly recommended** — it's what the GR pipeline does (the symbolic extraction in `symbolic_pipeline.jl` already handles the r→z change of variables and A_j factoring). The `extract_G_bespoke` function shows the pattern. For the sGB correction, the G coefficients are non-polynomial but can be represented as truncated Chebyshev×Legendre expansions.

### Alternative: direct source vector computation

If building the full D̃⁽¹⁾ matrix proves too hard, you only need the **product** `D̃⁽¹⁾(ω₀) · v₀` (a single vector). This can be computed directly:
1. Reconstruct the physical fields h_j(r, χ) = A_j × Σ v₀[j,n,l] T_n P_l^m at each point
2. Compute all h-derivatives from v₀
3. Evaluate source_k = Σ_d C[k,d] × h_deriv_d at each point
4. Transform point values to Galerkin coefficients

The cancellation issue is less severe here because the full eigenvector v₀ (not individual test functions) determines the h-values, and the eigenvector coefficients naturally suppress the outer-boundary contribution. However, you still need the Galerkin transform (step 4), which requires the weighted spectral projection.

## Bugs fixed this session

1. **`compile_sgb_correction` spin parameter bug** (`sgb_linearize.jl:339`): `a_s = aparams[1]` was getting `hfuncs[1]` (4th return of `compute_sgb_correction_equations`) instead of `params[1]` (3rd return = Kerr spin). Fixed to `a_s = params[1]`.

2. **`parse_h_term_map` derivative order parsing** (`collocation.jl:188-206`): the old code counted occurrences of `"Differential(r"` to determine derivative order. This fails for the compact Symbolics.jl format `Differential(r, 2)` which has order 2 but only one occurrence. Fixed to extract the numeric order argument via regex.

3. **`_test_function_derivs` ∂³P/∂χ³ placeholder** (`collocation.jl:155-158`): was `d3P_l = 0.0` (placeholder). Fixed to compute via central finite differences on d²P/dχ².

4. **World age barrier** (`sgb_linearize.jl:409`): `csc.fns[k](args)` fails because `eval`-compiled functions are in a newer world age. Fixed with `Base.invokelatest(csc.fns[k], args)`.

5. **GR eigenvector extraction**: `test_sgb_e2e.jl` now uses `solve_qep_with_vectors` (QEP solver, finds ω₀ to 1e-13) instead of `solve_qnm` (Newton, fails to converge at N=8).

## What the compiled evaluator produces

`compile_sgb_correction(2)` gives a `CompiledSGBCorrection` with:
- 10 compiled functions (one per equation component)
- 40 h-derivative terms (verified, names printed in test output)
- Derivatives up to 3rd order: ∂³h₅/∂χ³, ∂³h₆/∂χ³, ∂²r∂χ h₅, ∂r∂²χ h₅/h₆
- 24 H-parameters, 6 base variables (r, χ, ω_re, ω_im, iu, a)
- JIT compile time: ~26s. Evaluation: ~6s/point (3 iu values × 40 probes × 10 equations)

`extract_sgb_coefficients_complex(csc, r, χ, ω₀, a, hp)` returns a 10×40 complex matrix C where `correction_k = Σ_d C[k,d] × h_deriv_d`. These coefficients are O(1) and well-behaved — the problem is ONLY in evaluating h_j = A_j × u_j at large r.

## Critical technical knowledge

### The symbolic blowup and how we solved it

The sGB metric corrections H₁-H₄ are 27-term rational functions of (r, χ) at fixed spin. Naively inserting them into the symbolic linearization produces expressions of **100-155 million characters** because Symbolics.jl doesn't auto-simplify rational functions.

**Solution**: Use 24 ABSTRACT symbolic variables (`H1, H1_r, H1_chi, ...`) instead of the actual polynomial expressions. The correction equations are LINEAR in these, so expressions stay at ~200K chars. At evaluation time, substitute numerical values from `sgb_background`.

### The iu-polynomial trick

Complex arithmetic is handled by treating the imaginary unit as a symbolic variable `iu`. Evaluate at iu=0, 1, -1, extract polynomial coefficients, substitute iu→im. See `extract_coefficients_complex` in `galerkin.jl`.

### ω-polynomial separation

The D̃ matrices are quadratic in ω: `D̃(ω) = D̃₀ + ω·D̃₁ + ω²·D̃₂`. Extract by evaluating at ω=0, 1, -1 and solving the 3-point system.

### What D̃⁽¹⁾ is missing (known limitation)

The current `sgb_linearize.jl` computes D̃⁽¹⁾ from **source 1 only**: the O(ζ) correction to the linearized Ricci from the modified background metric g = g_Kerr + ζ·H. There is also **source 2**: the linearized sGB source tensor A_μ^ν (Eq. 12 of the paper). If results from source 1 alone don't match the paper, source 2 must be added.

### The bespoke SparsePoly CAS (`sparse_poly.jl`)

For the GR pipeline, we built a custom CAS to replace `Symbolics.simplify_fractions` (which hangs). A similar approach could be built for the sGB tensor algebra.

### Vandermonde transform: why it's correct (when it works)

The GR Galerkin matrix D_G and the collocation matrix D_C are related by D_C = V · D_G where V is the evaluation matrix V[(iz,iχ), (n,l)] = T_n(z_iz) P_l^m(χ_iχ). This holds EXACTLY for polynomial coefficients because the weighted spectral operators ((1-z²)d²/dz² etc.) cancel their weight factors when evaluated at points: the G coefficients absorb the compensating 1/(1-z²) factor. So V⁻¹ · D_C = D_G. For non-polynomial coefficients (sGB), this gives a spectral interpolation approximation that converges for smooth functions.

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
  [dtilde.jl, factored_assembly.jl] — earlier D̃ approaches (reference)
  [poly_extract.jl, zspace_extract.jl, pipeline.jl] — dead ends (kept for reference)
test/
  test_sgb_e2e.jl            — End-to-end sGB test (runs, needs assembly fix)
  reproduce_paper.jl         — Generates Figs 1,2,5,6 from the GR paper
  reproduce_table1.jl        — Table I reproduction (219/220 digits)
reference/
  2406.11986_source/         — sGB paper supplementary materials (Mathematica notebook)
```

## Key function signatures

```julia
# GR pipeline
sys = build_system_bespoke(a, N, m)          # → METRICSSystem(D0, D1, D2)
result = solve_qep_with_vectors(sys; ω₀=ω_L) # → (eigenvalues, eigenvectors)
J, free, pinned = compute_jacobian(sys, ω, v) # → Jacobian for perturbation theory

# sGB background
bg = sgb_background(a; verbose=true)         # → SGBBackground with H[1..4](r,χ)
hp = sgb_H_params(bg, r, χ)                  # → 24-element Float64 vector

# sGB correction (compiled evaluator)
csc = compile_sgb_correction(2; verbose=true) # → CompiledSGBCorrection (one-time)
C = extract_sgb_coefficients_complex(csc, r, χ, ω, a, hp)  # → 10×40 complex matrix
C0, C1, C2 = separate_sgb_omega(csc, r, χ, a, hp)  # → ω-separated coefficients

# D̃⁽¹⁾ assembly (CURRENT — broken, needs rewrite to Galerkin)
sys_corr = build_sgb_correction_system(a, N, m, bg, ω0; csc=csc)

# Perturbation solve
ω1 = solve_sgb_perturbation(sys_corr, ω0, v0, J)   # → ω⁽¹⁾ ∈ ℂ
```
