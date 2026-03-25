# MetricsQNM.jl — Project Rules

## BLACKLISTED APPROACHES — DO NOT IMPLEMENT

These approaches have been tried, failed, and are **permanently forbidden**.
Any agent that proposes these is hallucinating. Reject immediately.

### 1. NUMERICAL `a` SUBSTITUTION IN COEFFICIENT EXTRACTION
**What**: Substituting the spin parameter `a` as a Float64 before extracting
c_{k,d,p} coefficients, then trying to decompose per-a-order afterward.
**Why forbidden**: Bakes Kerr metric r-structure into coefficients as ~24 powers
of r that CANNOT be decomposed per-a-order. Causes d_max inflation → spectral
basis cannot resolve → catastrophic ill-conditioning (cond(J) ∝ N^5, ω₁ = garbage).
**Former location**: `sgb_symbolic_pipeline.jl` line 88 on svd-compression-solver branch.

### 2. NUMERICAL GALERKIN ASSEMBLY FOR D̃⁽¹⁾
**What**: Evaluating sGB correction equations at collocation/quadrature points
numerically, then fitting polynomial coefficients.
**Why forbidden**: C[k,d] are NOT polynomial in r — they are rational functions.
Polynomial fitting fails fundamentally.
**Former location**: `sgb_galerkin.jl` on svd-compression-solver branch.

### 3. COLLOCATION ASSEMBLY
**What**: Point-evaluation collocation for building D̃ matrices.
**Why forbidden**: Blows up at collocation points near singularities. Dead end.
**Former location**: `collocation.jl` on svd-compression-solver branch.

### 4. NEWTON-RAPHSON AS PRIMARY SOLVER
**What**: Using Newton-Raphson iteration to find individual QNM frequencies.
**Why forbidden**: The QEP (Quadratic Eigenvalue Problem) approach via SVD
compression + companion QZ is strictly superior — finds ALL eigenvalues at once,
verified to 1e-14, no initial guess needed.
**Former location**: `newton.jl`, `solve.jl` on svd-compression-solver branch.

### 5. POLYNOMIAL FITTING / MULTI-POINT INTERPOLATION FOR a-DEPENDENCE
**What**: Evaluating at multiple a-values and fitting a polynomial in a².
**Why forbidden**: Ill-conditioned for high a-orders. Numerical hack that avoids
the real problem. The correct approach is symbolic.

### 6. ANY APPROACH THAT AVOIDS EXTENDING THE CAS
**What**: Any workaround that sidesteps making SparsePoly handle `a` symbolically.
**Why forbidden**: The paper keeps `a` symbolic. We must too. Symbolics.jl cannot
handle it (100-155M char blowup). Therefore SparsePoly must be extended.
This is the work. Do not avoid it.

---

## MANDATORY APPROACH — THE ONLY CORRECT PATH

### Principle: Symbolic all the way, matching the paper

The paper (arxiv:2406.11986) uses Mathematica to keep `a` symbolic throughout the
entire pipeline. We must do the same using our bespoke SparsePoly CAS.

### The correct sGB pipeline:

1. **Load H_i as per-a-order RatPolys** (DONE — `sgb_background.jl`)
2. **Extend SparsePoly to N_VARS=6** with `a` as variable index 6
3. **Update SymToPolyCtx** to build Σ(r,χ,a), Δ(r,a) as 6-variable polynomials
4. **Update denominator identification** to match Σ(r,χ,a), Δ(r,a) in 6 variables
5. **Extract c_{k,d,p} with symbolic `a`** — NO numerical substitution
6. **Split c_{k,d,p} by a-exponent** → per-a-order c with moderate r-degree
7. **Multiply per-order c × H** → moderate d_max per order (~2k+5, not ~24)
8. **Assemble D̃⁽¹⁾ per a-order** via same Galerkin projection as GR
9. **Sum with a^{2k} weights** → evaluate `a` numerically ONLY here
10. **QEP solver** for eigenvalue perturbation (already done)

### Design patterns to follow:

- **TensorGR.jl**: Keep parameters symbolic, defer numerical evaluation
- **SparsePoly CAS**: Extend, don't replace. The foundation is proven (219/220 digits)
- **Per-a-order processing**: Proven approach for managing expression growth
- **Structural denominator tracking**: DenomSig is the right idea, extend to 6-var factors

---

## GROUND RULES

1. **Physics is ground truth.** The paper's results are correct. If our code
   disagrees, our code is wrong.

2. **Symbolic over numerical.** Never substitute a parameter numerically when
   it could remain symbolic. Numerical evaluation happens at final assembly only.

3. **No band-aids.** If something doesn't work, understand WHY. Fix the root
   cause. Do not patch around it.

4. **QEP is the solver.** SVD compression + companion QZ. Not Newton-Raphson.

5. **Rich diagnostics.** Flush after every print. Fail fast on unexpected values.
   Use proper Julia profiling tools (@time, @profile), not ad-hoc timing.

6. **TensorGR.jl is read-only.** Lives at `../TensorGR.jl`. Do NOT edit.

7. **Conventions**: M=1 everywhere. Table values are ωM (dimensionless).
   Use `--threads=8` for Julia.

---

## FILE MAP (clean main branch)

```
src/
  MetricsQNM.jl              — Module root
  kerr.jl                    — KerrParams, Σ, Δ, r±
  perturbation_ansatz.jl     — A_k(r), z↔r transforms
  spectral.jl                — ChebyshevBasis, LegendreBasis
  leaver.jl                  — Leaver CF QNM solver (reference oracle)
  linearize.jl               — Symbolic: Kerr → δΓ → δR → 10 field equations
  sparse_poly.jl             — Bespoke SparsePoly CAS (TO BE EXTENDED to 6 vars)
  coefficients.jl            — PDECoefficients struct
  symbolic_pipeline.jl       — extract_G_bespoke, build_system_bespoke (GR)
  assembly.jl                — Galerkin spectral inner products → D̃ matrices
  rectangular_qep.jl         — SVD compression QEP solver
  sgb_background.jl          — Parse H₁-H₄ from Mathematica notebook
  sgb_linearize.jl           — sGB correction equations (abstract H-params)
  sgb_perturbation.jl        — Eigenvalue perturbation solver (Eq. 111)

test/
  runtests.jl                — Test runner
  test_kerr.jl               — Kerr background tests
  test_leaver.jl             — Leaver oracle tests
  test_spectral.jl           — Spectral basis tests
  test_svd_qep.jl            — QEP solver tests
  test_symbolic_pipeline.jl  — G-extraction tests
  reproduce_table1.jl        — Table I reproduction (219/220 digits)
  reproduce_paper.jl         — Figures 1, 2, 5, 6

papers/                      — Paper sources (GR + sGB)
reference/                   — Reference PDFs and figures
figures/                     — Reproduced figures
```

## WHAT WAS ABANDONED (on svd-compression-solver branch)

The following files exist ONLY on the `svd-compression-solver` branch and must
NEVER be brought to main:

- `sgb_symbolic_pipeline.jl` — numerical `a` extraction (THE core heresy)
- `sgb_galerkin.jl` — numerical Galerkin (C[k,d] not polynomial)
- `collocation.jl` — collocation dead end
- `dtilde.jl` — earlier D̃ approach
- `factored_assembly.jl` — earlier assembly approach
- `poly_extract.jl` — dead end
- `zspace_extract.jl` — dead end
- `pipeline.jl` — dead end
- `galerkin.jl` — old Galerkin
- `newton.jl` — Newton-Raphson (superseded by QEP)
- `solve.jl` — old solver
- `symbolic_decompose.jl` — old decomposition

These are preserved on that branch for archaeological reference only.
