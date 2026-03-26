# Handoff — MetricsQNM.jl

## Project in one sentence

Compute quasinormal mode (QNM) frequencies in Kerr and scalar-Gauss-Bonnet (sGB)
gravity by spectral Galerkin methods + QEP solver, reproducing Tables I-III of
arxiv:2312.08435 and arxiv:2406.11986.

## What works (GR foundation — COMPLETE)

The GR METRICS pipeline reproduces **219/220 digits** of Table I (arxiv:2312.08435)
for the fundamental (n=0, l=2, m=2) Kerr QNM across 11 spins a=0.005..0.95.

Pipeline: `compute_field_equations(2)` → symbolic linearized Ricci →
`extract_G_bespoke` (bespoke SparsePoly CAS) → `build_system_bespoke(a, N, m)` →
`METRICSSystem(D0, D1, D2)` → `solve_qep_svd` (SVD compression + companion QZ)
→ eigenvalue at machine precision.

---

## ⚠️ STRATEGIC DECISION: RATIONAL GALERKIN — THE ONLY PATH FORWARD ⚠️

### The problem we hit

The sGB correction equations have **rational** coefficients — denominators with
Σ^p Δ^q (1-χ²)^s where the powers vary per H-parameter index p. Three bugs were
found and fixed in the clearing-based approach:

1. **Bug #1:** D̃⁽¹⁾ normalization must match D̃⁽⁰⁾ (FIXED)
2. **Bug #2:** Clearing must be uniform per equation k across all (d,p) (FIXED)
3. **Bug #3:** D̃⁽⁰⁾ and D̃⁽¹⁾ must use the SAME clearing factor (FIXED)

Fixing all three requires inflating clearing to P=6, Q=3, S=2 (global max across
all equations and H-parameters). This inflates GR d_max from 9 to 18, which:
- **Breaks Newton refinement** — spurious eigenvalues steal the basin of attraction
- **Requires N ≥ 19** — below this the basis can't resolve d_max=18
- **Makes full QEP impractical** — O(n³) QZ on ~5000×5000 companion at N=20
- **Still gives |ω₁| ~ O(10³)** vs reference O(0.1) even at N=16

### Why clearing is fundamentally wrong for sGB

The GR equations have denominators Σ^3 Δ^1 (1-χ²)^1 — moderate, uniform across
all terms. Clearing these to polynomials adds d_max=9, manageable at N=12.

The sGB equations have ADDITIONAL denominators from H-parameters (H1-H4 enter the
metric with different Σ/Δ factors). Clearing these adds up to Σ^6 Δ^3 (1-χ²)^2 —
**DOUBLE the GR clearing**. This is not a bug to fix; it's a fundamental mismatch
between polynomial clearing and rational coefficient structure.

**The paper's Mathematica handles rational coefficients natively.**
**We tried to avoid this by clearing. That path is exhausted. It does not work.**

### The correct approach: Rational Galerkin inner products

**DO NOT try to clear sGB denominators to match GR.** Instead:

1. **Clear to GR level only** (P=3, Q=1, S=1) — same as the working GR pipeline
2. **Track the residual denominator** per (k,d,p) as a DenomSig offset:
   - If sGB term needs Σ^5 but GR clears Σ^3, residual = 1/Σ^2
   - This residual is a known rational function of (r, χ)
3. **Extend assembly to compute weighted inner products:**
   ```
   ⟨T_n P_l | G(r,χ,ω) / Σ^k(r,χ) | T_n' P_l'⟩
   ```
   instead of the current polynomial-only:
   ```
   ⟨T_n P_l | G(r,χ,ω) | T_n' P_l'⟩
   ```
4. **D̃⁽⁰⁾ and D̃⁽¹⁾ automatically match** — both use P=3,Q=1,S=1 base clearing
5. **d_max stays at ~9** — Newton refinement works, N=12 resolves everything
6. **No degree inflation** — the rational factor is in the integration weight, not
   the polynomial degree

### How to compute weighted inner products

The denominators are products of:
- Σ = r² + a²χ²  →  quadratic in z after r→z transform
- Δ = r² - 2r + a²  →  quadratic in z
- (1-χ²)  →  quadratic in χ

For fixed (a, k, equation), 1/Σ^p is a smooth rational function on the (z,χ)
domain. Its Chebyshev×Legendre expansion converges exponentially. Options:

**Option A:** High-order quadrature (Gauss-Chebyshev × Gauss-Legendre) to evaluate
the weighted inner product to machine precision. ~100 quadrature points suffices.

**Option B:** Pre-expand 1/Σ^p as a Chebyshev×Legendre series (via FFT on fine
grid), then convolve with the polynomial inner products.

**Option C:** Symbolic partial fractions of 1/Σ^p in z, then exact Chebyshev
integrals of rational functions via residue calculus.

Option A is simplest, O(N_quad × n_basis²) per equation. Pre-computed for each
unique residual DenomSig. NOT "numerical Galerkin" in the forbidden sense — the
coefficients c_{k,d,p} are still extracted symbolically via SparsePoly CAS. Only
the integration step uses quadrature, which is inherently numerical anyway.

### What this preserves

- SparsePoly CAS with symbolic `a` — untouched
- Per-a-order extraction and splitting — untouched
- c × H convolution — untouched (works on RatPolys)
- QEP/Newton eigenvalue solver — untouched
- Perturbation theory — untouched
- GR pipeline — completely untouched (P=3 clearing, d_max=9)

### What this changes

- `assembly.jl`: extend `assemble_system` to accept per-equation DenomSig residuals
  and use weighted quadrature inner products for non-trivial residuals
- `build_sgb_Dtilde1`: pass residual DenomSig from extraction to assembly
- `extract_sgb_correction_symbolic_a`: return residual DenomSig per (k,d,p) instead
  of clearing to max power

### Why this was not done earlier

Previous sessions tried to avoid extending the CAS/assembly infrastructure by
using denominator clearing as a shortcut. This is explicitly blacklisted in
CLAUDE.md rule 6: "ANY APPROACH THAT AVOIDS EXTENDING THE CAS." Clearing IS such
an avoidance — it converts the hard problem (rational inner products) into an easy
one (polynomial inner products) at the cost of degree inflation. The inflation is
fatal for sGB because the extra denominators from H-parameters push clearing
powers far beyond what the spectral basis can resolve.

**This is the work. Do not avoid it.**

---

## sGB pipeline status (Source 1 only)

| Step | Function | Status |
|------|----------|--------|
| SparsePoly CAS (6 vars, symbolic `a`) | `sparse_poly.jl` | DONE |
| GR per-a-order extraction | `extract_G_bespoke_symbolic_a` | DONE, verified 5.77e-16 |
| GR per-a-order assembly | `build_system_bespoke_sgb` | DONE, 13.1 digits |
| sGB c_{k,d,p} extraction | `extract_sgb_correction_symbolic_a` | DONE (two-pass uniform) |
| c × H convolution per-a-order | `build_sgb_Dtilde1` | DONE (norm fixed) |
| **Rational Galerkin assembly** | `assemble_system` extension | **NOT STARTED — THE BLOCKER** |
| _r_to_z with negative δ | `_r_to_z` extended | DONE |
| Eigenvalue perturbation | `solve_sgb_perturbation` | DONE (code correct) |

---

## Bugs found and fixed (2026-03-26)

### Bug #1: Normalization mismatch (FIXED)
D̃⁽¹⁾ used independent normalization instead of GR norm factors.
**Files:** `src/symbolic_pipeline.jl`, `test/convergence_sgb_N.jl`

### Bug #2: Inconsistent denominator clearing per (k,d,p) (FIXED)
Two-pass approach: collect max denom per equation k, clear uniformly.
**Files:** `src/symbolic_pipeline.jl` (extraction refactored)
**Beads:** MetricsQNM.jl-wwy

### Bug #3: Clearing mismatch D̃⁽⁰⁾ vs D̃⁽¹⁾ (FIXED but superseded)
Added `sgb_clearing_targets()` returning global max P=6,Q=3,S=2.
**This fix is correct but causes d_max inflation (9→18). It will be
replaced by rational Galerkin inner products, which eliminate the
need for matched clearing entirely.**

---

## QEP solver: use Newton, not full QEP

Research in `../af-tests/examples13/` established:

- **Full QEP** (SVD compress + companion QZ) finds ALL eigenvalues — O(n³), ~16min at N=20
- **Newton on σ_min** (`solve_qep_newton`) refines ONE eigenvalue — O(k·mn²), ~seconds
- We already have Leaver ω₀ as initial guess → Newton converges in ~10 iterations
- `solve_qep_newton` is already implemented in `src/rectangular_qep.jl:90-116`
- Eigenvector v₀ comes from the SVD at convergence: `v₀ = V[:, end]`

**Use Newton for production. Full QEP only for validation at small N.**

Newton diverges with P=6 clearing (spurious eigenvalues from d_max=18).
With rational Galerkin (d_max=9), Newton will work as designed.

---

## Reference implementation findings

The Supplementary_materials.nb contains **NO CODE** — only pre-computed data:
- Φ (scalar field), H1-H4 (metric corrections) as series in a up to a^40
- Ω_H⁽¹⁾, κ⁽¹⁾ as polynomials in a

The actual METRICS solver is private Mathematica code (UIUC, not published).

Key facts:
1. Paper uses Newton-Raphson, not QEP — our Newton is equivalent
2. Paper keeps rational coefficients natively — we must do the same
3. Same N for both GR and sGB — same spectral basis
4. ω₁ via pinv(J) · source — single linear solve
5. Full system is 11×7 blocks (10 Einstein + 1 scalar, 6 metric + 1 scalar)
6. Paper does NOT decompose into Source 1-4 — all assembled together

---

## What to do next (priority order)

1. **Implement rational Galerkin inner products** — THE BLOCKER
   - Extend `assemble_system` to accept per-equation DenomSig residuals
   - Use quadrature (Gauss-Chebyshev × Gauss-Legendre) for weighted integrals
   - Validate: GR with residual DenomSig(0,0,0,0) must reproduce current results

2. **Rewire sGB pipeline** to use rational assembly
   - `extract_sgb_correction_symbolic_a`: clear to P=3,Q=1,S=1 only, return residual
   - `build_sgb_Dtilde1`: pass residual to assembly
   - Remove `sgb_clearing_targets()` — no longer needed

3. **N-convergence with rational Galerkin** — should converge at N=12-16
   - Newton refinement from Leaver (no full QEP needed)
   - |ω₁| should be O(0.1) if extraction is correct

4. **Implement Sources 2-4** (after Source 1 converges)

---

## Key function signatures

```julia
# GR pipeline (working, unchanged)
sys, nf = build_system_bespoke(a, N, m)
ω₀ = solve_qep_newton(sys, ω_L)  # or solve_qep_with_vectors for validation
J, free, pinned = compute_jacobian(sys, ω, v)

# GR per-a-order (verified)
sys, nf, disc = build_system_bespoke_sgb(a, N, m; verify=true, verbose=true)

# sGB extraction (two-pass uniform clearing, returns residual denoms)
c_kdp, a_powers, max_denom = extract_sgb_correction_symbolic_a(; verbose=true)

# sGB D̃⁽¹⁾ assembly (norm_factors required)
sys_corr = build_sgb_Dtilde1(a, N, m; norm_factors=nf, max_a_order=2, verbose=true)

# Eigenvalue perturbation
ω1 = solve_sgb_perturbation(sys_corr, ω0, v0, J)

# sGB background
H_per_order, a_powers = load_H_ratpolys_per_order(; verbose=true)
```

## Conventions

- `M = 1` everywhere. Table values are `ωM` (dimensionless).
- Use `--threads=8` for Julia. More threads cause OpenBLAS contention.
- TensorGR.jl at `../TensorGR.jl` — read-only dependency.
- Newton over full QEP for single eigenvalues. Full QEP only for validation.
- See CLAUDE.md for blacklisted approaches and mandatory rules.
- **3 subagents before core code changes, reviewer after. No exceptions.**
