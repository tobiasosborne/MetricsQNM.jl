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

## ⚠️ THE PAPER'S ACTUAL ALGORITHM (2026-03-28 session findings) ⚠️

### What the paper does (arxiv:2406.11986, verbatim from source)

The paper's approach, established by careful reading of the source tex:

1. **Clear ALL denominators** (Σ, Δ, 1-χ²) to polynomial form for BOTH
   GR and sGB terms together. No algebraic operators, no rational Galerkin.
   Quote (line 962): "after factorization and multiplication through common
   denominators"

2. **Divide out common prefactors** AFTER clearing — Σ^p Δ^q (1-χ²)^s
   factors that multiply the ENTIRE equation (not individual terms).
   Quote (line 976-978): "we divide the equations by them to simplify
   them and improve their numerical stability."

3. **Newton-Raphson on the FULL combined system** D̃⁰(ω) + ζ·D̃¹(ω),
   iterating on BOTH (v, ω) simultaneously. NOT σ_min Newton on ω alone.
   Quote (line 197): "a Newton-Raphson algorithm to simultaneously solve
   all the linear, homogeneous algebraic equations"

4. **Scan N from 1 to 25**, select optimal N via backward modulus difference.
   Quote (line 1302): "we compute ω⁽¹⁾ from N=1 to 25"

5. **Per-equation clearing** — each equation cleared independently.
   Quote (line 976): "After factorizing each of the linearized field equations"

6. **D̃ matrices are linear in ζ** — GR and sGB assembled together.
   Quote (line 1063): "which are all linear in ζ"

### What we tried and what failed (this session)

| Approach | Result | Why it failed |
|----------|--------|---------------|
| Independent clearing (D̃⁰ at P=3,Q=1,S=1; D̃¹ at per-eq) | |ω₁| = 1.26e8 | Gauge mismatch: pinv residual = 0.989 |
| Matched clearing + σ_min Newton on D̃⁰ alone | Diverges to spurious ω = 120-3171i | Inflated d_max=16 creates spurious eigenvalues that steal σ_min basin |
| Matched clearing + σ_min Newton on full D̃⁰+ζD̃¹ | All N=8..16 diverge to spurious | σ_min Newton has no eigenvector info → wrong basin |
| Matched clearing + FULL Newton on (v,ω) jointly | **N=12: promising but stalls** | N < d_max → basis aliasing. N≥16 not yet tested |

### Key diagnostic results (2026-03-28)

**Per-equation sGB clearing targets** (irreducible — verified by probing
1165 c_{k,d,p} triples for missed Σ/Δ cancellations: 0 missed Σ, 0 missed Δ):

```
eq  1: Σ^6 Δ^2 (1-χ²)^1   excess over GR: +8 r-deg
eq  2: Σ^5 Δ^3 (1-χ²)^1   excess over GR: +8 r-deg
eq  3: Σ^5 Δ^2 (1-χ²)^2   excess over GR: +6 r-deg
eq  4: Σ^6 Δ^2 (1-χ²)^1   excess over GR: +8 r-deg
eq  5: Σ^5 Δ^2 (1-χ²)^1   excess over GR: +6 r-deg
eq  6: Σ^5 Δ^2 (1-χ²)^2   excess over GR: +6 r-deg
eq  7: Σ^6 Δ^2 (1-χ²)^1   excess over GR: +8 r-deg
eq  8: Σ^5 Δ^2 (1-χ²)^1   excess over GR: +6 r-deg
eq  9: Σ^6 Δ^2 (1-χ²)^1   excess over GR: +8 r-deg
eq 10: Σ^6 Δ^2 (1-χ²)^1   excess over GR: +8 r-deg
```

**Per-a-order d_max** (with per-equation clearing, no global inflation):
```
a^0: 16   a^2: 14   a^4: 12   a^6: 10   a^8:  8
a^1: 15   a^3: 13   a^5: 11   a^7:  9   a^9:  7
```

**reduce_ratpoly quality**: 76/131 triples in eq 1 miss (1-χ²) cancellations,
but (1-χ²) has zero r-degree → doesn't affect d_max. Σ and Δ cancellations:
NONE missed. The denominators are irreducible.

---

## WHAT TO DO NEXT (priority order)

### 1. Complete the full Newton test at N≥16

**STATUS: IN PROGRESS — test running at session end (test/test_full_newton_v2.jl)**

The full Newton-Raphson (`solve_qep_full_newton` in `rectangular_qep.jl`) iterates
on (v, ω) jointly, using the Jacobian J = [D̃(ω)[:,free] | D̃'(ω)·v]. At N=12 it
showed promising behavior (ω converging toward physical region, residual dropping
from 4.7e-6 to 3.6e-7) but stalled because N=12 < d_max=16 → basis aliasing.

**Action:** Run `test/test_full_newton_v2.jl` and examine N=16, 18, 20 results.
If full Newton converges at N≥16, the paper's algorithm works and we're done
with the solver question. If it still fails, investigate:
- Is the "divide out common prefactors" step missing? (Our code doesn't do step 2)
- Is the initial v₀ from SVD at ω_Leaver good enough?

### 2. Implement "divide out common prefactors" (paper's step 2)

The paper says: after clearing to LCD, the resulting polynomial equation has
common factors of Σ^p Δ^q (1-χ²)^s that can be divided out. We DON'T do this.

This could reduce d_max. The approach:
- After clearing all (d,p) triples in equation k to the per-equation LCD
- SUM all cleared terms for equation k
- Test if the full sum is divisible by Σ, Δ, or (1-χ²)
- Divide out as many common factors as possible

This is DIFFERENT from our per-triple `reduce_ratpoly` (which finds no missed
Σ/Δ cancellations). Cross-term cancellations might appear only in the sum.

### 3. If Newton works: N-convergence study

Scan N=8..25 for a=0.3 with backward modulus difference B(N) = |ω(N) - ω(N-1)|.
Extract ω₁ ≈ (ω(ζ) - ω₀)/ζ at optimal N. Compare with paper Table II.

### 4. Implement Sources 2-4 (after Source 1 converges)

---

## New code added this session

### `_r_to_z_per_eq(K_r, rp, d_max_per_eq::Vector{Int})`
**File:** `src/symbolic_pipeline.jl` (after line 240)
Per-equation variant of `_r_to_z`. Each equation k uses its own
`(1+z)^{d_max_per_eq[k]}` multiplication. Required for matched clearing
where different equations have different denominator levels.

### `extract_G_bespoke_symbolic_a(; per_eq_clearing=Dict{Int,DenomSig})`
**File:** `src/symbolic_pipeline.jl` (modified)
Added optional `per_eq_clearing` parameter. When provided, each equation k
uses `per_eq_clearing[k]` instead of global (P,Q,S) for denominator clearing.

### `build_matched_sgb_system(a, N, m; max_a_order=2, verbose=false)`
**File:** `src/symbolic_pipeline.jl` (after `build_sgb_Dtilde1`)
Builds D̃⁰ and D̃¹ with matched per-equation clearing. Steps:
1. Extract sGB per-equation clearing targets (max_denom_per_k)
2. Extract GR coefficients with per-equation clearing matching sGB
3. Build D̃⁰ with `_r_to_z_per_eq`
4. Build D̃¹ with same `_r_to_z_per_eq` (same d_max_per_eq)
5. Normalize both with shared factors

### `solve_qep_full_newton(sys, ω₀, v₀; parity, max_iter, tol, verbose)`
**File:** `src/rectangular_qep.jl` (after `solve_qep_newton`)
The paper's Newton-Raphson: iterates on (v, ω) jointly.
Uses Jacobian J = [D̃(ω)[:,free] | D̃'(ω)·v], solves J\(D̃·v) each step.
Converges to the mode nearest (v₀, ω₀) in the full (v,ω) space,
avoiding spurious eigenvalue basins that plague σ_min Newton.

---

## Test scripts added this session

- `test/diagnose_clearing.jl` — Per-equation denominator diagnostic (90s)
- `test/diagnose_reduce.jl` — reduce_ratpoly cancellation quality test (130s)
- `test/test_independent_clearing.jl` — Independent clearing test (FAILED: gauge mismatch)
- `test/test_matched_clearing.jl` — Matched clearing + σ_min Newton (FAILED: spurious eigenvalues)
- `test/test_full_newton.jl` — σ_min Newton on full D̃⁰+ζD̃¹ (FAILED: all N diverge)
- `test/test_full_newton_v2.jl` — Full Newton on (v,ω) (**IN PROGRESS — check N≥16**)

---

## sGB pipeline status (Source 1 only, updated 2026-03-28)

| Step | Function | Status |
|------|----------|--------|
| SparsePoly CAS (6 vars, symbolic `a`) | `sparse_poly.jl` | DONE |
| GR per-a-order extraction | `extract_G_bespoke_symbolic_a` | DONE, verified 5.77e-16 |
| GR per-a-order assembly | `build_system_bespoke_sgb` | DONE, 13.1 digits |
| sGB c_{k,d,p} extraction | `extract_sgb_correction_symbolic_a` | DONE (two-pass uniform) |
| c × H convolution per-a-order | `build_sgb_Dtilde1` | DONE (norm fixed) |
| Per-equation matched clearing | `build_matched_sgb_system` | DONE (new this session) |
| Per-equation r→z transform | `_r_to_z_per_eq` | DONE (new this session) |
| Full Newton-Raphson (v,ω) | `solve_qep_full_newton` | DONE (new this session) |
| **Full Newton at N≥16** | test_full_newton_v2.jl | **IN PROGRESS — THE BLOCKER** |
| Divide-out common prefactors | not implemented | NEXT if Newton stalls |
| _r_to_z with negative δ | `_r_to_z` extended | DONE |
| Eigenvalue perturbation | `solve_sgb_perturbation` | DONE (code correct) |

---

## Solver strategies (ranked)

1. **Full Newton on (v,ω)** (`solve_qep_full_newton`) — paper's algorithm.
   Iterates on both eigenvector and eigenvalue. Robust to spurious eigenvalues.
   **USE THIS.** Test at N≥16 with matched clearing.

2. **σ_min Newton** (`solve_qep_newton`) — iterates on ω only.
   Works for GR (d_max=9, few spurious eigenvalues). FAILS for sGB with
   matched clearing (d_max=16, spurious eigenvalues steal basin).

3. **Full QEP** (`solve_qep_svd`) — finds ALL eigenvalues. O(n³).
   Too slow for N≥16. Only for validation at small N.

4. **Eigenvalue perturbation** (`solve_sgb_perturbation`) — single linear solve.
   REQUIRES D̃⁰ and D̃¹ in same gauge. Works only if D̃⁰ eigenvalue is found
   first (circular dependency with matched clearing).

---

## Key function signatures (updated)

```julia
# GR pipeline (working, unchanged)
sys, nf = build_system_bespoke(a, N, m)
ω₀ = solve_qep_newton(sys, ω_L)

# Matched sGB system (NEW)
sys_gr, sys_corr, nf, matched = build_matched_sgb_system(a, N, m;
    max_a_order=2, verbose=true)

# Combined system for full Newton
sys_full = METRICSSystem(
    sys_gr.D0 .+ ζ .* sys_corr.D0,
    sys_gr.D1 .+ ζ .* sys_corr.D1,
    sys_gr.D2 .+ ζ .* sys_corr.D2, N, m, a)

# Initial (v₀, ω₀) from SVD at Leaver frequency
P_init = Dtilde(sys_full, ω_leaver)
v_init = conj(svd(P_init).Vt[end, :])

# Full Newton-Raphson (paper's algorithm)
result = solve_qep_full_newton(sys_full, ω_leaver, v_init;
    parity=:polar, max_iter=30, tol=1e-12, verbose=true)
ω_sGB = result.ω
ω₁ = (ω_sGB - ω_leaver) / ζ

# Leaver reference
ω_L = leaver_qnm(a; s=-2, l=2, m=2, n=0)

# sGB background
H_per_order, a_powers = load_H_ratpolys_per_order(; verbose=true)
```

## Conventions

- `M = 1` everywhere. Table values are `ωM` (dimensionless).
- Use `--threads=8` for Julia. More threads cause OpenBLAS contention.
- TensorGR.jl at `../TensorGR.jl` — read-only dependency.
- **Follow the paper's algorithm. Not following the paper is heresy.**
- See CLAUDE.md for blacklisted approaches and mandatory rules.
- **3 subagents before core code changes, reviewer after. No exceptions.**
