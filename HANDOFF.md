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

## sGB symbolic pipeline — BROKEN (Source 1 only)

**STATUS: ω₁ produces O(100-1000) garbage. Two bugs identified, one fixed, one open.**

### Pipeline (all on main branch):

| Step | Function | Status |
|------|----------|--------|
| SparsePoly CAS (6 vars, symbolic `a`) | `sparse_poly.jl` | DONE |
| GR per-a-order extraction | `extract_G_bespoke_symbolic_a` | DONE, verified 5.77e-16 |
| GR per-a-order assembly | `build_system_bespoke_sgb` | DONE, 13.1 digits |
| sGB c_{k,d,p} extraction | `extract_sgb_correction_symbolic_a` | **BUGGY — see Bug #2 below** |
| c × H convolution per-a-order | `build_sgb_Dtilde1` | Normalization FIXED |
| _r_to_z with negative δ | `_r_to_z` extended | DONE |
| Eigenvalue perturbation | `solve_sgb_perturbation` | DONE (code correct, input data wrong) |

### N-convergence results (a=0.3, max_a_order=2, Source 1 only):

**BEFORE normalization fix (independent normalization — WRONG):**
```
N=6:  ω₁ = +0.263 + 0.335i   |ω₁| = 0.43
N=8:  ω₁ = -0.252 + 0.335i   |ω₁| = 0.42
N=10: ω₁ = +0.905 + 0.221i   |ω₁| = 0.93
N=12: ω₁ = -1.752 - 1.526i   |ω₁| = 2.32
```

**AFTER normalization fix (GR norm factors — CORRECT but still garbage):**
```
N=6:  ω₁ ~ O(100)
N=8:  ω₁ = -398 + 114i       |ω₁| = 413
N=10: ω₁ = -321 - 226i       |ω₁| = 392
N=12: ω₁ = -1215 + 137i      |ω₁| = 1222
```

Both show: sign-flipping, magnitude growing with N. Not converging. GR ω₀ is fine
(|Δω_L| ~ 1e-12 to 1e-13), so the bug is in D̃⁽¹⁾ construction.

---

## Bugs found and fixed this session

### Bug #1: Normalization mismatch (FIXED)

**Problem:** `build_sgb_Dtilde1` at `symbolic_pipeline.jl:1179` called
`normalize_system!(sys_corr)` — independent per-equation normalization. The paper
(2406.11986, Eq. 111) requires D̃⁽¹⁾ to use the SAME normalization as D̃⁽⁰⁾.

**Fix applied:** Changed to `normalize_system!(sys_corr, norm_factors)` using GR
factors. Made `norm_factors` a required keyword argument. The 2-arg method already
existed at line 1225 with a docstring about this exact use case.

**Files changed:** `src/symbolic_pipeline.jl` (signature + normalization call),
`test/convergence_sgb_N.jl` (caller updated).

**Result:** Fix is correct but insufficient. ω₁ went from O(1) to O(1000) —
the independent normalization was masking the true magnitude of the underlying bug.

### Bug #2: Inconsistent denominator clearing across H-parameter index p (OPEN)

**Problem (diagnosed by research agent, not yet fixed):**

In `extract_sgb_correction_symbolic_a` (line 941), each c_{k,d,p} is individually
cleared with `clear_denominators(rp, P, Q, S, ctx)`. The actual clearing powers
are auto-detected per (k,d,p) and the cleared polynomial is kept, but the actual
powers used are DISCARDED (`poly, _ = clear_denominators(...)`).

Different H-parameter indices `p` produce c_{k,d,p} with DIFFERENT denominator
structures because H1..H4 enter the metric correction with different Σ/Δ factors:
- H1 enters via δg_tt = -H1 (no extra Σ/Δ)
- H2 enters via δg_tφ with 1/Σ factor
- H3 enters via δg_rr with Σ/Δ, δg_χχ with Σ/(1-χ²)
- H4 enters via δg_φφ with 1/Σ

After linearization, `clear_denominators` auto-detects different powers per p:
- c_{k,d,1} cleared with Σ^3 (for H1-related terms)
- c_{k,d,7} cleared with Σ^4 (for H2-related terms, one extra Σ)

In the convolution (build_sgb_Dtilde1), these differently-cleared polynomials are
SUMMED into the same equation k:

    Σ^3 · c_{k,d,1} · H1 + Σ^4 · c_{k,d,7} · H2 + ...

This is NOT Σ^P · (c · H1 + c · H2) for any single P. The extra Σ factor on some
terms corrupts the physical equation.

**Why GR doesn't have this bug:** In GR extraction, there's no H-parameter index p.
Each (k,d) has a single coefficient. Denominators depend only on Kerr background,
consistent across d values within each equation k.

**Proposed fix (NOT YET IMPLEMENTED):**
1. First pass: compute symexpr_to_poly for all (k,d,p) to get RatPolys
2. Per equation k: find MAX denominator power across all (d,p) pairs
3. Second pass: clear ALL triples for equation k with those MAX powers
4. This ensures common clearing factor per equation → consistent convolution

**Beads issue:** MetricsQNM.jl-98u (normalization, DONE). Need new issue for Bug #2.

---

## Critical findings from reference implementation research

The Supplementary_materials.nb contains **NO CODE** — only pre-computed data:
- Φ (scalar field), H1-H4 (metric corrections) as series in a up to a^40
- Ω_H⁽¹⁾ (horizon angular velocity correction, polynomial in a up to a^39)
- κ⁽¹⁾ (surface gravity correction, polynomial in a up to a^40)

The actual METRICS solver is private Mathematica code (UIUC, not published).

### Key reference insights:
1. **Paper uses Newton-Raphson, not QEP** — our QEP is superior, gives same eigenvalues
2. **D̃⁽⁰⁾ and D̃⁽¹⁾ MUST use same normalization** — fixed
3. **Same N for both GR and sGB** — same spectral basis
4. **Per-equation normalization** divides by max coefficient magnitude — mandatory
5. **ω₁ via pinv(J) · source** — single linear solve, no iteration
6. **Paper does NOT decompose into Source 1-4** — all corrections assembled together
7. **Full system is 11×7 blocks** (10 Einstein + 1 scalar, 6 metric + 1 scalar field)

### Reference Table I values (022 mode, all sources, for comparison):
```
a=0.1, axial:  ω₁ = +0.07282 + 0.01353i  (δ ~ 1e-6)
a=0.1, polar:  ω₁ = -0.26228 - 0.07146i  (δ ~ 1e-5)
a=0.6, axial:  ω₁ = +0.13808 - 0.07249i  (δ ~ 1e-5)
a=0.6, polar:  ω₁ = -0.53314 + 0.10997i  (δ ~ 1e-6)
```

---

## Beads issue tracker

```
MetricsQNM.jl-98u [P0 BUG] Normalization mismatch — FIXED ──┐
MetricsQNM.jl-3dl [P1 BUG] d_max mismatch ───────────────────┤
                                                               ▼
MetricsQNM.jl-g63 [P1] N-convergence validation
                        │
                        ▼
MetricsQNM.jl-r28 [P2] Source 4: Ω_H⁽¹⁾, κ⁽¹⁾
                        │
                        ▼
MetricsQNM.jl-6rc [P2] Source 2: A_μ^ν ─────────┐
                        │                         │
                        ▼                         ▼
MetricsQNM.jl-ed6 [P2] Source 3: T_μ^ν    MetricsQNM.jl-29e [P2] h₇ + 11th eq
                        │                         │
                        └─────────────────────────┘
                                                  │
                                                  ▼
                        MetricsQNM.jl-5ja [P2] Final Table I validation
```

**CRITICAL: Need new P0 beads issue for Bug #2 (denominator clearing). This blocks
everything — until D̃⁽¹⁾ is correct, no convergence study or Source 2-4 work matters.**

---

## What to do next (priority order)

1. **Fix Bug #2** (denominator clearing in `extract_sgb_correction_symbolic_a`)
   - The fix is described above. Follow 3-subagent protocol before touching core code.
   - After fix: re-run N-convergence. ω₁ should drop to O(1e-5) or smaller for Source 1.

2. **Run diagnostic script** (test/convergence_sgb_N.jl, already updated)
   - A magnitude diagnostic comparing ||D̃⁽¹⁾_k||/||D̃⁽⁰⁾_k|| per equation would
     confirm the fix. Agent 3 wrote a comprehensive diagnostic script (not yet saved to file).

3. **Close MetricsQNM.jl-98u** and create issue for Bug #2

4. **Investigate d_max mismatch** (MetricsQNM.jl-3dl) — probably not the main issue
   but should be verified: GR uses d_max=9, sGB uses d_max=20.

---

## Key function signatures (UPDATED)

```julia
# GR pipeline (working)
sys, nf = build_system_bespoke(a, N, m)
result = solve_qep_with_vectors(sys; ω₀=ω_L)
J, free, pinned = compute_jacobian(sys, ω, v)

# GR per-a-order (verified matches numerical at 5.77e-16)
sys, nf, disc = build_system_bespoke_sgb(a, N, m; verify=true, verbose=true)

# sGB correction extraction (symbolic a — HAS BUG #2, denominator clearing)
c_kdp, a_powers = extract_sgb_correction_symbolic_a(; verbose=true)

# sGB D̃⁽¹⁾ assembly — norm_factors is REQUIRED (Bug #1 fix)
sys_corr = build_sgb_Dtilde1(a, N, m; norm_factors=nf, max_a_order=2, verbose=true)

# Eigenvalue perturbation (code correct, depends on correct D̃⁽¹⁾)
ω1 = solve_sgb_perturbation(sys_corr, ω0, v0, J)

# sGB background
H_per_order, a_powers = load_H_ratpolys_per_order(; verbose=true)
```

## Conventions

- `M = 1` everywhere. Table values are `ωM` (dimensionless).
- Use `--threads=8` for Julia. More threads cause OpenBLAS contention.
- TensorGR.jl at `../TensorGR.jl` — read-only dependency.
- QEP over Newton-Raphson. Always.
- See CLAUDE.md for blacklisted approaches and mandatory rules.
- **3 subagents before core code changes, reviewer after. No exceptions.**
