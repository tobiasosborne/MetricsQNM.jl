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

## sGB symbolic pipeline — WORKING (Source 1)

**First ω₁ computed 2026-03-25** via fully symbolic pipeline (no numerical hacks).

### Pipeline (all on main branch):

| Step | Function | Status |
|------|----------|--------|
| SparsePoly CAS (6 vars, symbolic `a`) | `sparse_poly.jl` | DONE |
| GR per-a-order extraction | `extract_G_bespoke_symbolic_a` | DONE, verified 5.77e-16 |
| GR per-a-order assembly | `build_system_bespoke_sgb` | DONE, 13.1 digits |
| sGB c_{k,d,p} extraction | `extract_sgb_correction_symbolic_a` | DONE, 1165 triples |
| c × H convolution per-a-order | `build_sgb_Dtilde1` | DONE, net_δ approach |
| _r_to_z with negative δ | `_r_to_z` extended | DONE |
| Eigenvalue perturbation | `solve_sgb_perturbation` | DONE |

### Results (a=0.3, N=12, max_a_order=2, Source 1 only):

```
GR:  ω₀ = 0.419526681761 - 0.087729271894i  (|Δω_L| = 2.68e-12)
sGB: ω₁ = -1.47e-5 + 3.29e-7i  (|ω₁| = 1.47e-5)
```

Per-a-order δ range: [-6, +14], z-basis d_max=20 (was 32 before net_δ fix).

### Key technical achievements:

1. **SparsePoly extended to 6 variables** (r, χ, ω_re, ω_im, iu, a)
2. **Symbolic `a` throughout extraction** — no numerical substitution until final assembly
3. **Per-a-order convolution** of c_{k,d,p} × H_p avoids d_max inflation
4. **Net r-degree tracking** allows negative δ (from H_p denominators),
   handled by exact (1+z)^{|δ|} expansion in _r_to_z
5. **Built-in verification** confirms symbolic path matches numerical at 5.77e-16

## What remains

### Immediate (convergence validation):
- N-convergence study: sweep N=8,10,12,15,18 to verify ω₁ converges
- a-order convergence: sweep max_a_order=2,4,6,8 at fixed N
- Compare to Table I of 2406.11986 (partial: Source 1 only)

### Missing sources for full paper reproduction:
- Source 2: linearized sGB source tensor A_μ^ν (involves ∂²ϑ, curvature)
- Source 3: linearized scalar stress-energy T_μ^ν
- Source 4: Ω_H¹, κ¹ corrections in A_k asymptotic factor
- 7th unknown h₇ (scalar field perturbation) + 11th equation

## Key function signatures

```julia
# GR pipeline (working)
sys, nf = build_system_bespoke(a, N, m)
result = solve_qep_with_vectors(sys; ω₀=ω_L)
J, free, pinned = compute_jacobian(sys, ω, v)

# GR per-a-order (verified matches numerical at 5.77e-16)
sys, nf, disc = build_system_bespoke_sgb(a, N, m; verify=true, verbose=true)

# sGB correction extraction (symbolic a, 1165 non-zero triples)
c_kdp, a_powers = extract_sgb_correction_symbolic_a(; verbose=true)

# sGB D̃⁽¹⁾ assembly (per-a-order, net_δ with negative values)
sys_corr, nf = build_sgb_Dtilde1(a, N, m; max_a_order=2, verbose=true)

# Eigenvalue perturbation
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
