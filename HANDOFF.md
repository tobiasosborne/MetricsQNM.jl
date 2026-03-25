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

The QEP solver finds ALL eigenvalues at once, verified to 1e-14 on synthetic benchmarks.

## sGB extension — what needs to happen

**Goal**: Eq. 111 of 2406.11986: `x⁽¹⁾ = -J⁻¹ · [D̃⁽¹⁾(ω⁰) · v⁰]`

### Components ready

| Component | Status | File |
|-----------|--------|------|
| GR solution (ω⁰, v⁰, J) | DONE | `rectangular_qep.jl` |
| sGB background (H₁-H₄, ϑ) | DONE | `sgb_background.jl` |
| sGB linearization (abstract H-params) | DONE | `sgb_linearize.jl` |
| Eigenvalue perturbation solver | DONE | `sgb_perturbation.jl` |
| β=3 assembly extension | DONE | `assembly.jl` |

### The work that remains: symbolic `a` in SparsePoly

The paper keeps the spin parameter `a` symbolic throughout the entire extraction
pipeline. Our SparsePoly CAS currently hardcodes 5 variables (r, χ, ω_re, ω_im, iu)
and evaluates `a` numerically. This must change.

**Step 1: Extend SparsePoly to 6 variables**
- Add `a` as variable index 6 (or make N_VARS configurable)
- Update all NTuple{5,Int} → NTuple{6,Int} throughout sparse_poly.jl
- Update var_poly, differentiate, cleanup!, all arithmetic

**Step 2: Update SymToPolyCtx for symbolic `a`**
- Σ(r,χ,a) = r² + a²χ² as a 6-variable polynomial
- Δ(r,a) = r² - 2r + a² as a 6-variable polynomial
- Remove `a_val::Float64` parameter, replace with symbolic variable

**Step 3: Update denominator identification**
- `_identify_denom` must recognize Σ(r,χ,a)^n, Δ(r,a)^n
- `_factor_known_denoms` must work with 6-variable factor templates
- `_match_power` must compare 6-variable polynomials

**Step 4: Extract c_{k,d,p} with symbolic `a`**
- Remove the `Symbolics.substitute(c_expr, Dict(a_s => a))` hack
- Let symexpr_to_poly handle the full expression with `a` symbolic
- Result: RatPoly in (r, χ, ω_re, ω_im, iu, a)

**Step 5: Per-a-order decomposition of c_{k,d,p}**
- Collect terms by a-exponent (same technique as load_H_ratpolys_per_order)
- Each per-a-order c has moderate r-degree (~5-10, not ~24)

**Step 6: Multiply per-order c × H and assemble**
- For each a-order: multiply c^{(2k)} × H_p^{(2k)} in RatPoly arithmetic
- d_max per order stays moderate → spectral basis resolves at N=8-12
- Assemble D̃⁽¹⁾ per order via same Galerkin projection as GR
- Sum: D̃⁽¹⁾ = Σ a^{2k} D̃⁽¹,2k⁾

**Step 7: End-to-end test**
- With moderate d_max, ω₁ should converge (not O(10^10) garbage)
- Compare to Table I of 2406.11986

### Missing for full paper reproduction (after core pipeline works)

- Source 2: linearized sGB source tensor A_μ^ν
- Source 3: linearized scalar stress-energy T_μ^ν
- Source 4: Ω_H¹, κ¹ corrections in A_k
- 7th unknown h₇ (scalar field perturbation) + 11th equation

## Key function signatures

```julia
# GR pipeline (working)
sys, nf = build_system_bespoke(a, N, m)
result = solve_qep_with_vectors(sys; ω₀=ω_L)
J, free, pinned = compute_jacobian(sys, ω, v)

# sGB background (working)
bg = sgb_background(a; verbose=true)
hp = sgb_H_params(bg, r, χ)
H_per_order, a_powers = load_H_ratpolys_per_order(; verbose=true)

# sGB correction equations (working)
csc = compile_sgb_correction(2; verbose=true)

# sGB perturbation (working)
ω1 = solve_sgb_perturbation(sys_corr, ω0, v0, J)
```

## Conventions

- `M = 1` everywhere. Table values are `ωM` (dimensionless).
- Use `--threads=8` for Julia. More threads cause OpenBLAS contention.
- TensorGR.jl at `../TensorGR.jl` — read-only dependency.
- QEP over Newton-Raphson. Always.
- Rich diagnostics with flush after every print. Fail fast.
- See CLAUDE.md for the full rules and blacklisted approaches.
