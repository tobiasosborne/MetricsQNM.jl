# METRICS-Julia: QNM Frequencies via TensorGR.jl

## Goal

Reproduce **Table I of arxiv:2312.08435** (Chung, Wagle, Yunes 2024) — the fundamental (n=0, l=2, m=2) quasi-normal mode frequencies of Kerr black holes for spins a = 0.005 to 0.95 — using a 100% open-source Julia implementation that leverages TensorGR.jl for symbolic tensor algebra.

## Success Criterion

Match the METRICS column of Table I to ~10 significant digits:

| a     | ω_Re (METRICS)  | ω_Im (METRICS)   | ω_Re (Leaver)       | ω_Im (Leaver)          | axial R           | polar R           |
|-------|-----------------|-------------------|----------------------|------------------------|-------------------|-------------------|
| 0.005 | 0.3743023147    | −0.0889522054     | 0.374302314745705    | −0.0889522053640457    | 8.02 × 10⁻¹¹     | 1.49 × 10⁻¹³     |
| 0.1   | 0.3870175384    | −0.0887056990     | 0.3870175383645592   | −0.0887056990268991    | 9.36 × 10⁻¹²     | 3.85 × 10⁻¹¹     |
| 0.2   | 0.4021453242    | −0.0883111662     | 0.4021453241072112   | −0.08831116615465      | 6.68 × 10⁻¹³     | 1.35 × 10⁻¹²     |
| 0.3   | 0.4195266818    | −0.0877292719     | 0.4195266799093153   | −0.0877292712328145    | 4.85 × 10⁻⁹      | 2.38 × 10⁻⁹      |
| 0.4   | 0.4398419217    | −0.0868819620     | 0.439841909727434    | −0.0868819580547294    | 9.32 × 10⁻¹¹     | 4.34 × 10⁻¹¹     |
| 0.5   | 0.4641230260    | −0.0856388350     | 0.4641229739649294   | −0.0856388194008764    | 6.21 × 10⁻¹⁰     | 1.62 × 10⁻¹⁰     |
| 0.6   | 0.4940447818    | −0.0837652022     | 0.4940446109217166   | −0.0837651572095065    | 1.33 × 10⁻⁹      | 1.59 × 10⁻¹⁰     |
| 0.7   | 0.5326002436    | −0.0807928732     | 0.5325997998444519   | −0.0807927741196761    | 5.00 × 10⁻¹¹     | 1.31 × 10⁻¹¹     |
| 0.8   | 0.5860169749    | −0.0756295524     | 0.5860160981862801   | −0.07562938913772186   | 1.89 × 10⁻⁹      | 1.36 × 10⁻⁹      |
| 0.9   | 0.6716142721    | −0.0648692359     | 0.671613259501218    | −0.06486906741255006   | 3.31 × 10⁻¹¹     | 1.01 × 10⁻¹¹     |
| 0.95  | 0.7463199985    | −0.0531490080     | 0.7463194371599231   | −0.05314891507283093   | 4.72 × 10⁻¹¹     | 8.34 × 10⁻¹¹     |

(Source: Table I of `papers/source/main.tex`, lines 925–950)

---

## Papers & Local Source Files

| Paper | Topic | PDF | TeX source |
|-------|-------|-----|------------|
| arxiv:2312.08435 | METRICS for Kerr QNMs in GR | `papers/2312.08435.pdf` | `papers/source/main.tex` |
| arxiv:2406.11986 | METRICS for sGB beyond-GR QNMs | `papers/2406.11986.pdf` | `papers/2406.11986_src/main.tex` |

Supplementary Mathematica notebook with background metric to 40th order in spin: `papers/2406.11986_src/Supplementary_materials.nb`

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│  Phase 1: SYMBOLIC (TensorGR.jl + Symbolics.jl)             │
│  Kerr metric → linearize Einstein eqs → extract G_{...}     │
│  coefficient arrays as rational functions of (r, χ, ω)       │
└───────────────────────┬──────────────────────────────────────┘
                        │ exported G coefficient tables
┌───────────────────────▼──────────────────────────────────────┐
│  Phase 2: SPECTRAL PROJECTION (new package: METRICS.jl)      │
│  G coefficients × Chebyshev/Legendre inner products          │
│  → assemble D̃₀, D̃₁, D̃₂ constant rectangular matrices      │
└───────────────────────┬──────────────────────────────────────┘
                        │ constant matrices (depend on m, a)
┌───────────────────────▼──────────────────────────────────────┐
│  Phase 3: NEWTON-RAPHSON EIGENVALUE SOLVE                    │
│  D̃(ω)v = 0 via Newton-Raphson + Moore-Penrose pseudoinverse │
│  → ω_QNM                                                     │
└──────────────────────────────────────────────────────────────┘
```

---

## Dependencies (All Open-Source Julia)

| Package | Purpose | Notes |
|---------|---------|-------|
| **TensorGR.jl** | Symbolic tensor algebra: Kerr metric, linearize Einstein eqs, curvature tensors | Local, at `../../TensorGR.jl` |
| **Symbolics.jl** (v7.16+) | CAS for scalar expressions, polynomial decomposition, code generation | Core symbolic engine |
| **ClassicalOrthogonalPolynomials.jl** (v0.15+) | Chebyshev T_n, associated Legendre P_l^m evaluation and recurrences | Spectral basis |
| **FastTransforms.jl** (v0.17+) | Fast polynomial transforms, coefficient-space operations | Performance for large N |
| **LinearAlgebra** (stdlib) | `pinv()` for Moore-Penrose pseudoinverse, `norm()`, `eigen()` | Newton-Raphson core |
| **GenericSchur.jl** (v0.5+) | Eigenvalue decomposition for BigFloat/extended precision | High-precision validation |
| **MultiFloats.jl** | Double64/Float64x4 for extended precision at moderate cost | Alternative to BigFloat |
| **SpinWeightedSpheroidalHarmonics.jl** (v1.3+) | Angular Teukolsky eigenvalues, validation | Independent check |
| **QuasinormalModes.jl** (v1.1+) | Asymptotic Iteration Method QNM solver | Independent validation method |
| **JLD2.jl** | Serialization of coefficient tables | Cache symbolic results |

---

## Package Structure

```
grstuff/
├── METRICS.jl/                    # new Julia package
│   ├── Project.toml
│   ├── src/
│   │   ├── METRICS.jl             # module root, exports
│   │   ├── kerr.jl                # Kerr metric quantities (Σ, Δ, r±, b, Ω_H, r_*)
│   │   ├── perturbation_ansatz.jl # h_k Regge-Wheeler decomposition, A_k(r) factors
│   │   ├── spectral.jl            # Chebyshev/Legendre basis, products, derivative identities
│   │   ├── linearize.jl           # Interface to TensorGR.jl: δΓ, δR_μ^ν on Kerr
│   │   ├── coefficients.jl        # Extract G_{k,γ,δ,σ,α,β,j} polynomial structure
│   │   ├── assembly.jl            # Build D̃₀, D̃₁, D̃₂ matrices from G coefficients
│   │   ├── newton.jl              # Newton-Raphson solver with Moore-Penrose pseudoinverse
│   │   └── leaver.jl              # Leaver continued-fraction for independent validation
│   └── test/
│       ├── runtests.jl
│       ├── test_kerr.jl           # Kerr metric quantities, Ricci = 0 check
│       ├── test_spectral.jl       # Orthogonality, recurrences, inner products
│       ├── test_leaver.jl         # Leaver QNM frequencies vs known values
│       ├── test_schwarzschild.jl  # a→0 limit reproduces Zerilli/Regge-Wheeler
│       └── test_table1.jl         # Full Table I reproduction
├── papers/                        # downloaded arxiv papers (already populated)
│   ├── 2312.08435.pdf
│   ├── source/main.tex
│   ├── 2406.11986.pdf
│   └── 2406.11986_src/main.tex
├── notebooks/                     # Pluto notebooks for exploration
└── plan.md                        # this file
```

---

## Phase 0: Validation Infrastructure

### 0.1 Kerr Metric Quantities

Numerical functions for the Kerr background. Convention: M = 1 throughout (as in the paper).

```julia
struct KerrParams
    a::Float64    # dimensionless spin, 0 ≤ a < 1
    m::Int        # azimuthal mode number (m = 2 for 022 mode)
end

# Derived quantities (Eq. 2 of 2312.08435):
b(a)    = sqrt(1 - a^2)
r_plus(a)  = 1 + b(a)          # M = 1
r_minus(a) = 1 - b(a)
Omega_H(a) = a / (2 * r_plus(a))

# Metric functions of (r, χ):
Sigma(r, χ, a) = r^2 + a^2 * χ^2
Delta(r, a)    = (r - r_plus(a)) * (r - r_minus(a))
```

**Validation:** Ricci tensor R_μν computed from the Kerr metric must vanish identically (vacuum solution). Use TensorGR.jl's `symbolic_ricci()` to verify.

### 0.2 Leaver Continued-Fraction Solver

Independent QNM frequency oracle. Leaver's method (1985) solves the radial Teukolsky equation via a three-term recurrence relation. The QNM frequencies are zeros of an infinite continued fraction, truncated at ~150 terms.

**Algorithm:**
1. Compute angular separation constant A_lm(aω) via spheroidal harmonic eigenvalue (use SpinWeightedSpheroidalHarmonics.jl)
2. Set up three-term recurrence coefficients α_n, β_n, γ_n for the radial Teukolsky equation
3. Evaluate the continued fraction: β₀ - α₀γ₁/(β₁ - α₁γ₂/(β₂ - ...))
4. Find ω such that the continued fraction equals zero (root-finding with Newton's method)
5. Use BigFloat arithmetic for 16-digit precision

**Validation:** Reproduce Leaver column of Table I to all displayed digits.

---

## Phase 1: Symbolic Linearization via TensorGR.jl

This is the most TensorGR.jl-intensive phase. The output is the coefficient array G_{k,γ,δ,σ,α,β,j} from Eq. (22) of the paper.

### 1.1 Kerr Metric in Boyer-Lindquist Coordinates with χ = cos(θ)

**What TensorGR.jl already provides:**
- `define_metric!`, `define_curvature_tensors!`, `define_covd!`
- `symbolic_christoffel()`, `symbolic_riemann()`, `symbolic_ricci()`
- `SymbolicMetric` for component-level computation
- `TensorGRSymbolicsExt` extension for CAS scalar manipulation

**Implementation:**

Define the Kerr metric as a 4×4 symbolic matrix in coordinates x^μ = (t, r, χ, φ) using Symbolics.jl:

```julia
using Symbolics
@variables r χ a_sym  # M = 1 throughout

Σ_sym = r^2 + a_sym^2 * χ^2
Δ_sym = r^2 - 2r + a_sym^2

# Eq. 1 of 2312.08435:
g = zeros(Num, 4, 4)
g[1,1] = -(1 - 2r/Σ_sym)                           # g_tt
g[1,4] = g[4,1] = -2a_sym*r*(1 - χ^2)/Σ_sym        # g_tφ
g[2,2] = Σ_sym / Δ_sym                               # g_rr
g[3,3] = Σ_sym / (1 - χ^2)                           # g_χχ
g[4,4] = (r^2 + a_sym^2 + 2a_sym^2*r*(1-χ^2)/Σ_sym) * (1 - χ^2)  # g_φφ
```

Then wrap in TensorGR.jl's `SymbolicMetric` and compute:
```julia
Γ = symbolic_christoffel(g, [r, χ])  # 4×4×4 array of Symbolics expressions
```

**Validation:** Compute `symbolic_ricci(g, ...)` and verify all 10 independent components simplify to zero.

### 1.2 Metric Perturbation Ansatz (Regge-Wheeler Gauge)

The 6 unknown functions h_k(r, χ) are arranged in 4×4 matrices (Eqs. 8a, 8b of 2312.08435). After factoring out the common e^{imφ − iωt} phase:

**Odd (axial) sector** (Eq. 8a, `papers/source/main.tex` line 308):
```
h^odd_μν = e^{imφ-iωt} ×
⎛ 0    0    −im/(1−χ²) h₅     (1−χ²) ∂_χ h₅ ⎞
⎜ *    0    −im/(1−χ²) h₆     (1−χ²) ∂_χ h₆ ⎟
⎜ *    *     0                  0              ⎟
⎝ *    *     *                  0              ⎠
```

**Even (polar) sector** (Eq. 8b, `papers/source/main.tex` line 321):
```
h^even_μν = −e^{imφ-iωt} ×
⎛ h₁    h₂     0                0              ⎞
⎜ *     h₃     0                0              ⎟
⎜ *     *      h₄/(1−χ²)        0              ⎟
⎝ *     *      *            (1−χ²) h₄          ⎠
```

**Implementation:**
```julia
@variables ω_sym
@variables h1(r, χ) h2(r, χ) h3(r, χ) h4(r, χ) h5(r, χ) h6(r, χ)

function regge_wheeler_perturbation(r, χ, m_mode)
    # Returns 4×4 symbolic matrix (without the e^{imφ-iωt} factor)
    # Derivatives ∂_χ h₅, ∂_χ h₆ expressed via Symbolics.Differential
    ...
end
```

### 1.3 Linearize the Trace-Reversed Einstein Equations

Substitute g_μν = g^(0)_μν + ε h_μν into R_μ^ν = 0 and linearize in ε.

**Component-level approach** (preferred for correctness and directness):

**Step 1 — Perturbed Christoffel symbols** (standard GR formula):
```
δΓ^α_βγ = (1/2) g^{(0)αδ} (∇_β h_{γδ} + ∇_γ h_{βδ} − ∇_δ h_{βγ})
```
where ∇ is the background covariant derivative. In components:
```
δΓ^α_βγ = (1/2) g^{(0)αδ} (∂_β h_{γδ} + ∂_γ h_{βδ} − ∂_δ h_{βγ}
           − 2 Γ^{(0)ε}_{βγ} h_{εδ}
           + Γ^{(0)ε}_{βδ} h_{εγ} + Γ^{(0)ε}_{γδ} h_{εβ})
```

Wait — more precisely, in components with the background covariant derivative:
```
δΓ^α_βγ = (1/2) g^{(0)αδ} (∂_β h_{γδ} + ∂_γ h_{βδ} − ∂_δ h_{βγ})
```
because the connection-dependent terms cancel when using the coordinate basis expression for δΓ directly. This is the Palatini identity in coordinate components.

**Step 2 — Linearized Ricci tensor:**
```
δR_μν = ∂_α δΓ^α_μν − ∂_μ δΓ^α_αν
        + Γ^{(0)α}_{αβ} δΓ^β_μν + δΓ^α_{αβ} Γ^{(0)β}_μν
        − Γ^{(0)α}_{μβ} δΓ^β_αν − δΓ^α_{μβ} Γ^{(0)β}_αν
```

**Step 3 — Trace-reversed form:**
```
δR_μ^ν = g^{(0)να} δR_μα − h^{να} R^{(0)}_μα
```
The second term vanishes because R^{(0)}_μα = 0 (Kerr is Ricci-flat).

So: δR_μ^ν = g^{(0)να} δR_μα, giving 10 independent equations (4×4 symmetric, but δR_μ^ν is not symmetric — we get all 16 components, of which 10 are independent due to the contracted Bianchi identity).

**TensorGR.jl role:** Use `symbolic_christoffel()` for background Γ, and compute δΓ and δR by systematic component-level evaluation in Symbolics.jl. TensorGR.jl's existing component infrastructure (`symbolic_christoffel`, `symbolic_riemann`, etc.) provides the building blocks. The key new work is the linearization pipeline.

**New TensorGR.jl functionality to add:**
```julia
# In src/components/linearized_curvature.jl (new file)
function linearized_christoffel(g_bg, g_inv_bg, Gamma_bg, h_pert, coords)
    # Returns δΓ^α_βγ as 4×4×4 Symbolics array
end

function linearized_ricci(g_bg, g_inv_bg, Gamma_bg, h_pert, coords)
    # Returns δR_μν as 4×4 Symbolics array
end

function linearized_ricci_mixed(g_inv_bg, delta_R_cov)
    # Returns δR_μ^ν = g^{(0)να} δR_μα as 4×4 Symbolics array
end
```

### 1.4 Factor Out Time and Azimuthal Dependence

Substitute the Regge-Wheeler ansatz h_μν = e^{imφ − iωt} × H_μν(r, χ) into the 10 linearized equations. Every ∂_t acts on the exponential giving a factor −iω, and every ∂_φ gives a factor im. The exponential cancels from all equations.

Result: 10 PDEs in two variables (r, χ) with ω as a parameter, for the 6 unknown functions h_k(r, χ).

### 1.5 Extract Polynomial Coefficient Structure

Each PDE has the form (Eq. 22 of the paper, `papers/source/main.tex` line 477):
```
∑_{j=1}^{6} ∑_{α+β≤3} ∑_{γ=0}^{2} ∑_{δ=0}^{d_r} ∑_{σ=0}^{d_χ}
  G_{k,γ,δ,σ,α,β,j} · ω^γ · r^δ · χ^σ · ∂_r^α ∂_χ^β h_j = 0
```

where:
- k ∈ {1,...,10} indexes the equation
- j ∈ {1,...,6} indexes which h_j function
- α, β are derivative orders (α + β ≤ 3)
- γ ∈ {0, 1, 2} is the power of ω
- δ, σ are powers of r and χ
- G_{k,γ,δ,σ,α,β,j} depends only on M(=1), m, and a

**Key challenge:** Each equation has "several thousand terms" (paper, paragraph below Eq. 22). The largest coefficient modulus varies by ~10 orders of magnitude across equations and ~20 orders within a single equation.

**Implementation strategy:**

1. For each of the 10 equations, start from the symbolic expression δR_μ^ν(r, χ, ω, h_k, ∂h_k)
2. Multiply through by the common denominator (powers of Σ, Δ, (1−χ²)) to make all coefficients polynomial in r, χ
3. For each (j, α, β) triple, collect the coefficient of ∂_r^α ∂_χ^β h_j
4. Decompose that coefficient as a polynomial in (ω, r, χ) to extract G_{k,γ,δ,σ,α,β,j}

**New functionality:**
```julia
# In src/coefficients.jl (new file in METRICS.jl)
struct PDECoefficients
    # G[k][(γ,δ,σ,α,β,j)] = Complex{Rational{BigInt}} or Float64
    equations::Vector{Dict{NTuple{6,Int}, ComplexF64}}
    d_r::Vector{Int}     # max power of r in each equation
    d_chi::Vector{Int}   # max power of χ in each equation
end

function extract_pde_coefficients(linearized_eqs, h_functions, r, χ, ω)
    # Uses Symbolics.jl polynomial decomposition
    # Returns PDECoefficients
end
```

### 1.6 Normalize Equations

Divide each equation by its largest |G| coefficient so max modulus = 1:
```julia
for k in 1:10
    max_G = maximum(abs, values(coeffs.equations[k]))
    for key in keys(coeffs.equations[k])
        coeffs.equations[k][key] /= max_G
    end
end
```

This prevents floating-point overflow (paper notes coefficients span ~20 orders of magnitude within one equation).

### 1.7 Cache and Serialize

The G coefficients depend on (m, a) only (not on N, the spectral truncation order). Once computed for a given spin value, cache to disk via JLD2:

```julia
using JLD2
@save "coefficients_m2_a$(a).jld2" pde_coefficients
```

For the symbolic derivation (which is expensive), consider keeping a as symbolic and evaluating numerically later. This gives one symbolic computation for all spin values.

### 1.8 Schwarzschild Validation

At a = 0, the Kerr metric reduces to Schwarzschild, and the 10 coupled PDEs decouple into:
- The Zerilli equation (polar/even sector, h₁–h₄)
- The Regge-Wheeler equation (axial/odd sector, h₅–h₆)

**Validation:** Set a = 0 in the G coefficients and verify the resulting equations match the known Zerilli and Regge-Wheeler forms. This is a critical correctness check before proceeding to Phase 2.

---

## Phase 2: Spectral Projection

### 2.1 Asymptotic Controlling Factor A_k(r)

From Eq. (15) of the paper (`papers/source/main.tex` line 399):
```
A_k(r) = e^{iωr} · r^{2iMω + ρ_∞^(k)} · ((r − r₊)/r)^{−iM(ω − mΩ_H)(1+b)/b − ρ_H^(k)}
```

With ρ values from Eq. (16) (`papers/source/main.tex` line 407):
```
ρ_H^(k) = 2  for k = 3
           1  for k = 2 or 6
           0  otherwise

ρ_∞^(k) = 2  for k ≠ 4
           1  for k = 4
```

**Note:** A_k depends on ω, which is the unknown. The ω-dependence propagates into the D̃ matrices.

**Implementation:**
```julia
function rho_H(k::Int)
    k == 3 && return 2
    (k == 2 || k == 6) && return 1
    return 0
end

function rho_inf(k::Int)
    k == 4 && return 1
    return 2
end

function asymptotic_factor(r, k, ω, params::KerrParams)
    a = params.a; m = params.m
    rp = r_plus(a)
    b_val = b(a)
    ΩH = Omega_H(a)

    exp(im*ω*r) * r^(2im*ω + rho_inf(k)) *
        ((r - rp)/r)^(-im*(ω - m*ΩH)*(1 + b_val)/b_val - rho_H(k))
end
```

### 2.2 Compactified Radial Coordinate

From Eq. (17) (`papers/source/main.tex` line 427):
```
z(r) = 2r₊/r − 1,    mapping r ∈ [r₊, ∞) → z ∈ [−1, +1]
```

Inverse: r(z) = 2r₊/(1 + z)

Derivative chain rule:
```
∂/∂r = −(1+z)²/(2r₊) · ∂/∂z
```

Higher derivatives via repeated application. This transforms the G coefficients (functions of r) into K coefficients (functions of z):

```
∑ K_{k,γ,δ,σ,α,β,j} · ω^γ · z^δ · χ^σ · ∂_z^α ∂_χ^β u_j = 0
```

(Eq. 26 of the paper, `papers/source/main.tex` line 517)

**Implementation:**
```julia
function transform_r_to_z(pde_coefficients::PDECoefficients, rp::Float64)
    # Substitutes r = 2rp/(1+z) into the G coefficients
    # Converts ∂_r^α → expressions in ∂_z^α using chain rule
    # Returns new PDECoefficients in (z, χ) variables
end
```

### 2.3 Factor Out A_k and Simplify

After substituting h_k = A_k · u_k, the asymptotic factor A_k and its derivatives contribute additional rational factors. Since A_k is never zero in the computational domain (z ∈ (−1, +1)), divide it out. Also divide out common factors like Δ, (1−χ²), Σ that are nonzero in the interior.

The result is 10 PDEs for the bounded correction functions u_k(z, χ).

### 2.4 Chebyshev and Associated Legendre Basis

**Spectral expansion** (Eq. 20, `papers/source/main.tex` line 432):
```
u_k(z, χ) = ∑_{n=0}^{N_z} ∑_{l=|m|}^{N_χ+|m|} v_k^{nl} T_n(z) P_l^{|m|}(χ)
```

where T_n are Chebyshev polynomials of the first kind and P_l^{|m|} are associated Legendre polynomials. In practice, N_z = N_χ = N ("along the diagonal").

**Key identities for eliminating second derivatives** (Eq. 29, `papers/source/main.tex` line 555):
```
d²T_n/dz² = (z · dT_n/dz − n² · T_n) / (1 − z²)

d²P_l^{|m|}/dχ² = [2χ · dP_l^{|m|}/dχ − l(l+1) · P_l^{|m|} − m²/(1−χ²) · P_l^{|m|}] / (1 − χ²)
```

These eliminate all second (and higher) derivatives, reducing the spectral projection to products of the form z^δ · T_n(z) and χ^σ · P_l^m(χ), which are computable via **linearization coefficients**:

**Chebyshev product rule:**
```
z · T_n(z) = (T_{n+1}(z) + T_{n−1}(z)) / 2
```
and by induction:
```
z^δ · T_n(z) = ∑_{n'} c^{(δ)}_{n,n'} T_{n'}(z)
```

**Associated Legendre product rule:**
```
χ · P_l^m(χ) = [(l−m+1)P_{l+1}^m + (l+m)P_{l−1}^m] / (2l+1)
```
and by induction:
```
χ^σ · P_l^m(χ) = ∑_{l'} d^{(σ,m)}_{l,l'} P_{l'}^m(χ)
```

**First derivatives** via recurrence relations:
```
dT_n/dz: expressible as sum of T_{n'} via Chebyshev derivative formula
dP_l^m/dχ: expressible as sum of P_{l'}^m via associated Legendre recurrences
```

**Implementation using ClassicalOrthogonalPolynomials.jl:**
```julia
using ClassicalOrthogonalPolynomials

struct SpectralBasis
    N::Int               # truncation order
    m::Int               # azimuthal mode number
    # Precomputed linearization coefficient matrices:
    z_product::Matrix{Float64}   # z^δ · T_n → ∑ c T_{n'}
    chi_product::Matrix{Float64} # χ^σ · P_l^m → ∑ d P_{l'}^m
    dT_coeffs::Matrix{Float64}   # dT_n/dz → ∑ c' T_{n'}
    dP_coeffs::Matrix{Float64}   # dP_l^m/dχ → ∑ d' P_{l'}^m
end
```

### 2.5 Spectral Inner Products → D Block Matrices

Substituting the spectral expansion into the PDEs and using the linearization coefficients, each PDE becomes (Eq. 30, `papers/source/main.tex` line 565):
```
∑_n ∑_l w_k^{nl} T_n(z) P_l^{|m|}(χ) = 0
```

By orthogonality of T_n and P_l^m, each coefficient w_k^{nl} = 0. These are linear in the spectral coefficients v_j^{n'l'} (Eq. 31, `papers/source/main.tex` line 572):
```
w_k^{nl} = ∑_{j=1}^{6} ∑_{n'} ∑_{l'} [D_{nl,n'l'}(ω)]_{kj} · v_j^{n'l'} = 0
```

where D_{nl,n'l'} are **10×6 matrices** whose elements are at most **quadratic polynomials in ω**.

**How to compute D_{nl,n'l'}:**

For each term G_{k,γ,δ,σ,α,β,j} · ω^γ · z^δ · χ^σ · ∂_z^α ∂_χ^β (T_{n'} P_{l'}^m):
1. Apply ∂_z^α to T_{n'}: get a linear combination of T_{n''} (using derivative + second-derivative-elimination identities)
2. Multiply by z^δ: get a linear combination of T_{n'''} (using product linearization)
3. Apply ∂_χ^β to P_{l'}^m: get a linear combination of P_{l''}^m
4. Multiply by χ^σ: get a linear combination of P_{l'''}^m
5. Read off the coefficient of T_n P_l^m: this is the contribution to [D_{nl,n'l'}]_{kj}
6. The ω^γ factor means this contribution goes into D₀ (γ=0), D₁ (γ=1), or D₂ (γ=2)

**Implementation:**
```julia
function compute_D_blocks(K_coefficients::PDECoefficients, basis::SpectralBasis, ω)
    N = basis.N; m_mode = basis.m
    n_eqs = 10; n_unknowns = 6
    block_size = (N + 1) * (N + 1)  # number of (n, l) pairs

    D0 = zeros(ComplexF64, n_eqs * block_size, n_unknowns * block_size)
    D1 = zeros(ComplexF64, n_eqs * block_size, n_unknowns * block_size)
    D2 = zeros(ComplexF64, n_eqs * block_size, n_unknowns * block_size)

    for (k, γ, δ, σ, α, β, j), G_val in K_coefficients
        # ... compute contribution to D0/D1/D2 using linearization coefficients
    end

    return D0, D1, D2
end
```

### 2.6 Assemble Global D̃ Matrices

Stack the D block matrices into the global system (Eq. 35, `papers/source/main.tex` line 600):
```
w = D̃(ω) v = [D̃₀ + D̃₁ω + D̃₂ω²] v = 0
```

**Matrix dimensions** for N = 30:
- Rows: 10 × (30+1)² = 10 × 961 = 9,610
- Columns: 6 × (30+1)² = 6 × 961 = 5,766
- Rectangular (more equations than unknowns)

D̃₀, D̃₁, D̃₂ are **constant** (independent of ω) rectangular matrices.

---

## Phase 3: Newton-Raphson QNM Solver

### 3.1 Normalization Convention

To break the trivial/scaling degeneracy of the homogeneous system D̃(ω)v = 0:

**Polar-led** (Eq. 36, `papers/source/main.tex` line 633):
```
v_{k=1}^{n=0, l=|m|} = 1
```
Initial guess: all v_{k=5,6} = 0 (axial sector starts at zero).

**Axial-led** (Eq. 46):
```
v_{k=5}^{n=0, l=|m|} = 1
```
Initial guess: all v_{k=1,...,4} = 0 (polar sector starts at zero).

After fixing one coefficient, the unknowns are:
```
x = {remaining 6(N+1)² − 1 spectral coefficients, ω}
```
Total: 6(N+1)² unknowns (the fixed v is removed, ω is added).

### 3.2 Residual Vector f(x)

With the convention enforced (Eq. 38, `papers/source/main.tex` line 643):
```
f_k(x) = [∑_j ∑_{n'} ∑_{l'} D_{nl,n'l'}(ω) v_j^{n'l'}]_{v_fixed = 1}
```

This is a 10(N+1)²-vector.

### 3.3 Jacobian Matrix J(x)

The Jacobian is 10(N+1)² × 6(N+1)², with two blocks:

**Derivative w.r.t. spectral coefficients v:**
```
∂f_i/∂v_j^{n'l'} = [D̃(ω)]_{i, index(j,n',l')}
```
(just the columns of D̃ corresponding to the free v components)

**Derivative w.r.t. ω:**
```
∂f_i/∂ω = [(D̃₁ + 2D̃₂ω) v]_i
```

**Implementation:**
```julia
function residual_and_jacobian(sys::METRICSSystem, x::AbstractVector{ComplexF64};
                                parity::Symbol=:polar)
    ω = x[end]
    v = reconstruct_full_v(x[1:end-1], sys.N, parity)  # insert fixed coeff

    D = sys.D0 + sys.D1 * ω + sys.D2 * ω^2
    f = D * v  # residual: 10(N+1)²-vector

    # Jacobian columns for free v:
    free_cols = free_column_indices(sys.N, parity)
    J_v = D[:, free_cols]

    # Jacobian column for ω:
    J_ω = (sys.D1 + 2 * sys.D2 * ω) * v

    J = hcat(J_v, J_ω)  # 10(N+1)² × 6(N+1)²
    return f, J
end
```

### 3.4 Newton-Raphson Iteration

From Eq. (44) of the paper (`papers/source/main.tex` line 676):
```
x_{n+1} = x_n − J⁻¹ · f(x_n)
```

where J⁻¹ is the **Moore-Penrose pseudoinverse** (since J is rectangular: more rows than columns).

**Convergence criterion** (Eq. 45, `papers/source/main.tex` line 683):
```
‖f(x_n)‖₂ < ε = 10⁻⁷
```

Maximum 20 iterations per solve.

**Implementation:**
```julia
function solve_qnm(sys::METRICSSystem, ω_guess::ComplexF64;
                    parity::Symbol = :polar,
                    tol::Float64 = 1e-7,
                    maxiter::Int = 20)
    x = initial_guess(sys, ω_guess, parity)
    residual_initial = NaN

    for iter in 1:maxiter
        f, J = residual_and_jacobian(sys, x; parity)

        residual_norm = norm(f)
        if iter == 1
            residual_initial = residual_norm
        end

        if residual_norm < tol
            ω_result = x[end]
            R = residual_norm / residual_initial  # residual ratio
            return (ω = ω_result,
                    coefficients = x[1:end-1],
                    iterations = iter,
                    residual = residual_norm,
                    residual_ratio = R)
        end

        # Moore-Penrose pseudoinverse update
        x = x - pinv(J) * f
    end
    error("Newton-Raphson did not converge in $maxiter iterations")
end
```

### 3.5 Initial Guess Strategy

The paper uses the first two significant digits of the known QNM frequency:

1. **From Leaver oracle:** Use our Phase 0 Leaver solver to get ω_Leaver at each spin, then round to 2 significant digits as the initial guess.
2. **Bootstrapping:** Start at small spin (a = 0.005), solve, then use that solution as initial guess for the next spin value.
3. **Spectral coefficients:** Initialize the leading-parity v to have only v_{k}^{0,|m|} = 1, all others zero. The opposite parity starts at zero.

### 3.6 Full Computation for Table I

```julia
function reproduce_table_1(; N=30, m=2, tol=1e-7)
    spins = [0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    results = []

    for a in spins
        # Build system for this spin
        K = compute_K_coefficients(a, m)       # Phase 1 output
        D0, D1, D2 = assemble_D_matrices(K, N) # Phase 2 output
        sys = METRICSSystem(D0, D1, D2, N, m, a)

        # Get initial guess from Leaver
        ω_leaver = leaver_qnm(a, l=2, m=2, n=0)
        ω_guess = round(ω_leaver, sigdigits=2)

        # Solve both parities
        result_polar = solve_qnm(sys, ω_guess; parity=:polar, tol)
        result_axial = solve_qnm(sys, ω_guess; parity=:axial, tol)

        # Average (isospectrality check)
        ω_avg = (result_polar.ω + result_axial.ω) / 2

        push!(results, (a=a, ω_polar=result_polar.ω, ω_axial=result_axial.ω,
                        ω_avg=ω_avg, ω_leaver=ω_leaver,
                        R_polar=result_polar.residual_ratio,
                        R_axial=result_axial.residual_ratio))
    end
    return results
end
```

**Isospectrality check:** |ω_polar − ω_axial| should be small (vanishes in exact arithmetic for GR Kerr).

---

## Phase 4: New TensorGR.jl Functionality

Summary of extensions needed in TensorGR.jl to support this project and make it reusable for future beyond-GR work.

### 4.1 Component-Level Linearized Curvature (`src/components/linearized_curvature.jl`)

```julia
"""
    linearized_christoffel(g_inv, Gamma, h, coords) → Array{Num, 3}

Compute δΓ^α_βγ from background metric inverse, Christoffel symbols,
and metric perturbation h_μν, all as Symbolics.jl arrays.
"""
function linearized_christoffel(g_inv, Gamma, h, coords)
    dim = size(g_inv, 1)
    δΓ = Array{Num}(undef, dim, dim, dim)
    D = Symbolics.Differential
    for α in 1:dim, β in 1:dim, γ in 1:dim
        δΓ[α,β,γ] = sum(g_inv[α,δ] * (
            D(coords[β])(h[γ,δ]) + D(coords[γ])(h[β,δ]) - D(coords[δ])(h[β,γ])
        ) for δ in 1:dim) / 2
    end
    return δΓ
end

"""
    linearized_ricci(Gamma, δΓ, coords) → Array{Num, 2}

Compute δR_μν from background Christoffel Γ and perturbed Christoffel δΓ.
"""
function linearized_ricci(Gamma, δΓ, coords)
    dim = size(Gamma, 1)
    D = Symbolics.Differential
    δR = Array{Num}(undef, dim, dim)
    for μ in 1:dim, ν in 1:dim
        δR[μ,ν] = (
            sum(D(coords[α])(δΓ[α,μ,ν]) for α in 1:dim)
          - sum(D(coords[μ])(δΓ[α,α,ν]) for α in 1:dim)
          + sum(Gamma[α,α,β]*δΓ[β,μ,ν] + δΓ[α,α,β]*Gamma[β,μ,ν]
              - Gamma[α,μ,β]*δΓ[β,α,ν] - δΓ[α,μ,β]*Gamma[β,α,ν]
                for α in 1:dim, β in 1:dim)
        )
    end
    return δR
end
```

### 4.2 Kerr Metric Convenience Module (`src/metrics/kerr.jl`)

Pre-built Kerr metric in various coordinate systems with all curvature quantities pre-computed and simplified.

### 4.3 Regge-Wheeler Gauge Ansatz Builder (`src/perturbations/regge_wheeler.jl`)

Given coordinate system and mode numbers (l, m), construct the structured h_μν perturbation matrix with the correct angular dependence factored out.

### 4.4 PDE Polynomial Coefficient Extractor (`src/spectral/pde_coefficients.jl`)

Generic tool for decomposing a PDE system into the multi-index coefficient structure needed for spectral methods.

### 4.5 Gauss-Bonnet Scalar on Curved Background

For the beyond-GR extension:
```julia
function gauss_bonnet_scalar(Riemann, Ricci, RicciScalar, g, g_inv)
    # G = R² − 4 R_{ab}R^{ab} + R_{abcd}R^{abcd}
    # TensorGR.jl already has curvature_invariant("Kretschner") etc.
end
```

---

## Phase 5: Beyond-GR Extension (arxiv:2406.11986)

Once Table I is reproduced in GR, extend to **scalar-Gauss-Bonnet (sGB) gravity**.

### 5.1 Theory

Lagrangian density (Eq. 1 of 2406.11986, `papers/2406.11986_src/main.tex`):
```
16π L = R − (1/2) ∇_μΦ ∇^μΦ + α Φ G
```
where G = R² − 4R_{ab}R^{ab} + R_{abcd}R^{abcd} is the Gauss-Bonnet invariant.

Coupling parameter: ζ = α²/M⁴ ≪ 1.

### 5.2 Background Metric Corrections

The Kerr metric gets O(ζ) corrections parameterized by four functions H₁,...,H₄(r,χ) (Eq. 16 of 2406.11986). These are computed as power series in spin a to 40th order. The explicit series are in the supplementary Mathematica notebook.

**Strategy:** Parse the Mathematica expressions from `papers/2406.11986_src/Supplementary_materials.nb` and convert to Julia/Symbolics.jl format.

### 5.3 Background Scalar Field ϑ

Solve □ϑ + G = 0 on the GR Kerr background, order by order in spin a. The solution is also in the supplementary notebook.

### 5.4 Seven Perturbation Functions

The system grows from 6 to 7 unknowns: h₁,...,h₆ (metric) plus h₇ = Φ (scalar field). The field equations grow from 10 to 11 (10 tensor + 1 scalar).

### 5.5 Modified Field Equations

The linearized equations take the form (Eq. 93 of 2406.11986):
```
∑_{j=1}^{7} ∑_{η=0}^{1} ∑_{α+β≤3} ∑_{γ≤2} ∑_δ ∑_σ
  G_{k,η,γ,δ,σ,α,β,j} · ζ^η · ω^γ · r^δ · χ^σ · ∂_r^α ∂_χ^β h_j = 0
```

The key difference from GR: j runs to 7 (not 6), η runs to 1 (ζ⁰ = GR, ζ¹ = sGB correction).

### 5.6 Eigenvalue Perturbation Theory

The central methodological advance of the second paper. Expand in ζ:
```
D̃ = D̃^(0) + ζ D̃^(1)
ω = ω^(0) + ζ ω^(1)
v = v^(0) + ζ v^(1)
```

At zeroth order: solve the GR problem (Phase 3 above).

At first order (Eq. 111 of 2406.11986):
```
x^(1) = −J⁻¹ · [D̃^(1)(ω^(0)) · v^(0)]
```

where J⁻¹ is the **same** Moore-Penrose pseudoinverse already computed during the GR Newton-Raphson solve. The beyond-GR correction requires only **one additional matrix-vector product**.

### 5.7 sGB Modification Tensor A_μ^ν

From Eq. 12 of 2406.11986:
```
A_μ^ν = δ^{νσαβ}_{μλγδ} R^{γδ}_{αβ} ∇^λ∇_σϑ
      − (1/2) δ_μ^ν δ^{ησαβ}_{ηλγδ} R^{γδ}_{αβ} ∇^λ∇_σϑ
```

This involves the **rank-8 generalized Kronecker delta** (Eq. 13):
```
δ^{α₁α₂α₃α₄}_{β₁β₂β₃β₄} = det[δ^{αᵢ}_{βⱼ}]
```

contracted with Riemann tensor and second covariant derivatives of ϑ.

**TensorGR.jl can handle this:** Its abstract index machinery, Riemann tensor, covariant derivatives, and epsilon tensor / generalized Kronecker delta support are sufficient to compute A_μ^ν symbolically.

### 5.8 Modified Asymptotic Factor

From Eq. 86 of 2406.11986:
```
A_k(r) = exp(i(1 + ζ/2 · H₃⁽⁰⁾)ωr) · r^{2iMω + ρ_∞^(k)}
        · ((r−r₊)/r)^{−i(ω−mΩ_H)/(2κ) − ρ_H^(k)}
```

where Ω_H = Ω_H^(0) + ζΩ_H^(1) and κ = κ^(0) + ζκ^(1) include O(ζ) corrections.

### 5.9 Validation

Compare ω^(1) values against Tables II–IV and fitting polynomials (Eqs. 125–130) of 2406.11986 for nlm = 022, 033, 021 modes.

Key physics results to verify:
- **Isospectrality breaking:** ω^(1)_polar ≠ ω^(1)_axial
- **Polar dominance:** Polar perturbations are modified more strongly (by ~order of magnitude)
- **Accuracy:** ≤ 10⁻⁵ for a ≤ 0.6, ≤ 10⁻⁴ for 0.6 < a ≤ 0.7

---

## Milestones & Dependencies

```
M0: Package skeleton, Kerr quantities, Leaver oracle
 │
 ├── M1: Asymptotic factors, compactified coords, spectral basis
 │    │
 │    └── M2: Spectral projection machinery (linearization coeffs, inner products)
 │
 ├── M3: Symbolic linearization of Einstein eqs on Kerr via TensorGR.jl
 │    │
 │    └── M3a: Extract G coefficient arrays, validate at Schwarzschild limit
 │
 ├── [M2 + M3a both complete]
 │    │
 │    └── M4: Assemble D̃ matrices, end-to-end test at a=0.005
 │         │
 │         └── M5: Newton-Raphson solver, reproduce Table I
 │              │
 │              └── M6: Beyond-GR extension (sGB gravity)
```

### Milestone 0: Validation Infrastructure
- [ ] METRICS.jl package skeleton with Project.toml
- [ ] `kerr.jl`: Σ, Δ, r±, b, Ω_H — numerical functions, tested
- [ ] `leaver.jl`: Leaver continued-fraction QNM solver
- [ ] Test: reproduce Leaver column of Table I to 16 digits using BigFloat

### Milestone 1: Asymptotic Factors & Spectral Basis
- [ ] `perturbation_ansatz.jl`: A_k(r) with ρ_H, ρ_∞ values
- [ ] Compactified coordinate z(r) = 2r₊/r − 1 and its inverse
- [ ] `spectral.jl`: Chebyshev T_n(z) evaluation and recurrences
- [ ] Associated Legendre P_l^m(χ) evaluation and recurrences
- [ ] Derivative identities (Eq. 29) for eliminating second derivatives
- [ ] Product linearization coefficients: z^δ T_n → ∑ T_{n'}, χ^σ P_l^m → ∑ P_{l'}^m
- [ ] Test: verify orthogonality, completeness, derivative identities numerically

### Milestone 2: Spectral Projection Machinery
- [ ] `assembly.jl`: Given K coefficient array, compute D_{nl,n'l'} block matrices
- [ ] Assemble global D̃₀, D̃₁, D̃₂ from blocks
- [ ] Test: verify matrix dimensions, sparsity structure

### Milestone 3: Symbolic Linearization (Hardest Phase)
- [ ] Kerr metric in (t, r, χ, φ) using TensorGR.jl SymbolicMetric + Symbolics.jl
- [ ] Background Christoffel symbols Γ^α_βγ (40 independent components)
- [ ] Background inverse metric g^{(0)μν}
- [ ] Regge-Wheeler gauge perturbation matrix H_μν(r, χ, ω, m) for all 6 functions
- [ ] Perturbed Christoffel δΓ^α_βγ in components (4³ = 64 components, ~40 independent)
- [ ] Linearized Ricci tensor δR_μν (10 independent components)
- [ ] Mixed form δR_μ^ν = g^{(0)να} δR_μα (10 equations)
- [ ] Factor out e^{imφ−iωt}: replace ∂_t → −iω, ∂_φ → im
- [ ] Multiply through common denominators, extract polynomial structure
- [ ] `coefficients.jl`: Extract G_{k,γ,δ,σ,α,β,j} arrays
- [ ] Normalize each equation (max |G| = 1)
- [ ] **Validation:** Schwarzschild limit (a→0) reproduces Zerilli/Regge-Wheeler equations
- [ ] Cache: serialize G coefficients to JLD2

### Milestone 3a: Coordinate Transform & A_k Factorization
- [ ] Transform G coefficients from r to z = 2r₊/r − 1 → K coefficients
- [ ] Substitute h_k = A_k · u_k, factor out A_k and common denominators
- [ ] Result: K coefficients for the bounded correction functions u_k(z, χ)

### Milestone 4: End-to-End Assembly
- [ ] Spectral project K coefficients → D block matrices
- [ ] Assemble global D̃₀, D̃₁, D̃₂ for a = 0.005, N = 10 (small test)
- [ ] Verify matrix structure, rank, conditioning
- [ ] **Test:** With known ω from Leaver, verify that D̃(ω_Leaver) has a near-null vector

### Milestone 5: Newton-Raphson Solver & Table I
- [ ] `newton.jl`: residual f(x), Jacobian J(x), pseudoinverse update
- [ ] Polar-led and axial-led initialization
- [ ] Convergence monitoring (residual norm, residual ratio)
- [ ] Solve at a = 0.005, N = 30 → verify first row of Table I
- [ ] Sweep all spins: a = 0.005, 0.1, 0.2, ..., 0.95
- [ ] Compute residual ratios R, verify ~10⁻⁹ to 10⁻¹³
- [ ] Isospectrality check: |ω_polar − ω_axial| < tolerance
- [ ] **SUCCESS: Full Table I reproduced**

### Milestone 6: Beyond-GR Extension
- [ ] Parse supplementary Mathematica notebook → Julia (H₁...H₄, ϑ, Ω_H^(1), κ^(1))
- [ ] Gauss-Bonnet scalar G on Kerr via TensorGR.jl curvature invariants
- [ ] sGB modification tensor A_μ^ν using TensorGR.jl (generalized Kronecker delta, Riemann, ∇∇ϑ)
- [ ] Linearize the full sGB field equations (11 equations, 7 unknowns)
- [ ] Extract D̃^(1) matrix (beyond-GR correction to spectral matrix)
- [ ] Modified asymptotic factor with O(ζ) corrections
- [ ] Eigenvalue perturbation: ω^(1) = −J⁻¹ · D̃^(1)(ω^(0)) · v^(0)
- [ ] Compare to Tables II–IV of arxiv:2406.11986
- [ ] Verify isospectrality breaking: ω^(1)_polar ≠ ω^(1)_axial
- [ ] Verify polar dominance and accuracy bounds

---

## Key Technical Risks & Mitigations

### Risk 1: Symbolic Expression Blowup

The linearized Einstein equations on Kerr have thousands of terms per equation. Symbolics.jl may struggle with simplification and polynomial decomposition at this scale.

**Mitigations:**
1. Set M = 1 from the start (eliminate one variable)
2. Fix m = 2 initially (reduce parameter count)
3. Evaluate a numerically rather than keeping symbolic — produce G tables per spin value
4. Use Symbolics.jl `build_function()` to compile to fast Julia code, avoiding symbolic overhead at runtime
5. Parallelize: the 10 PDEs are independent, linearize them in parallel
6. Cache intermediate results (δΓ, δR components) to avoid recomputation

### Risk 2: Numerical Precision of D̃ Matrices

Coefficients span ~20 orders of magnitude. Floating-point cancellation may corrupt small entries.

**Mitigations:**
1. Equation normalization (max |G| = 1 per equation)
2. Use Double64 (MultiFloats.jl) for assembly if Float64 is insufficient
3. Condition number monitoring of D̃ and J
4. Compare against symbolic (exact rational) computation for small N

### Risk 3: Newton-Raphson Convergence

May fail to converge at high spin (a > 0.8) where the spectral expansion converges more slowly.

**Mitigations:**
1. Bootstrap from low spin (converged solution at a = 0.7 → initial guess for a = 0.8)
2. Ramp up N: solve at small N first, use solution as initial guess at larger N
3. Line search or damped Newton if standard NR oscillates
4. Check residual ratio at each iteration for monotonic decrease

### Risk 4: Correctness of Symbolic Linearization

A single sign error or index error in δΓ or δR invalidates everything.

**Mitigations:**
1. **Schwarzschild test:** At a = 0, verify against known Zerilli/Regge-Wheeler equations
2. **Ricci-flat check:** Verify δR_μν reduces correctly when h_μν = 0
3. **Trace check:** Verify g^{μν} δR_μν = δR (scalar)
4. **Bianchi identity:** Verify ∇^μ δG_μν = 0 (contracted linearized Bianchi)
5. **Component-by-component comparison** with a Mathematica/xAct reference at one spin value

---

## Why TensorGR.jl Is the Right Tool

1. **Correctness guarantees:** Christoffel symbols, curvature tensors, and perturbation formulas are already tested against known results (337k+ passing tests)
2. **Index bookkeeping:** Abstract index machinery prevents sign errors and index placement bugs that plague manual GR calculations
3. **Symmetry exploitation:** Riemann symmetries, Bianchi identities reduce expression size
4. **Symbolics.jl integration:** The `TensorGRSymbolicsExt` extension enables seamless CAS operations on scalar sub-expressions
5. **Component pipeline:** `symbolic_christoffel()`, `symbolic_riemann()`, etc. provide the exact building blocks needed
6. **Beyond-GR extensibility:** The same infrastructure handles sGB/dCS gravity with minimal new code (curvature invariants, Gauss-Bonnet scalar, generalized Kronecker delta all exist)
7. **LaTeX parser:** Can ingest equations directly from the paper's tex source for validation
8. **Perturbation theory:** Existing `define_metric_perturbation!` and `expand_perturbation()` provide the abstract framework, even though we work at component level here

---

## Performance Targets

The paper reports ~1200 seconds (~20 minutes) per spin value on a single CPU with Mathematica for N = 30.

Julia advantages:
- Compiled numerical code (vs Mathematica interpreter)
- BLAS-accelerated `pinv()` for the pseudoinverse
- Potential for GPU acceleration of matrix operations (CUDA.jl)
- Symbolics.jl code generation produces optimized Julia functions

**Target:** < 5 minutes per spin value for the full pipeline (assembly + Newton-Raphson) on a single CPU.

---

## Notation Reference

| Symbol | Meaning | Paper Eq. |
|--------|---------|-----------|
| M | Black hole mass (= 1) | below Eq. 2 |
| a | Dimensionless spin (0 ≤ a < 1) | Eq. 2 |
| b | √(1 − a²) | Eq. 2 |
| r± | M(1 ± b), horizon radii | Eq. 2 |
| Σ | r² + M²a²χ² | Eq. 2 |
| Δ | (r − r₊)(r − r₋) | Eq. 2 |
| Ω_H | a/(2r₊), horizon angular velocity | Eq. 3 |
| χ | cos(θ) | below Eq. 1 |
| h_k | 6 metric perturbation functions (k=1,...,6) | Eqs. 8a, 8b |
| A_k(r) | Asymptotic controlling factor | Eq. 15 |
| ρ_H^(k), ρ_∞^(k) | Asymptotic divergence parameters | Eq. 16 |
| u_k | Bounded correction functions: h_k = A_k · u_k | Eq. 14 |
| z | Compactified radial coord: 2r₊/r − 1 | Eq. 17 |
| T_n | Chebyshev polynomial of first kind | Eq. 20 |
| P_l^{\|m\|} | Associated Legendre polynomial | Eq. 20 |
| v_k^{nl} | Spectral coefficients | Eq. 20 |
| N | Spectral truncation order (N_z = N_χ = N) | below Eq. 20 |
| G_{k,γ,δ,σ,α,β,j} | PDE polynomial coefficients | Eq. 22 |
| K_{k,γ,δ,σ,α,β,j} | Same in compactified coords | Eq. 26 |
| D_{nl,n'l'} | 10×6 spectral block matrices | Eq. 31 |
| D̃(ω) | Global rectangular matrix = D̃₀ + D̃₁ω + D̃₂ω² | Eq. 35 |
| J | Jacobian matrix (10(N+1)² × 6(N+1)²) | Eq. 44 |
| J⁻¹ | Moore-Penrose pseudoinverse of J | Eq. 44 |
| ε | Newton-Raphson tolerance (= 10⁻⁷) | Eq. 45 |
| R | Residual ratio ‖f_final‖/‖f_initial‖ | Table I |
| ζ | Beyond-GR coupling: α²/M⁴ (sGB) | Eq. 10 of 2406.11986 |
| G | Gauss-Bonnet invariant | Eq. 2 of 2406.11986 |
| ϑ | Background scalar field | Eq. 15 of 2406.11986 |
| H₁,...,H₄ | Metric correction functions | Eq. 16 of 2406.11986 |
