module MetricsQNM

using LinearAlgebra
using SparseArrays
using Printf

# ── Kerr background ──────────────────────────────────────────────────────────
include("kerr.jl")
export KerrParams, b, r_plus, r_minus, Omega_H, Sigma, Delta, kappa

# ── Asymptotic factors & compactified coordinate ─────────────────────────────
include("perturbation_ansatz.jl")
export rho_H, rho_inf, asymptotic_factor
export z_of_r, r_of_z, dz_dr, dr_dz

# ── Spectral bases ───────────────────────────────────────────────────────────
include("spectral.jl")
export ChebyshevBasis, chebyshev_basis
export LegendreBasis, legendre_basis
export SpectralBasis, spectral_basis, nl_index

# ── Leaver continued-fraction solver (reference oracle) ──────────────────────
include("leaver.jl")
export leaver_qnm

# ── Symbolic linearization (Kerr → δR → 10 field equations) ─────────────────
include("linearize.jl")

# ── Bespoke SparsePoly CAS ──────────────────────────────────────────────────
include("sparse_poly.jl")

# ── PDE coefficient extraction ───────────────────────────────────────────────
include("coefficients.jl")
export PDECoefficients

# ── Spectral Galerkin assembly (D̃ matrices) ──────────────────────────────────
include("assembly.jl")
export METRICSSystem, assemble_system

# ── Bespoke symbolic G-extraction pipeline ───────────────────────────────────
include("symbolic_pipeline.jl")

# ── Rectangular QEP solver (SVD compression + companion QZ) ─────────────────
include("rectangular_qep.jl")

# ── sGB background corrections (Mathematica parser → H₁-H₄) ────────────────
include("sgb_background.jl")

# ── sGB linearization (abstract H-params → correction equations) ────────────
include("sgb_linearize.jl")

# ── sGB eigenvalue perturbation solver (Eq. 111) ────────────────────────────
include("sgb_perturbation.jl")

end
