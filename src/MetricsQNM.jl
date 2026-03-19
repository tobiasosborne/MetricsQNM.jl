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

# ── Leaver continued-fraction solver ─────────────────────────────────────────
include("leaver.jl")
export leaver_qnm

# ── Symbolic linearization ───────────────────────────────────────────────────
include("linearize.jl")

# ── PDE coefficient extraction ───────────────────────────────────────────────
include("coefficients.jl")
export PDECoefficients

# ── Spectral assembly ────────────────────────────────────────────────────────
include("assembly.jl")
export METRICSSystem, assemble_system

# ── D̃ matrix assembly (collocation + Galerkin) ──────────────────────────────
include("collocation.jl")
include("galerkin.jl")
include("dtilde.jl")
include("factored_assembly.jl")
include("poly_extract.jl")
include("pipeline.jl")
include("symbolic_decompose.jl")
include("zspace_extract.jl")
include("sparse_poly.jl")
include("symbolic_pipeline.jl")

# ── Newton-Raphson solver ────────────────────────────────────────────────────
include("newton.jl")
export solve_qnm, reproduce_table1

# ── SVD-based direct solver ──────────────────────────────────────────────────
include("solve.jl")

# ── Rectangular QEP solver (SVD compression + companion QZ) ─────────────────
include("rectangular_qep.jl")

# ── sGB background corrections ──────────────────────────────────────────────
include("sgb_background.jl")

# ── sGB linearization (modified background metric → D̃⁽¹⁾) ─────────────────
include("sgb_linearize.jl")

# ── sGB eigenvalue perturbation solver ──────────────────────────────────────
include("sgb_perturbation.jl")

end
