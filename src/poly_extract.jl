# Polynomial coefficient extraction: the key step in the Galerkin pipeline.
#
# After A_k factorization, the equation coefficients C̃_γ(r, χ) are rational
# functions.  Multiplying by the common denominator Σ^P Δ^Q (1-χ²)^S makes
# them polynomial in (r, χ).
#
# Pipeline:
# 1. Evaluate C̃_γ at a grid of (r, χ) points
# 2. Multiply by Σ^P Δ^Q (1-χ²)^S
# 3. Fit as polynomial: Σ g_{δσ} r^δ χ^σ
# 4. Transform r → z: r = 2r₊/(1+z), re-expand as polynomial in z
# 5. Store as PDECoefficients → feed to assembly.jl

export extract_polynomial_coefficients

"""
    extract_polynomial_coefficients(cfe, a, m; P=4, Q=2, S=2, dr=14, dχ=8)

Extract the G coefficient arrays from the compiled field equations.

Returns a tuple (K₀, K₁, K₂) of PDECoefficients in the compactified z-variable,
ready for spectral projection via `assemble_system`.
"""
function extract_polynomial_coefficients(cfe::CompiledFieldEquations,
                                         a::Float64, m::Int;
                                         h_term_strings::Vector{String},
                                         P::Int=4, Q::Int=2, S::Int=2,
                                         dr::Int=14, dχ::Int=8,
                                         nr::Int=25, nχ::Int=15)
    h_map = _parse_h_terms(h_term_strings)
    params = KerrParams(a, m)
    rp = r_plus(a)
    rm = r_minus(a)

    # ── Step 1: Evaluate factored C̃_γ at (r, χ) grid ────────────────────
    r_grid = range(rp + 0.1, 20.0, length=nr)
    χ_grid = range(-0.95, 0.95, length=nχ)
    n_pts = nr * nχ
    n_h = length(h_map)

    # For each ω-power γ ∈ {0,1,2}: array of C̃_γ[k, d] at all grid points
    C̃_at_pts = [zeros(ComplexF64, 10, n_h, n_pts) for _ in 1:3]

    pt = 0
    for r in r_grid, χ in χ_grid
        pt += 1
        # Separate ω-dependence
        C₀, C₁, C₂ = _separate_omega_at_point(cfe, r, χ, a)
        # Transform h → u for each ω-power
        for (γ, Cγ) in enumerate((C₀, C₁, C₂))
            C̃ = _transform_h_to_u(Cγ, h_map, r, ComplexF64(0), params)
            C̃_at_pts[γ][:, :, pt] = C̃
        end
    end

    # ── Step 2: Clear denominators and fit polynomials ────────────────────
    # Common denominator: Σ^P × Δ^Q × (1-χ²)^S
    denom_at_pts = zeros(Float64, n_pts)
    pt = 0
    for r in r_grid, χ in χ_grid
        pt += 1
        Σv = r^2 + a^2 * χ^2
        Δv = (r - rp) * (r - rm)
        s2 = 1 - χ^2
        denom_at_pts[pt] = Σv^P * Δv^Q * s2^S
    end

    # Vandermonde matrix for polynomial fitting
    n_mono = (dr + 1) * (dχ + 1)
    V = zeros(Float64, n_pts, n_mono)
    pt = 0
    for r in r_grid, χ in χ_grid
        pt += 1
        col = 0
        for δ in 0:dr, σ in 0:dχ
            col += 1
            V[pt, col] = r^δ * χ^σ
        end
    end
    V_pinv = pinv(V)  # precompute for all fits

    # Build PDECoefficients for each ω-power
    Ks = [PDECoefficients([Dict{NTuple{6,Int},ComplexF64}() for _ in 1:10],
                          fill(dr, 10), fill(dχ, 10)) for _ in 1:3]

    for γ in 1:3, k in 1:10, d in 1:n_h
        j_d, α_d, β_d = h_map[d]

        # Values of C̃_γ[k, d] × denominator at grid points
        vals_re = [real(C̃_at_pts[γ][k, d, pt]) * denom_at_pts[pt] for pt in 1:n_pts]
        vals_im = [imag(C̃_at_pts[γ][k, d, pt]) * denom_at_pts[pt] for pt in 1:n_pts]

        # Skip if all zero
        (maximum(abs, vals_re) < 1e-14 && maximum(abs, vals_im) < 1e-14) && continue

        # Fit polynomial: real and imaginary parts separately
        c_re = V_pinv * vals_re
        c_im = V_pinv * vals_im

        # Store nonzero coefficients
        col = 0
        for δ in 0:dr, σ in 0:dχ
            col += 1
            g_val = complex(c_re[col], c_im[col])
            abs(g_val) < 1e-14 && continue
            Ks[γ].equations[k][(γ - 1, δ, σ, α_d, β_d, j_d)] = g_val
        end
    end

    return Ks[1], Ks[2], Ks[3]
end

# Helper: separate ω-dependence at a single (r, χ) point
function _separate_omega_at_point(cfe, r, χ, a)
    C_0  = extract_coefficients_complex(cfe, r, χ, ComplexF64(0), a)
    C_1  = extract_coefficients_complex(cfe, r, χ, ComplexF64(1), a)
    C_m1 = extract_coefficients_complex(cfe, r, χ, ComplexF64(-1), a)
    C₀ = C_0
    C₁ = (C_1 - C_m1) / 2
    C₂ = (C_1 + C_m1) / 2 - C_0
    return C₀, C₁, C₂
end
