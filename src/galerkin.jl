# Galerkin (spectral projection) assembly: extract coefficient functions
# from compiled field equations, transform to compactified coordinates,
# factor out A_k, then use spectral inner products to build D̃₀, D̃₁, D̃₂.
#
# This implements the paper's approach (Eqs. 22-35) and produces constant
# D̃ matrices (independent of ω), unlike the collocation approach.

export build_Dtilde_galerkin

# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1: Extract coefficient functions C_{k,j,α,β}(r, χ, ω, a)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    extract_coefficients(cfe, k, r, χ, ω_re, ω_im, a, h_map) → Matrix{ComplexF64}

Extract the 10 × n_h coefficient matrix at a single point (r, χ, ω, a).
C[k, d] = coefficient of the d-th h-derivative term in equation k.

Since the equations are linear in h-derivatives, we probe each term:
set p_d = 1, all others = 0, evaluate → C[k, d].
"""
function extract_coefficients(cfe::CompiledFieldEquations,
                              r, χ, ω_re, ω_im, a;
                              iu_val=0.0)
    n_h = cfe.n_args - 6
    n_eqs = 10
    C = zeros(Float64, n_eqs, n_h)

    args = zeros(Float64, cfe.n_args)
    args[1] = r; args[2] = χ; args[3] = ω_re; args[4] = ω_im
    args[5] = iu_val; args[6] = a

    for d in 1:n_h
        args[7:end] .= 0.0
        args[6 + d] = 1.0
        for k in 1:n_eqs
            C[k, d] = cfe.funcs[k](args)
        end
    end
    return C
end

"""
    extract_coefficients_complex(cfe, r, χ, ω, a) → Matrix{ComplexF64}

Extract the complex coefficient matrix by evaluating with iu=0 and iu=1.
"""
function extract_coefficients_complex(cfe::CompiledFieldEquations,
                                      r, χ, ω::ComplexF64, a)
    # The compiled function has f(ω_re, ω_im, iu) = a₀ + a₁·iu + a₂·iu²
    # where iu should be im (imaginary unit, im² = -1).
    # Extract polynomial coefficients in iu at iu = 0, 1, -1:
    f0  = extract_coefficients(cfe, r, χ, real(ω), imag(ω), a; iu_val=0.0)
    f1  = extract_coefficients(cfe, r, χ, real(ω), imag(ω), a; iu_val=1.0)
    fm1 = extract_coefficients(cfe, r, χ, real(ω), imag(ω), a; iu_val=-1.0)
    # a₀ = f(0), a₁ = (f(1)-f(-1))/2, a₂ = (f(1)+f(-1))/2 - a₀
    a0 = f0
    a1 = (f1 - fm1) / 2
    a2 = (f1 + fm1) / 2 - f0
    # Reconstruct: f(im) = a₀ + a₁·im + a₂·(im²) = (a₀ - a₂) + im·a₁
    return (a0 - a2) + im * a1
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2: Separate ω dependence → D̃₀, D̃₁, D̃₂
# ═══════════════════════════════════════════════════════════════════════════════

"""
    separate_omega_dependence(cfe, r, χ, a, h_map) → (C0, C1, C2)

Extract coefficient matrices at three ω values and solve for the
ω-polynomial decomposition: C(ω) = C0 + C1·ω + C2·ω².

Each C_γ is a (10, n_h) real matrix (with the understanding that
the "iu" factor handles the imaginary structure).
"""
function separate_omega_dependence(cfe::CompiledFieldEquations, r, χ, a)
    # C(ω) = C₀ + C₁ω + C₂ω²  (polynomial in complex ω)
    # Evaluate the COMPLEX coefficient matrix at ω = 0, 1, -1
    C_at_0  = extract_coefficients_complex(cfe, r, χ, ComplexF64(0), a)
    C_at_1  = extract_coefficients_complex(cfe, r, χ, ComplexF64(1), a)
    C_at_m1 = extract_coefficients_complex(cfe, r, χ, ComplexF64(-1), a)

    C0 = C_at_0
    C1 = (C_at_1 - C_at_m1) / 2
    C2 = (C_at_1 + C_at_m1) / 2 - C0

    return C0, C1, C2
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3: Build D̃ at specific ω via coefficient probing (hybrid approach)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_Dtilde_galerkin(cfe, a, N, m; h_term_strings) → METRICSSystem

Build D̃₀, D̃₁, D̃₂ via the Galerkin approach:
1. Extract coefficient functions at collocation points
2. Separate ω dependence
3. For each basis function, compute test function values
4. Project onto spectral basis

Returns a METRICSSystem with ω-independent D̃₀, D̃₁, D̃₂.
"""
function build_Dtilde_galerkin(cfe::CompiledFieldEquations,
                               a::Float64, N::Int, m::Int;
                               h_term_strings::Vector{String})
    # TODO: Full Galerkin implementation requires:
    # 1. G coefficient extraction at many (r,χ) points
    # 2. Polynomial fitting for G_{k,γ,δ,σ,α,β,j}
    # 3. Coordinate transform r → z
    # 4. A_k factorization
    # 5. Spectral projection using assembly.jl
    #
    # For now, use a "numerical Galerkin" shortcut:
    # Evaluate at collocation points and project using quadrature weights.

    error("Full Galerkin assembly not yet implemented — use build_Dtilde_collocation")
end
