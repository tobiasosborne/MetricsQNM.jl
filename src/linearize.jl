# Symbolic linearization of Einstein equations on Kerr background.
# Uses TensorGR.jl for Christoffel symbols and Symbolics.jl for CAS.
#
# After factoring out e^{imφ-iωt}, ALL time/azimuthal derivatives become
# algebraic: ∂_t → -iω, ∂_φ → im.  We introduce a symbolic ω_s and use
# integer m throughout.

using Symbolics
using TensorGR: symbolic_metric, symbolic_christoffel, sym_deriv

# ═══════════════════════════════════════════════════════════════════════════════
#  Mode derivative:  ∂_t → -iω,  ∂_φ → im,  ∂_r/∂_χ → sym_deriv
# ═══════════════════════════════════════════════════════════════════════════════

"""
    mode_deriv(expr, coord, t, phi, ω_s, m_mode)

Derivative appropriate for a mode with time-azimuthal factor e^{imφ-iωt}
already removed. `ω_s` is a symbolic frequency variable.
"""
# Mode derivative: ∂_t → -iω, ∂_φ → im, with ω split into (ω_re, ω_im)
# and imaginary unit encoded as symbolic variable `iu` (set to im at evaluation).
function mode_deriv(expr, coord, t, phi, ω_re, ω_im, iu, m_mode)
    if isequal(coord, t)
        # -iω f = (ω_im - i ω_re) f = ω_im f + iu * (-ω_re) f
        return (ω_im - iu * ω_re) * expr
    elseif isequal(coord, phi)
        # im f = i m f
        return iu * m_mode * expr
    else
        return sym_deriv(expr, coord)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Kerr metric (Eq. 1, M = 1)
# ═══════════════════════════════════════════════════════════════════════════════

function kerr_symbolic_metric(; M=1)
    @variables t r chi a_s phi

    Sig = r^2 + a_s^2 * chi^2
    Del = r^2 - 2M * r + a_s^2

    g = zeros(Num, 4, 4)
    g[1,1] = -(1 - 2M * r / Sig)
    g[1,4] = g[4,1] = -2M^2 * a_s * r * (1 - chi^2) / Sig
    g[2,2] = Sig / Del
    g[3,3] = Sig / (1 - chi^2)
    g[4,4] = (r^2 + M^2 * a_s^2 + 2M^3 * a_s^2 * r * (1 - chi^2) / Sig) * (1 - chi^2)

    sm = symbolic_metric([t, r, chi, phi], g)
    return sm, [t, r, chi, phi], (a_s,)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Regge-Wheeler perturbation (Eqs. 8a, 8b)
# ═══════════════════════════════════════════════════════════════════════════════

function regge_wheeler_perturbation(coords, m_mode::Int)
    rv, cv = coords[2], coords[3]
    @variables h1(rv, cv) h2(rv, cv) h3(rv, cv) h4(rv, cv) h5(rv, cv) h6(rv, cv)
    Dχ = Symbolics.Differential(cv)
    χ = cv
    mm = Num(m_mode)

    h = Array{Any}(nothing, 4, 4)
    for i in 1:4, j in 1:4; h[i,j] = Num(0); end

    # Even (polar) — Eq. 8b, with overall minus sign
    h[1,1] = -h1
    h[1,2] = h[2,1] = -h2
    h[2,2] = -h3
    h[3,3] = -h4 / (1 - χ^2)
    h[4,4] = -(1 - χ^2) * h4

    # Odd (axial) — Eq. 8a
    # The -im/(1-χ²) factor: im is from the gauge choice, kept as symbolic mm
    # (the actual im will be restored when extracting complex G coefficients)
    h[1,3] = h[3,1] = -mm / (1 - χ^2) * h5
    h[1,4] = h[4,1] = (1 - χ^2) * Symbolics.expand_derivatives(Dχ(h5))
    h[2,3] = h[3,2] = -mm / (1 - χ^2) * h6
    h[2,4] = h[4,2] = (1 - χ^2) * Symbolics.expand_derivatives(Dχ(h6))

    return h, [h1, h2, h3, h4, h5, h6]
end

# ═══════════════════════════════════════════════════════════════════════════════
#  δΓ^α_βγ  (Palatini identity with mode derivatives)
# ═══════════════════════════════════════════════════════════════════════════════

function linearized_christoffel(ginv, h, coords, ω_re, ω_im, iu, m_mode)
    dim = size(ginv, 1)
    t, phi = coords[1], coords[4]
    δΓ = Array{Any}(undef, dim, dim, dim)
    for α in 1:dim, β in 1:dim, γ in 1:dim
        s = Num(0)
        for δ in 1:dim
            dh1 = mode_deriv(h[γ, δ], coords[β], t, phi, ω_re, ω_im, iu, m_mode)
            dh2 = mode_deriv(h[β, δ], coords[γ], t, phi, ω_re, ω_im, iu, m_mode)
            dh3 = mode_deriv(h[β, γ], coords[δ], t, phi, ω_re, ω_im, iu, m_mode)
            s += ginv[α, δ] * (dh1 + dh2 - dh3)
        end
        δΓ[α, β, γ] = s / 2
    end
    δΓ
end

linearized_christoffel(ginv, h, coords) =
    linearized_christoffel(ginv, h, coords, Num(0), Num(0), Num(0), 0)

# ═══════════════════════════════════════════════════════════════════════════════
#  δR_μν and δR_μ^ν  (with mode derivatives)
# ═══════════════════════════════════════════════════════════════════════════════

function linearized_ricci(Γ, δΓ, coords, ω_re, ω_im, iu, m_mode)
    dim = size(Γ, 1)
    t, phi = coords[1], coords[4]
    δR = Array{Any}(undef, dim, dim)
    for μ in 1:dim, ν in 1:dim
        s = Num(0)
        for α in 1:dim
            s += mode_deriv(δΓ[α, μ, ν], coords[α], t, phi, ω_re, ω_im, iu, m_mode)
            s -= mode_deriv(δΓ[α, α, ν], coords[μ], t, phi, ω_re, ω_im, iu, m_mode)
        end
        for α in 1:dim, β in 1:dim
            s += Γ[α, α, β] * δΓ[β, μ, ν]
            s += δΓ[α, α, β] * Γ[β, μ, ν]
            s -= Γ[α, μ, β] * δΓ[β, α, ν]
            s -= δΓ[α, μ, β] * Γ[β, α, ν]
        end
        δR[μ, ν] = s
    end
    δR
end

linearized_ricci(Γ, δΓ, coords) =
    linearized_ricci(Γ, δΓ, coords, Num(0), Num(0), Num(0), 0)

function linearized_ricci_mixed(ginv, δR_cov)
    dim = size(ginv, 1)
    [sum(ginv[ν, α] * δR_cov[μ, α] for α in 1:dim) for μ in 1:dim, ν in 1:dim]
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Full pipeline: Kerr → 10 linearized field equations
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_field_equations(m_mode=2)

Run the full symbolic pipeline and return the 10 linearized field equations
δR_μ^ν = 0 as symbolic expressions involving h_k(r,χ), their derivatives,
ω_s (symbolic frequency), and a_s (symbolic spin).

Returns (eqs, coords, params, hfuncs) where eqs is a length-10 vector.
"""
function compute_field_equations(m_mode::Int=2)
    @variables omega_re omega_im iu_sym

    sm, coords, params = kerr_symbolic_metric()
    Γ = symbolic_christoffel(sm)
    h, hfuncs = regge_wheeler_perturbation(coords, m_mode)
    δΓ = linearized_christoffel(sm.ginv, h, coords, omega_re, omega_im, iu_sym, m_mode)
    δR = linearized_ricci(Γ, δΓ, coords, omega_re, omega_im, iu_sym, m_mode)
    δR_mixed = linearized_ricci_mixed(sm.ginv, δR)

    idx = [(1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,3),(3,4),(4,4)]
    eqs = [δR_mixed[μ, ν] for (μ, ν) in idx]

    return eqs, coords, params, hfuncs, (omega_re, omega_im, iu_sym)
end
