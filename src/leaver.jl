# Leaver continued-fraction QNM solver (Leaver 1985).
# Recurrence follows Cook & Zalutskiy (2014) / Stein (2019).  M = 1 units.
#
# Angular eigenvalue: SpinWeightedSpheroidalHarmonics.jl returns the Teukolsky
# separation constant λ = A_slm + c² − 2mc.  The Leaver recurrence needs A_slm,
# so we convert: A_slm = λ − c² + 2mc.

using SpinWeightedSpheroidalHarmonics: spin_weighted_spheroidal_eigenvalue

"""
    leaver_qnm(a; s=-2, l=2, m=2, n=0, Nterms=300, tol=1e-12, maxiter=200)

Compute the QNM frequency ω for a Kerr black hole with dimensionless spin `a`.
Returns complex ω in M = 1 units (Re > 0, Im < 0 for damped modes).
"""
function leaver_qnm(a::Real; s::Int=-2, l::Int=2, m::Int=2, n::Int=0,
                     Nterms::Int=300, tol::Real=1e-12, maxiter::Int=200)
    ω0 = _initial_guess(a, s, l, m, n)
    _newton_solve(ω0, a, s, l, m, n, Nterms, tol, maxiter)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  RADIAL recurrence: Cook & Zalutskiy (2014) Eqs. (31)-(39)
# ═══════════════════════════════════════════════════════════════════════════════

function _recurrence(n, ω, a, s, m, Alm)
    root = sqrt(1 - a^2)
    rp = 1 + root
    rm = 1 - root

    σ₊ = (2ω * rp - m * a) / (2root)
    σ₋ = (2ω * rm - m * a) / (2root)

    ζ = im * ω
    ξ = -s - im * σ₊
    η = -im * σ₋

    p = root * ζ
    α = 1 + 2s + ξ + η - 2ζ
    γ = 1 + s + 2η
    δ = 1 + s + 2ξ
    σ = Alm + a^2 * ω^2 - 8ω^2 +
        p * (2α + γ - δ) +
        (1 + s - (γ + δ) / 2) * (s + (γ + δ) / 2)

    D0 = δ
    D1 = 4p - 2α + γ - δ - 2
    D2 = 2α - γ + 2
    D3 = α * (4p - δ) - σ
    D4 = α * (α - γ + 1)

    α_n = n^2 + (D0 + 1) * n + D0
    β_n = -2n^2 + (D1 + 2) * n + D3
    γ_n = n^2 + (D2 - 3) * n + D4 - D2 + 2

    return α_n, β_n, γ_n
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Angular eigenvalue A_slm(c) where c = aω
# ═══════════════════════════════════════════════════════════════════════════════

function _angular_eigenvalue(c, s, l, m)
    # SpinWeightedSpheroidalHarmonics.jl returns the Teukolsky constant
    # λ = A_slm + c² − 2mc.  The Leaver recurrence needs A_slm.
    λ = spin_weighted_spheroidal_eigenvalue(s, l, m, c)
    return λ - c^2 + 2m * c
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Radial continued fraction (Cook & Zalutskiy Eq. 44)
# ═══════════════════════════════════════════════════════════════════════════════

function _continued_fraction(ω, a, s, l, m, n_inv, Nterms)
    Alm = _angular_eigenvalue(a * ω, s, l, m)

    N = Nterms
    αs = Vector{ComplexF64}(undef, N + 1)
    βs = Vector{ComplexF64}(undef, N + 1)
    γs = Vector{ComplexF64}(undef, N + 1)
    for n in 0:N
        αs[n+1], βs[n+1], γs[n+1] = _recurrence(n, ω, a, s, m, Alm)
    end

    conv1 = zero(ComplexF64)
    for i in 0:(n_inv - 1)
        conv1 = αs[i+1] / (βs[i+1] - γs[i+1] * conv1)
    end

    conv2 = -1.0 + 0.0im
    for i in N:-1:(n_inv + 1)
        conv2 = γs[i+1] / (βs[i+1] - αs[i+1] * conv2)
    end

    return βs[n_inv+1] - γs[n_inv+1] * conv1 - αs[n_inv+1] * conv2
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Initial guess & Newton solver
# ═══════════════════════════════════════════════════════════════════════════════

function _initial_guess(a, s, l, m, n)
    ω0_table = Dict(
        2 => complex(0.3736716844553446, -0.0889623156104721),
        3 => complex(0.5994110842,       -0.0927030234),
    )
    ω0 = get(ω0_table, l, complex(l + 0.5, -(n + 0.5)) / sqrt(27.0))
    rp = 1 + sqrt(1 - a^2)
    return ω0 + m * a / (2rp)
end

function _newton_solve(ω0, a, s, l, m, n_inv, Nterms, tol, maxiter)
    ω = ComplexF64(ω0)
    h = 1e-8

    for _ in 1:maxiter
        f  = _continued_fraction(ω, a, s, l, m, n_inv, Nterms)
        df = (_continued_fraction(ω + h, a, s, l, m, n_inv, Nterms) - f) / h

        Δω = -f / df
        ω += Δω

        abs(Δω) < tol * (1 + abs(ω)) && return ω
    end

    @warn "Leaver: no convergence after $maxiter iterations" ω
    return ω
end
