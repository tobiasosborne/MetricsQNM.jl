# D̃ matrix assembly: the bridge from symbolic field equations to eigenvalue problem.
#
# Two approaches, designed for cross-checking:
#   1. Collocation: evaluate field equations at grid points with A_k × T_n × P_l test functions
#   2. Galerkin:    extract ω-polynomial coefficients, project onto spectral basis
#
# The collocation approach builds D̃(ω) at a specific ω.
# The Galerkin approach builds constant D̃₀, D̃₁, D̃₂.

export build_Dtilde

# ═══════════════════════════════════════════════════════════════════════════════
#  Grid data: precomputed basis values at collocation points
# ═══════════════════════════════════════════════════════════════════════════════

struct RadialGrid
    z::Vector{Float64}           # Chebyshev nodes z_i
    r::Vector{Float64}           # r_i = r_of_z(z_i, rp)
    T::Matrix{Float64}           # T[i, n+1]  = T_n(z_i)
    dT_dz::Matrix{Float64}       # dT_n/dz at z_i
    d2T_dz2::Matrix{Float64}     # d²T_n/dz² at z_i
    dz_dr::Vector{Float64}       # dz/dr at r_i
    d2z_dr2::Vector{Float64}     # d²z/dr² at r_i
end

function RadialGrid(N::Int, rp::Float64)
    # Chebyshev-Gauss nodes (interior, avoid boundary singularities)
    z = [cos(π * (2k - 1) / (2 * (N + 1))) for k in 1:(N+1)]
    r = [r_of_z(zi, rp) for zi in z]

    T    = zeros(N + 1, N + 1)
    dTdz = zeros(N + 1, N + 1)
    d2Tdz2 = zeros(N + 1, N + 1)

    for (i, zi) in enumerate(z)
        # Three-term recurrence for T_n and dT_n/dz
        T[i, 1] = 1.0;  T[i, 2] = zi
        dTdz[i, 1] = 0.0;  dTdz[i, 2] = 1.0
        for n in 2:N
            T[i, n+1]    = 2zi * T[i, n] - T[i, n-1]
            dTdz[i, n+1] = 2T[i, n] + 2zi * dTdz[i, n] - dTdz[i, n-1]
        end
        # Second derivative via identity: (1-z²)T'' = zT' - n²T
        for n in 0:N
            z2m1 = 1 - zi^2
            if abs(z2m1) > 1e-14
                d2Tdz2[i, n+1] = (zi * dTdz[i, n+1] - n^2 * T[i, n+1]) / z2m1
            else
                d2Tdz2[i, n+1] = n^2 * (n^2 - 1) / 3.0  # limit at z = ±1
            end
        end
    end

    dz = [dz_dr(ri, rp) for ri in r]
    d2z = [4rp / ri^3 for ri in r]  # d²z/dr² = d/dr(-2rp/r²) = 4rp/r³

    RadialGrid(z, r, T, dTdz, d2Tdz2, dz, d2z)
end

struct AngularGrid
    χ::Vector{Float64}           # Legendre-Gauss nodes
    P::Matrix{Float64}           # P[i, l-|m|+1]  = P_l^|m|(χ_i)
    dP::Matrix{Float64}          # dP_l^|m|/dχ at χ_i
    d2P::Matrix{Float64}         # d²P_l^|m|/dχ² at χ_i
    d3P::Matrix{Float64}         # d³P_l^|m|/dχ³ at χ_i
end

function AngularGrid(N::Int, m::Int)
    am = abs(m)
    # Use Chebyshev-Gauss nodes for χ as well (interior points)
    χ = [cos(π * (2k - 1) / (2 * (N + 1))) for k in 1:(N+1)]

    lmax = am + N
    P   = zeros(N + 1, N + 1)
    dP  = zeros(N + 1, N + 1)
    d2P = zeros(N + 1, N + 1)
    d3P = zeros(N + 1, N + 1)

    for (i, χi) in enumerate(χ)
        _fill_legendre!(view(P, i, :), view(dP, i, :), view(d2P, i, :),
                        view(d3P, i, :), χi, N, am)
    end

    AngularGrid(χ, P, dP, d2P, d3P)
end

function _fill_legendre!(P, dP, d2P, d3P, χ, N, am)
    # P_{|m|}^{|m|} via double factorial
    fact = 1.0
    for k in 1:am; fact *= (2k - 1); end
    s2 = 1 - χ^2
    P[1] = (-1)^am * fact * s2^(am / 2)

    # First three derivatives of P_{|m|}^{|m|}
    if am > 0
        dP[1] = -am * χ * P[1] / s2
        d2P[1] = (-am * P[1] + (-am * χ) * dP[1] * s2 - (-am * χ * P[1]) * (-2χ)) / s2^2
        # ... third derivative is complex, use finite differences if needed
    end

    # P_{|m|+1}^{|m|}
    if N ≥ 1
        l = am + 1
        P[2]  = χ * (2am + 1) * P[1]
        dP[2] = (2am + 1) * (P[1] + χ * dP[1])
    end

    # Higher l via standard recurrence
    for ii in 3:(N + 1)
        l = am + ii - 1
        P[ii]  = ((2l - 1) * χ * P[ii-1] - (l + am - 1) * P[ii-2]) / (l - am)
        dP[ii] = ((2l - 1) * (P[ii-1] + χ * dP[ii-1]) - (l + am - 1) * dP[ii-2]) / (l - am)
    end

    # Second and third derivatives via the identity:
    # (1-χ²) P'' = 2χ P' - l(l+1) P + m²/(1-χ²) P
    for ii in 1:(N + 1)
        l = am + ii - 1
        if abs(s2) > 1e-14
            d2P[ii] = (2χ * dP[ii] - l * (l + 1) * P[ii] + am^2 / s2 * P[ii]) / s2
        end
        # d³P/dχ³ via differentiating the identity (for third-order terms)
        # (1-χ²)P''' = 4χP'' - (l(l+1)-2)P' + 2m²χ/(1-χ²)²P + m²/(1-χ²)P'
        if abs(s2) > 1e-14
            d3P[ii] = (4χ * d2P[ii] - (l * (l + 1) - 2) * dP[ii] +
                       2am^2 * χ / s2^2 * P[ii] + am^2 / s2 * dP[ii]) / s2
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Asymptotic factor A_k and its r-derivatives at grid points
# ═══════════════════════════════════════════════════════════════════════════════

struct AsymptoticGrid
    A::Matrix{ComplexF64}     # A[i, k] = A_k(r_i, ω)
    dA::Matrix{ComplexF64}    # dA_k/dr
    d2A::Matrix{ComplexF64}   # d²A_k/dr²
end

function AsymptoticGrid(rgrid::RadialGrid, ω::ComplexF64, params::KerrParams)
    nr = length(rgrid.r)
    A   = zeros(ComplexF64, nr, 6)
    dA  = zeros(ComplexF64, nr, 6)
    d2A = zeros(ComplexF64, nr, 6)

    rp = r_plus(params.a)
    b_ = b(params.a)
    ΩH = Omega_H(params.a)

    for k in 1:6
        ρH = rho_H(k)
        ρ∞ = rho_inf(k)
        σ₊ = -im * (ω - params.m * ΩH) * (1 + b_) / b_ - ρH

        for (i, r) in enumerate(rgrid.r)
            x = (r - rp) / r
            logA = im * ω * r + (2im * ω + ρ∞) * log(r) + σ₊ * log(complex(x))
            A[i, k] = exp(logA)

            dlogA = im * ω + (2im * ω + ρ∞) / r + σ₊ * rp / (r * (r - rp))
            dA[i, k] = A[i, k] * dlogA

            d2logA = -(2im * ω + ρ∞) / r^2 - σ₊ * rp * (2r - rp) / (r * (r - rp))^2
            d2A[i, k] = A[i, k] * (d2logA + dlogA^2)
        end
    end

    AsymptoticGrid(A, dA, d2A)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Test function h-derivatives at a grid point
# ═══════════════════════════════════════════════════════════════════════════════

"""
Compute h and all its (r,χ)-derivatives for a test function.

Without A_k: h = T_n(z(r)) × P_l^m(χ)
With A_k:    h = A_k(r,ω) × T_n(z(r)) × P_l^m(χ)
"""
function _h_derivatives(Tn, dTn_dz, d2Tn_dz2,
                        Pl, dPl, d2Pl, d3Pl,
                        dz, d2z;
                        Ak=1.0, dAk=0.0, d2Ak=0.0)
    # Base function u = T_n P_l and its partials
    u      = Tn * Pl
    du_r   = dTn_dz * dz * Pl           # ∂u/∂r via chain rule
    du_χ   = Tn * dPl
    d2u_rr = (d2Tn_dz2 * dz^2 + dTn_dz * d2z) * Pl
    du_rχ  = dTn_dz * dz * dPl
    d2u_χχ = Tn * d2Pl
    d3u_rrχ = (d2Tn_dz2 * dz^2 + dTn_dz * d2z) * dPl
    d3u_rχχ = dTn_dz * dz * d2Pl
    d3u_χχχ = Tn * d3Pl

    # h = A × u  (A_k = 1 when not using asymptotic factorization)
    h      = Ak * u
    dh_r   = dAk * u + Ak * du_r
    dh_χ   = Ak * du_χ
    d2h_rr = d2Ak * u + 2dAk * du_r + Ak * d2u_rr
    dh_rχ  = dAk * du_χ + Ak * du_rχ
    d2h_χχ = Ak * d2u_χχ
    d3h_rrχ = d2Ak * du_χ + 2dAk * du_rχ + Ak * d3u_rrχ
    d3h_rχχ = dAk * d2u_χχ + Ak * d3u_rχχ
    d3h_χχχ = Ak * d3u_χχχ

    return (h=h, dr=dh_r, dχ=dh_χ, drr=d2h_rr, drχ=dh_rχ, dχχ=d2h_χχ,
            drrχ=d3h_rrχ, drχχ=d3h_rχχ, dχχχ=d3h_χχχ)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Map h-derivative names → (j, α_r, β_χ)
# ═══════════════════════════════════════════════════════════════════════════════

function _parse_h_terms(h_term_strings::Vector{String})
    map = Vector{Tuple{Int, Int, Int}}()
    for s in h_term_strings
        j = findfirst(jj -> occursin("h$(jj)", s), 1:6)
        @assert j !== nothing "Cannot parse j from: $s"
        α_r = count("Differential(r", s)
        β_χ = count("Differential(chi", s)
        push!(map, (j, α_r, β_χ))
    end
    return map
end

function _lookup_deriv(hd, α_r, β_χ)
    # Map (α_r, β_χ) to the right field of the NamedTuple
    (α_r, β_χ) == (0, 0) && return hd.h
    (α_r, β_χ) == (1, 0) && return hd.dr
    (α_r, β_χ) == (0, 1) && return hd.dχ
    (α_r, β_χ) == (2, 0) && return hd.drr
    (α_r, β_χ) == (1, 1) && return hd.drχ
    (α_r, β_χ) == (0, 2) && return hd.dχχ
    (α_r, β_χ) == (2, 1) && return hd.drrχ
    (α_r, β_χ) == (1, 2) && return hd.drχχ
    (α_r, β_χ) == (0, 3) && return hd.dχχχ
    error("Unsupported derivative order ($α_r, $β_χ)")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Main D̃ assembly — collocation approach
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_Dtilde(cfe, a, ω, N, m; h_term_strings, use_Ak=false) → Matrix{ComplexF64}

Build D̃(ω) via collocation at a specific complex ω.

If `use_Ak=true`, test functions are h_j = A_j(r,ω) × T_n(z) × P_l^m(χ).
If `use_Ak=false` (default), test functions are h_j = T_n(z) × P_l^m(χ) directly.

The A_k-free version is numerically more stable for small N but converges slower.
"""
function build_Dtilde(cfe::CompiledFieldEquations,
                      a::Float64, ω::ComplexF64, N::Int, m::Int;
                      h_term_strings::Vector{String}, use_Ak::Bool=false)
    params = KerrParams(a, m)
    rp = r_plus(a)
    am = abs(m)
    sz = N + 1
    bs = sz^2
    n_eqs = 10
    n_fields = 6
    n_h = length(h_term_strings)

    # Parse h-term ordering
    h_map = _parse_h_terms(h_term_strings)

    # Precompute grids
    rgrid = RadialGrid(N, rp)
    agrid = AngularGrid(N, m)
    afact = use_Ak ? AsymptoticGrid(rgrid, ω, params) : nothing

    # ── Step 1: Extract coefficient matrices at all grid points ──────────
    C = zeros(ComplexF64, n_eqs, n_h, sz, sz)
    for i_r in 1:sz, i_χ in 1:sz
        C[:, :, i_r, i_χ] = extract_coefficients_complex(
            cfe, rgrid.r[i_r], agrid.χ[i_χ], ω, a)
    end

    # ── Step 2: Build D̃ by contracting coefficients with test function derivatives ─
    D = zeros(ComplexF64, n_eqs * bs, n_fields * bs)

    for j₀ in 1:n_fields, n₀ in 0:N, l₀ in am:(am + N)
        col = (j₀ - 1) * bs + n₀ * sz + (l₀ - am) + 1
        in = n₀ + 1       # 1-based Chebyshev index
        il = l₀ - am + 1  # 1-based Legendre index

        for i_r in 1:sz
            Ak_kw = if use_Ak
                (Ak=afact.A[i_r, j₀], dAk=afact.dA[i_r, j₀], d2Ak=afact.d2A[i_r, j₀])
            else
                (;)
            end

            for i_χ in 1:sz
                hd = _h_derivatives(
                    rgrid.T[i_r, in], rgrid.dT_dz[i_r, in], rgrid.d2T_dz2[i_r, in],
                    agrid.P[i_χ, il], agrid.dP[i_χ, il],
                    agrid.d2P[i_χ, il], agrid.d3P[i_χ, il],
                    rgrid.dz_dr[i_r], rgrid.d2z_dr2[i_r];
                    Ak_kw...)

                for k in 1:n_eqs
                    row = (k - 1) * bs + (i_r - 1) * sz + i_χ
                    val = zero(ComplexF64)
                    @inbounds for d in 1:n_h
                        j_d, α_d, β_d = h_map[d]
                        j_d == j₀ || continue
                        val += C[k, d, i_r, i_χ] * _lookup_deriv(hd, α_d, β_d)
                    end
                    D[row, col] = val
                end
            end
        end
    end

    return D
end

# ═══════════════════════════════════════════════════════════════════════════════
#  D̃₀, D̃₁, D̃₂ via Galerkin (ω-independent constant matrices)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_Dtilde_constant(cfe, a, N, m; h_term_strings) → METRICSSystem

Build constant D̃₀, D̃₁, D̃₂ by evaluating at three ω values and solving
the linear system D̃(ω) = D̃₀ + D̃₁ω + D̃₂ω².

This is the Galerkin approach: the matrices are independent of ω.
"""
function build_Dtilde_constant(cfe::CompiledFieldEquations,
                               a::Float64, N::Int, m::Int;
                               h_term_strings::Vector{String})
    # Evaluate D̃ at three ω values
    ω₀ = ComplexF64(0)
    ω₁ = ComplexF64(1)
    ω₂ = ComplexF64(-1)

    D_at_0  = build_Dtilde(cfe, a, ω₀, N, m; h_term_strings)
    D_at_1  = build_Dtilde(cfe, a, ω₁, N, m; h_term_strings)
    D_at_m1 = build_Dtilde(cfe, a, ω₂, N, m; h_term_strings)

    # D̃(ω) = D₀ + D₁ω + D₂ω²
    D0 = D_at_0
    D1 = (D_at_1 - D_at_m1) / 2
    D2 = (D_at_1 + D_at_m1) / 2 - D_at_0

    METRICSSystem(D0, D1, D2, N, m, a)
end
