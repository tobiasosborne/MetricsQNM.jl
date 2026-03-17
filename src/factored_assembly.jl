# Factored D̃ assembly: the correct Galerkin approach from the paper.
#
# Pipeline:
# 1. Extract h-derivative coefficients C_{k,d}(r, χ, ω, a)
# 2. Transform h → u via A_k factorization: divide each field's terms by A_j
# 3. Separate ω: C̃ = C̃₀ + C̃₁ω + C̃₂ω²  (verified: exactly degree 2)
# 4. For each ω-power: evaluate C̃_γ at collocation points
# 5. Build D̃_γ by contracting with spectral basis values
#
# This produces CONSTANT D̃₀, D̃₁, D̃₂ matrices (independent of ω).

export build_factored_system

# ═══════════════════════════════════════════════════════════════════════════════
#  A_k log-derivatives (rational functions of r and ω)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Compute [1, (dA_j/dr)/A_j, (d²A_j/dr²)/A_j] at a point (r, ω).
These are the building blocks for the h → u transformation.
"""
function _Ak_ratios(j::Int, r, ω, params::KerrParams)
    rp  = r_plus(params.a)
    b_  = b(params.a)
    ΩH  = Omega_H(params.a)
    ρ∞  = rho_inf(j)
    σ₊  = -im * (ω - params.m * ΩH) * (1 + b_) / b_ - rho_H(j)

    # d(log A)/dr = iω + (2iω + ρ∞)/r + σ₊ r₊/(r(r - r₊))
    L1 = im * ω + (2im * ω + ρ∞) / r + σ₊ * rp / (r * (r - rp))

    # d²(log A)/dr² = -(2iω + ρ∞)/r² - σ₊ r₊(2r - r₊)/(r(r - r₊))²
    L2 = -(2im * ω + ρ∞) / r^2 - σ₊ * rp * (2r - rp) / (r * (r - rp))^2

    return (one(ω), L1, L2 + L1^2)  # [A/A, A'/A, A''/A]
end

# ═══════════════════════════════════════════════════════════════════════════════
#  h → u coefficient transformation at a single point
# ═══════════════════════════════════════════════════════════════════════════════

"""
Transform h-derivative coefficients to u-derivative coefficients at one point.

Given C[k, d] (10 × n_h), returns C̃[k, d'] where d' indexes u-derivatives.
The transformation uses the product rule: ∂_r^α(A u) = Σ binom(α,a)(∂^a A)(∂^{α-a} u)
and divides by A_j for each field j.
"""
function _transform_h_to_u(C::Matrix{ComplexF64}, h_map, r, ω, params::KerrParams)
    n_eqs, n_h = size(C)
    # u-derivative terms: same set as h-derivatives (same (j, α, β) patterns)
    C̃ = zeros(ComplexF64, n_eqs, n_h)

    # Group h-terms by field j
    for j in 1:6
        ratios = _Ak_ratios(j, r, ω, params)

        for (d_target, (j_t, b, β_t)) in enumerate(h_map)
            j_t != j && continue
            # C̃ coefficient of ∂_r^b ∂_χ^β u_j =
            # Σ_{d: j_d=j, β_d=β, α_d≥b} C[k,d] × binom(α_d, α_d-b) × ratio[α_d-b]
            for (d_source, (j_s, α_s, β_s)) in enumerate(h_map)
                j_s != j && continue
                β_s != β_t && continue
                a = α_s - b  # derivative order applied to A_j
                (a < 0 || a > 2) && continue
                bc = binomial(α_s, a)
                @views C̃[:, d_target] .+= C[:, d_source] .* (bc * ratios[a + 1])
            end
        end
    end

    return C̃
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Main factored assembly
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_factored_system(cfe, a, N, m; h_term_strings) → METRICSSystem

Build constant D̃₀, D̃₁, D̃₂ via the proper Galerkin approach:

1. At each collocation point, extract field equation coefficients
2. Transform from h-basis to u-basis (A_k factorization)
3. Separate ω-polynomial: C̃ = C̃₀ + C̃₁ω + C̃₂ω²
4. Build D̃_γ by contracting C̃_γ with spectral basis derivatives

The spectral test functions are u_j = T_n(z) P_l^m(χ), and the D̃ matrices
map spectral coefficients to field equation residuals.  They are INDEPENDENT
of ω, enabling efficient Newton-Raphson.
"""
function build_factored_system(cfe::CompiledFieldEquations,
                               a::Float64, N::Int, m::Int;
                               h_term_strings::Vector{String},
                               Nquad::Int=N  # quadrature oversampling: use Nquad+1 points
                               )
    params = KerrParams(a, m)
    rp = r_plus(a)
    am = abs(m)
    sz = N + 1
    bs = sz^2
    Nq = Nquad
    szq = Nq + 1
    bsq = szq^2
    n_h = length(h_term_strings)
    h_map = _parse_h_terms(h_term_strings)

    # Quadrature grid (oversampled if Nquad > N)
    rgrid = RadialGrid(Nq, rp)
    agrid = AngularGrid(Nq, m)  # uses Gauss-Legendre nodes

    # Basis values at quadrature points (truncated to N)
    # T_n for n=0..N at the Nq+1 quadrature z-nodes
    rgrid_basis = RadialGrid(N, rp)  # just for T, dT, d2T at quadrature points
    # Actually we need T_n(z_quadrature) for n=0..N — recompute at quadrature nodes
    T_at_q = zeros(szq, sz)
    dT_at_q = zeros(szq, sz)
    d2T_at_q = zeros(szq, sz)
    for (i, zi) in enumerate(rgrid.z)
        T_at_q[i, 1] = 1.0; T_at_q[i, 2] = zi
        dT_at_q[i, 1] = 0.0; dT_at_q[i, 2] = 1.0
        for n in 2:N
            T_at_q[i, n+1] = 2zi * T_at_q[i, n] - T_at_q[i, n-1]
            dT_at_q[i, n+1] = 2T_at_q[i, n] + 2zi * dT_at_q[i, n] - dT_at_q[i, n-1]
        end
        for n in 0:N
            z2m1 = 1 - zi^2
            d2T_at_q[i, n+1] = abs(z2m1) > 1e-14 ?
                (zi * dT_at_q[i, n+1] - n^2 * T_at_q[i, n+1]) / z2m1 :
                n^2 * (n^2 - 1) / 3.0
        end
    end

    # Quadrature weights: Chebyshev for z, Gauss-Legendre for χ
    w_z = fill(π / szq, szq)               # Gauss-Chebyshev
    _, w_χ = _gauss_legendre(szq)           # Gauss-Legendre

    # ── Build D̃ via weighted Galerkin projection ─────────────────────────
    D0 = zeros(ComplexF64, 10bs, 6bs)
    D1 = zeros(ComplexF64, 10bs, 6bs)
    D2 = zeros(ComplexF64, 10bs, 6bs)

    for (i_r, r_i) in enumerate(rgrid.r)
        for (i_χ, χ_j) in enumerate(agrid.χ)
            C̃s = map([0.0+0im, 1.0+0im, -1.0+0im]) do ω_eval
                C = extract_coefficients_complex(cfe, r_i, χ_j, ω_eval, a)
                _transform_h_to_u(C, h_map, r_i, ω_eval, params)
            end
            C̃0 = C̃s[1]
            C̃1 = (C̃s[2] - C̃s[3]) / 2
            C̃2 = (C̃s[2] + C̃s[3]) / 2 - C̃s[1]

            w_val = w_z[i_r] * w_χ[i_χ]  # quadrature weight

            for (γ, C̃γ) in enumerate((C̃0, C̃1, C̃2))
                Dγ = (D0, D1, D2)[γ]
                _scatter_galerkin!(Dγ, C̃γ, h_map,
                    T_at_q, dT_at_q, d2T_at_q, rgrid, agrid,
                    i_r, i_χ, N, Nq, am, bs, sz, szq, w_val)
            end
        end
    end

    METRICSSystem(D0, D1, D2, N, m, a)
end

"""
Scatter one quadrature point's contribution into D̃_γ via Galerkin projection.
Row (k, n, l) = equation k projected onto T_n(z) P_l(χ).
"""
function _scatter_galerkin!(Dγ, C̃γ, h_map,
                            T_at_q, dT_at_q, d2T_at_q, rgrid, agrid,
                            i_r, i_χ, N, Nq, am, bs, sz, szq, w)
    # Column test functions (u-basis): loop over (j₀, n', l')
    for j₀ in 1:6, n0 in 0:N, l0 in am:(am + N)
        col = (j₀ - 1) * bs + n0 * sz + (l0 - am) + 1
        in_ = n0 + 1
        il_ = l0 - am + 1

        # u-derivatives of T_{n'}(z) P_{l'}(χ) at this quadrature point
        hd = _h_derivatives(
            T_at_q[i_r, in_], dT_at_q[i_r, in_], d2T_at_q[i_r, in_],
            agrid.P[i_χ, il_], agrid.dP[i_χ, il_],
            agrid.d2P[i_χ, il_], agrid.d3P[i_χ, il_],
            rgrid.dz_dr[i_r], rgrid.d2z_dr2[i_r])

        # Contract coefficients with derivatives
        eq_vals = zeros(ComplexF64, 10)
        @inbounds for d in eachindex(h_map)
            j_d, α_d, β_d = h_map[d]
            j_d == j₀ || continue
            dv = _lookup_deriv(hd, α_d, β_d)
            for k in 1:10
                eq_vals[k] += C̃γ[k, d] * dv
            end
        end

        # Project onto row test functions T_n(z) P_l(χ) with quadrature weight
        for n in 0:N, l in am:(am + N)
            row = (0:9) .* bs .+ (n * sz + (l - am) + 1)
            Tn_val = T_at_q[i_r, n + 1]
            Pl_val = agrid.P[i_χ, l - am + 1]
            wt = w * Tn_val * Pl_val
            @inbounds for k in 1:10
                Dγ[row[k], col] += wt * eq_vals[k]
            end
        end
    end
end
