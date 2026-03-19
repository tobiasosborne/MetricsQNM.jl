# Collocation assembly: build D̃(ω) by evaluating compiled field equations
# at spectral grid points. Cross-checks the Galerkin (spectral projection) approach.

export build_Dtilde_collocation

# ═══════════════════════════════════════════════════════════════════════════════
#  Numerical evaluation of basis functions and asymptotic factors
# ═══════════════════════════════════════════════════════════════════════════════

# Chebyshev T_n(z) and derivatives at a point
function _chebyshev_vals(z, N)
    T  = zeros(N + 1)   # T_n(z)
    dT = zeros(N + 1)   # dT_n/dz
    T[1] = 1.0; T[2] = z
    dT[1] = 0.0; dT[2] = 1.0
    for n in 2:N
        T[n+1]  = 2z * T[n] - T[n-1]
        dT[n+1] = 2T[n] + 2z * dT[n] - dT[n-1]
    end
    return T, dT
end

# Associated Legendre P_l^m(χ) and dP/dχ at a point
function _legendre_vals(χ, N, m)
    am = abs(m)
    P  = zeros(N + 1)   # P[i] = P_{am+i-1}^m(χ)
    dP = zeros(N + 1)   # dP/dχ

    # P_{|m|}^{|m|}(χ) = (-1)^m (2m-1)!! (1-χ²)^{m/2}
    fact = 1.0
    for k in 1:am; fact *= (2k - 1); end
    P[1] = (-1)^am * fact * (1 - χ^2)^(am / 2)
    # dP_{|m|}^{|m|}/dχ
    dP[1] = am > 0 ? (-1)^am * fact * (-am * χ) * (1 - χ^2)^(am / 2 - 1) : 0.0

    if N ≥ 1
        l = am + 1
        P[2]  = χ * (2am + 1) * P[1]
        dP[2] = (2am + 1) * (P[1] + χ * dP[1])
    end

    for i in 3:(N + 1)
        l = am + i - 1
        P[i]  = ((2l - 1) * χ * P[i-1] - (l + am - 1) * P[i-2]) / (l - am)
        dP[i] = ((2l - 1) * (P[i-1] + χ * dP[i-1]) - (l + am - 1) * dP[i-2]) / (l - am)
    end
    return P, dP
end

# Asymptotic factor A_k(r, ω, params) and its first two r-derivatives
function _asymptotic_and_derivs(r, k::Int, ω, p::KerrParams)
    rp  = r_plus(p.a)
    b_  = b(p.a)
    ΩH  = Omega_H(p.a)

    # A_k = exp(iωr) × r^(2iω + ρ∞) × ((r-rp)/r)^(-i(ω-mΩH)(1+b)/b - ρH)
    ρH  = rho_H(k)
    ρ∞  = rho_inf(k)
    σ₊  = -im * (ω - p.m * ΩH) * (1 + b_) / b_ - ρH

    x = (r - rp) / r   # x = 1 - rp/r
    log_A = im * ω * r + (2im * ω + ρ∞) * log(r) + σ₊ * log(x)
    A = exp(log_A)

    # d(log A)/dr:
    dlog_A = im * ω + (2im * ω + ρ∞) / r + σ₊ * rp / (r * (r - rp))
    dA = A * dlog_A

    # d²(log A)/dr²:
    d2log_A = -(2im * ω + ρ∞) / r^2 - σ₊ * rp * (2r - rp) / (r * (r - rp))^2
    d2A = A * (d2log_A + dlog_A^2)

    return A, dA, d2A
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Build h_j test function derivatives at a point
# ═══════════════════════════════════════════════════════════════════════════════

"""
For u_j = T_n(z) P_l^m(χ), compute h_j = A_j × u_j and all derivatives
∂_r^α ∂_χ^β h_j for α + β ≤ 2 at a single point (r, χ).

Returns a Dict mapping string-sorted h-term names to values.
"""
function _test_function_derivs(j, n, l, r, χ, ω, params, N, m, h_term_map)
    rp = r_plus(params.a)
    z = z_of_r(r, rp)
    dz = dz_dr(r, rp)       # dz/dr
    d2z = 2rp / r^3          # d²z/dr² (from dz/dr = -2rp/r²)... wait let me compute properly
    # dz/dr = -2rp/r² → d²z/dr² = 4rp/r³
    d2z_val = 4rp / r^3

    # Chebyshev values
    Tv, dTv = _chebyshev_vals(z, N)
    T_n  = Tv[n + 1]
    dT_n = dTv[n + 1]

    # Legendre values
    am = abs(m)
    Pv, dPv = _legendre_vals(χ, N, m)
    l_idx = l - am + 1
    P_l  = Pv[l_idx]
    dP_l = dPv[l_idx]

    # Second derivatives
    # d²T_n/dz² via identity: (1-z²) d²T/dz² = z dT/dz - n² T
    if abs(1 - z^2) > 1e-14
        d2T_n = (z * dT_n - n^2 * T_n) / (1 - z^2)
    else
        d2T_n = n^2 * (n^2 - 1) / 3.0  # limiting value at z = ±1
    end

    # d²P_l^m/dχ² via identity
    ll1 = l * (l + 1)
    if abs(1 - χ^2) > 1e-14
        d2P_l = (2χ * dP_l - ll1 * P_l + m^2 / (1 - χ^2) * P_l) / (1 - χ^2)
    else
        d2P_l = 0.0  # boundary value
    end

    # Asymptotic factor and derivatives
    A, dA, d2A = _asymptotic_and_derivs(r, j, ω, params)

    # u = T_n(z) P_l(χ) and derivatives
    u    = T_n * P_l
    du_r = dT_n * dz * P_l              # ∂u/∂r = dT/dz × dz/dr × P
    du_χ = T_n * dP_l                    # ∂u/∂χ = T × dP/dχ

    # ∂²u/∂r² = d²T/dz² × (dz/dr)² × P + dT/dz × d²z/dr² × P
    d2u_rr = (d2T_n * dz^2 + dT_n * d2z_val) * P_l
    # ∂²u/(∂r∂χ) = dT/dz × dz/dr × dP/dχ
    du_rχ = dT_n * dz * dP_l
    # ∂²u/∂χ² = T × d²P/dχ²
    d2u_χχ = T_n * d2P_l

    # h_j = A × u  and derivatives via product rule
    h    = A * u
    dh_r = dA * u + A * du_r
    dh_χ = A * du_χ
    d2h_rr = d2A * u + 2dA * du_r + A * d2u_rr
    dh_rχ  = dA * du_χ + A * du_rχ
    d2h_χχ = A * d2u_χχ

    # Third derivatives (if needed): ∂³h/∂r²∂χ, ∂r∂χ², ∂χ³
    # For now, compute what we need based on h_term_map
    # ∂³u/∂r²∂χ = (d²T/dz² (dz/dr)² + dT/dz d²z/dr²) × dP/dχ
    d3u_rrχ = (d2T_n * dz^2 + dT_n * d2z_val) * dP_l
    d3h_rrχ = d2A * du_χ + 2dA * du_rχ + A * d3u_rrχ   # rough

    # ∂³u/∂r∂χ² = dT/dz × dz/dr × d²P/dχ²
    d3u_rχχ = dT_n * dz * d2P_l
    d3h_rχχ = dA * d2u_χχ + A * d3u_rχχ

    # ∂³P/dχ³ via finite difference on d²P/dχ²
    ε_fd = 1e-5
    if abs(1 - χ^2) > 2ε_fd
        Pv_p, dPv_p = _legendre_vals(χ + ε_fd, N, m)
        Pv_m, dPv_m = _legendre_vals(χ - ε_fd, N, m)
        P_p, dP_p = Pv_p[l_idx], dPv_p[l_idx]
        P_m, dP_m = Pv_m[l_idx], dPv_m[l_idx]
        ll1_ = l * (l + 1)
        d2P_p = (2(χ+ε_fd) * dP_p - ll1_ * P_p + m^2 / (1-(χ+ε_fd)^2) * P_p) / (1-(χ+ε_fd)^2)
        d2P_m = (2(χ-ε_fd) * dP_m - ll1_ * P_m + m^2 / (1-(χ-ε_fd)^2) * P_m) / (1-(χ-ε_fd)^2)
        d3P_l = (d2P_p - d2P_m) / (2ε_fd)
    else
        d3P_l = 0.0  # boundary value
    end
    d3h_χχχ = A * T_n * d3P_l

    # Map to the 40-entry h_term_map ordering
    # h_term_map: index → (j_target, string_repr)
    # We need to set values for all terms matching this j
    vals = zeros(ComplexF64, length(h_term_map))

    for (idx, (j_t, α_r, β_χ)) in enumerate(h_term_map)
        j_t != j && continue
        if     (α_r, β_χ) == (0, 0); vals[idx] = h
        elseif (α_r, β_χ) == (1, 0); vals[idx] = dh_r
        elseif (α_r, β_χ) == (0, 1); vals[idx] = dh_χ
        elseif (α_r, β_χ) == (2, 0); vals[idx] = d2h_rr
        elseif (α_r, β_χ) == (1, 1); vals[idx] = dh_rχ
        elseif (α_r, β_χ) == (0, 2); vals[idx] = d2h_χχ
        elseif (α_r, β_χ) == (2, 1); vals[idx] = d3h_rrχ
        elseif (α_r, β_χ) == (1, 2); vals[idx] = d3h_rχχ
        elseif (α_r, β_χ) == (0, 3); vals[idx] = d3h_χχχ
        end
    end
    return vals
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Parse h-term ordering from compiled field equations
# ═══════════════════════════════════════════════════════════════════════════════

"""
Parse the 40 h-derivative term strings to extract (j, α_r, β_χ) for each.
"""
function parse_h_term_map(h_term_strings::Vector{String})
    map = Vector{Tuple{Int, Int, Int}}()
    for s in h_term_strings
        j = 0
        for jj in 1:6
            if occursin("h$(jj)", s)
                j = jj
                break
            end
        end
        @assert j > 0 "Could not parse j from: $s"

        # Sum derivative orders.  Handles both:
        #   Differential(r, 2)(...)          compact format (order as arg)
        #   Differential(r)(Differential(r)(...))   nested format (implicit order 1)
        α_r = 0
        for m in eachmatch(r"Differential\(r(?:,\s*(\d+))?\)", s)
            α_r += m.captures[1] === nothing ? 1 : parse(Int, m.captures[1])
        end
        β_χ = 0
        for m in eachmatch(r"Differential\(chi(?:,\s*(\d+))?\)", s)
            β_χ += m.captures[1] === nothing ? 1 : parse(Int, m.captures[1])
        end
        push!(map, (j, α_r, β_χ))
    end
    return map
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Main collocation assembly
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_Dtilde_collocation(cfe, a, ω, N, m; h_term_strings) → Matrix{ComplexF64}

Build D̃(ω) via collocation: evaluate the compiled field equations at
Chebyshev-Legendre grid points with test functions h_j = A_j T_n P_l^m.

Returns the (10(N+1)² × 6(N+1)²) matrix D̃(ω).
"""
function build_Dtilde_collocation(cfe::CompiledFieldEquations,
                                  a::Float64, ω::ComplexF64, N::Int, m::Int;
                                  h_term_strings::Vector{String})
    params = KerrParams(a, m)
    rp = r_plus(a)
    am = abs(m)
    bs = (N + 1)^2
    n_h = length(h_term_strings)

    # Parse h-term ordering
    h_map = parse_h_term_map(h_term_strings)

    # Collocation nodes
    z_nodes = [cos(π * k / N) for k in 0:N]
    χ_nodes = [cos(π * k / N) for k in 0:N]  # use Chebyshev nodes for both

    # Avoid boundary singularities
    z_nodes = clamp.(z_nodes, -0.9999, 0.9999)
    χ_nodes = clamp.(χ_nodes, -0.9999, 0.9999)

    n_eqs = 10
    n_fields = 6
    nrows = n_eqs * bs
    ncols = n_fields * bs
    D = zeros(ComplexF64, nrows, ncols)

    args = zeros(Float64, cfe.n_args)
    args[4] = imag(ω)  # ω_im
    args[5] = 0.0       # iu_sym = 0 for real part
    args[6] = a

    # For each column (j₀, n₀, l₀):
    for j₀ in 1:n_fields
        for n₀ in 0:N
            for l₀ in am:(am + N)
                col_idx = (j₀ - 1) * bs + nl_index(n₀, l₀, N, m)

                # For each collocation point:
                for (iz, z_i) in enumerate(z_nodes)
                    r_i = r_of_z(z_i, rp)
                    for (iχ, χ_j) in enumerate(χ_nodes)
                        # Compute h-derivative values for this test function
                        hvals = _test_function_derivs(
                            j₀, n₀, l₀, r_i, χ_j, ω, params, N, m, h_map)

                        # Evaluate with real part (iu=0)
                        args[1] = r_i
                        args[2] = χ_j
                        args[3] = real(ω)
                        args[5] = 0.0  # iu = 0
                        args[7:end] .= real.(hvals)
                        res_re = [cfe.funcs[k](args) for k in 1:n_eqs]

                        # Evaluate with iu=1 for imaginary part
                        args[5] = 1.0
                        args[7:end] .= real.(hvals)
                        res_im = [cfe.funcs[k](args) for k in 1:n_eqs]

                        # But wait: we also need imaginary parts of hvals...
                        # The full evaluation needs complex arithmetic.
                        # Use: f(h_re + i h_im) = f(h_re) + i f(h_im) for linear f
                        # And: f(iu=0, h_re) + i f(iu=1, h_re) + i f(iu=0, h_im) - f(iu=1, h_im)

                        # For real hvals with iu trick:
                        # result = f(iu=0) + im * f(iu=1) with real inputs
                        # Then for complex hvals: need to separate
                        # Actually this is getting complicated. Let me simplify.

                        # Since the equations are linear in h, and we're passing
                        # complex hvals through a real-only interface:
                        # result = eval(Re(h)) + i * eval(Im(h))
                        # But each eval also needs the iu trick for ω:
                        # eval(Re(h), iu=0) + i*eval(Re(h), iu=1) + i*eval(Im(h), iu=0) - eval(Im(h), iu=1)

                        # This requires 4 evaluations per point. Let me do it properly.
                        hr = real.(hvals)
                        hi = imag.(hvals)

                        # f(iu=0, h_re)
                        args[5] = 0.0; args[7:end] .= hr
                        f00 = [cfe.funcs[k](args) for k in 1:n_eqs]
                        # f(iu=1, h_re)
                        args[5] = 1.0; args[7:end] .= hr
                        f10 = [cfe.funcs[k](args) for k in 1:n_eqs]
                        # f(iu=0, h_im)
                        args[5] = 0.0; args[7:end] .= hi
                        f01 = [cfe.funcs[k](args) for k in 1:n_eqs]
                        # f(iu=1, h_im)
                        args[5] = 1.0; args[7:end] .= hi
                        f11 = [cfe.funcs[k](args) for k in 1:n_eqs]

                        # Full complex result:
                        # f(ω_re + i ω_im, h_re + i h_im)
                        # = [f00 - f11] + i [f10 + f01]
                        result = (f00 - f11) + im * (f10 + f01)

                        # Store in D̃
                        for k in 1:n_eqs
                            row = (k - 1) * bs + (iz - 1) * (N + 1) + iχ
                            D[row, col_idx] = result[k]
                        end
                    end
                end
            end
        end
    end

    return D
end
