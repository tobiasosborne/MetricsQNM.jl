# Z-space polynomial coefficient extraction with A_k factorization
# and exact Chebyshev-basis assembly.
#
# Key insight: A_k factorization is ESSENTIAL for convergence, but introduces
# rational r-dependence (poles at r=0 and r=r_+).  We clear these poles by
# multiplying by r²(r-r_+)² before the Chebyshev transform, preserving the
# polynomial structure needed for exact spectral inner products.

export build_system_zspace

# ═══════════════════════════════════════════════════════════════════════════════
#  Chebyshev DCT
# ═══════════════════════════════════════════════════════════════════════════════

"""
Chebyshev expansion coefficients from values at Gauss nodes z_k = cos(π(2k-1)/(2N)).
"""
function _chebyshev_coeffs(values::AbstractVector{T}, n_coeffs::Int) where T
    N = length(values)
    z = [cos(π * (2k - 1) / (2N)) for k in 1:N]

    # Precompute T_p(z_k)
    Tm = zeros(N, n_coeffs)
    for i in 1:N
        Tm[i, 1] = 1.0
        n_coeffs > 1 && (Tm[i, 2] = z[i])
        for p in 2:n_coeffs-1
            Tm[i, p+1] = 2z[i] * Tm[i, p] - Tm[i, p-1]
        end
    end

    coeffs = zeros(T, n_coeffs)
    for p in 1:n_coeffs
        s = zero(T)
        @inbounds for k in 1:N; s += values[k] * Tm[k, p]; end
        coeffs[p] = (p == 1 ? one(Float64) : 2.0) / N * s
    end
    return coeffs
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Coefficient table type
# ═══════════════════════════════════════════════════════════════════════════════

struct ChebyshevPDECoefficients
    equations::Vector{Dict{NTuple{5,Int}, ComplexF64}}
    d_z::Int
    d_chi::Int
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Extraction with A_k factorization
# ═══════════════════════════════════════════════════════════════════════════════

"""
    extract_chebyshev_coefficients(cfe, a, m; ...) → (K0, K1, K2)

Extract field equation coefficients in Chebyshev-z × monomial-χ basis,
WITH A_k factorization for proper asymptotic behavior.

The cleared polynomial is:
  F(z,χ,ω) = (1+z)^{d_z} × Σ^P Δ^Q (1-χ²)^S × r²(r-r₊)² × C̃(r(z),χ,ω,a)
where C̃ = A_k-factored coefficient matrix.

The extra r²(r-r₊)² factor clears the poles from the A_k log-derivatives.
"""
function extract_chebyshev_coefficients(cfe::CompiledFieldEquations,
                                        a::Float64, m::Int;
                                        h_term_strings::Vector{String},
                                        P::Int=3, Q::Int=1, S::Int=1,
                                        d_z::Int=24, d_chi::Int=10,
                                        nz::Int=0, nchi::Int=0,
                                        use_Ak::Bool=true)
    h_map = _parse_h_terms(h_term_strings)
    params = KerrParams(a, m)
    rp = r_plus(a)
    rm = r_minus(a)
    n_h = length(h_map)

    Nz = nz > 0 ? nz : 2(d_z + 1)
    Nχ = nchi > 0 ? nchi : 2(d_chi + 1)

    z_nodes = [cos(π * (2k - 1) / (2Nz)) for k in 1:Nz]
    χ_nodes = [cos(π * (2k - 1) / (2Nχ)) for k in 1:Nχ]

    # χ-monomial Vandermonde
    Vχ = zeros(Nχ, d_chi + 1)
    for (j, χ) in enumerate(χ_nodes), σ in 0:d_chi
        Vχ[j, σ+1] = χ^σ
    end
    Vχ_qr = qr(Vχ)

    ω_evals = [ComplexF64(0), ComplexF64(1), ComplexF64(-1)]
    G_at_ω = [zeros(ComplexF64, 10, n_h, d_z + 1, d_chi + 1) for _ in 1:3]

    for (ω_idx, ω_eval) in enumerate(ω_evals)
        Tp_at_χ = zeros(ComplexF64, 10, n_h, d_z + 1, Nχ)

        for (j_χ, χ_val) in enumerate(χ_nodes)
            F_re = zeros(10, n_h, Nz)
            F_im = zeros(10, n_h, Nz)

            for (i_z, z_val) in enumerate(z_nodes)
                r_val = r_of_z(z_val, rp)

                # Extract coefficient matrix
                C = extract_coefficients_complex(cfe, r_val, χ_val, ω_eval, a)

                # A_k factorization: transform h-basis → u-basis
                if use_Ak
                    C = _transform_h_to_u(C, h_map, r_val, ω_eval, params)
                end

                # Clearing factor: remove all denominators to get polynomial
                Σv = r_val^2 + a^2 * χ_val^2
                Δv = (r_val - rp) * (r_val - rm)
                s2 = 1 - χ_val^2

                # Clearing factor: removes all rational denominators
                # Σ^P Δ^Q (1-χ²)^S clears metric-inverse poles
                denom = Σv^P * Δv^Q * s2^S

                # A_k pole clearing: r²(r-r₊)² clears A_k log-derivative poles
                if use_Ak
                    denom *= r_val^2 * (r_val - rp)^2
                end

                # z-space clearing: (1+z)^d_z clears r→z denominators.
                # This makes F a true polynomial in z, enabling exact Chebyshev
                # representation and spectral inner products.
                scale = denom * (1 + z_val)^d_z

                for k in 1:10, d in 1:n_h
                    v = C[k, d] * scale
                    F_re[k, d, i_z] = real(v)
                    F_im[k, d, i_z] = imag(v)
                end
            end

            # Chebyshev transform in z
            for k in 1:10, d in 1:n_h
                c_re = _chebyshev_coeffs(@view(F_re[k, d, :]), d_z + 1)
                c_im = _chebyshev_coeffs(@view(F_im[k, d, :]), d_z + 1)
                for p in 1:(d_z+1)
                    Tp_at_χ[k, d, p, j_χ] = complex(c_re[p], c_im[p])
                end
            end
        end

        # Fit monomial in χ for each (k, d, p)
        for k in 1:10, d in 1:n_h, p in 0:d_z
            vals = collect(@view Tp_at_χ[k, d, p+1, :])
            maximum(abs, vals) < 1e-14 && continue
            c_re = Vχ_qr \ real.(vals)
            c_im = Vχ_qr \ imag.(vals)
            for σ in 0:d_chi
                g_val = complex(c_re[σ+1], c_im[σ+1])
                abs(g_val) < 1e-12 && continue
                G_at_ω[ω_idx][k, d, p+1, σ+1] = g_val
            end
        end
    end

    # Separate ω: D̃(ω) = D₀ + D₁ω + D₂ω²
    K = [ChebyshevPDECoefficients(
            [Dict{NTuple{5,Int}, ComplexF64}() for _ in 1:10], d_z, d_chi)
         for _ in 1:3]

    for k in 1:10, d in 1:n_h
        j_d, α_d, β_d = h_map[d]
        for p in 0:d_z, σ in 0:d_chi
            g0  = G_at_ω[1][k, d, p+1, σ+1]
            g1  = G_at_ω[2][k, d, p+1, σ+1]
            gm1 = G_at_ω[3][k, d, p+1, σ+1]

            G0 = g0
            G1 = (g1 - gm1) / 2
            G2 = (g1 + gm1) / 2 - g0

            for (γ, Gval) in enumerate((G0, G1, G2))
                abs(Gval) < 1e-13 && continue
                key = (p, σ, α_d, β_d, j_d)
                K[γ].equations[k][key] =
                    get(K[γ].equations[k], key, 0.0im) + Gval
            end
        end
    end

    return K[1], K[2], K[3]
end

# ═══════════════════════════════════════════════════════════════════════════════
#  T_p multiplication operator (Chebyshev product rule)
# ═══════════════════════════════════════════════════════════════════════════════

function _chebyshev_product_matrix(p::Int, N::Int)
    sz = N + 1
    p == 0 && return sparse(I, sz, sz)
    I_r = Int[]; I_c = Int[]; V = Float64[]
    for n in 0:N
        lo = abs(p - n)
        lo ≤ N && (push!(I_r, lo+1); push!(I_c, n+1); push!(V, 0.5))
        hi = p + n
        hi ≤ N && (push!(I_r, hi+1); push!(I_c, n+1); push!(V, 0.5))
    end
    sparse(I_r, I_c, V, sz, sz)
end

function _Tz_operator(cb::ChebyshevBasis, α::Int, p::Int, d2)
    Tp = _chebyshev_product_matrix(p, cb.N)
    α == 0 && return Tp
    α == 1 && return Tp * cb.D
    α == 2 && return Tp * d2
    error("α = $α > 2 not supported")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Assembly from Chebyshev-z × monomial-χ coefficients
# ═══════════════════════════════════════════════════════════════════════════════

function assemble_system_chebyshev(K::ChebyshevPDECoefficients,
                                   basis::SpectralBasis, a::Float64)
    N = basis.N; bs = basis.block_size; sz = N + 1
    D = zeros(ComplexF64, 10bs, 6bs)

    cheb_d2 = _cheb_d2(basis.cheb)
    leg_d2  = _leg_d2(basis.leg)

    for k in 1:10
        isempty(K.equations[k]) && continue
        row_off = (k - 1) * bs
        for ((p, σ, α, β, j), K_val) in K.equations[k]
            iszero(K_val) && continue
            p > N && continue
            Oz = _Tz_operator(basis.cheb, α, p, cheb_d2)
            Oχ = _χ_operator(basis.leg, β, σ, leg_d2)
            _scatter!(D, ComplexF64(K_val), Oz, Oχ, row_off, (j-1)*bs, sz)
        end
    end

    METRICSSystem(D, zeros(ComplexF64, 10bs, 6bs), zeros(ComplexF64, 10bs, 6bs),
                  N, basis.m, a)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Full pipeline
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_system_zspace(cfe, a, N, m; ...) → METRICSSystem

Build D̃₀, D̃₁, D̃₂ via z-space Chebyshev extraction with A_k factorization.
"""
function build_system_zspace(cfe::CompiledFieldEquations,
                             a::Float64, N::Int, m::Int;
                             h_term_strings::Vector{String},
                             P::Int=3, Q::Int=1, S::Int=1,
                             d_z::Int=24, d_chi::Int=10,
                             use_Ak::Bool=true,
                             verbose::Bool=false)
    verbose && println("Extracting Chebyshev coefficients (A_k=$use_Ak)...")
    K0, K1, K2 = extract_chebyshev_coefficients(
        cfe, a, m; h_term_strings, P, Q, S, d_z, d_chi, use_Ak)

    if verbose
        for (name, K) in [("K0", K0), ("K1", K1), ("K2", K2)]
            n = sum(length(d) for d in K.equations)
            println("  $name: $n terms")
        end
    end

    verbose && println("Assembling D̃ matrices (N=$N)...")
    basis = spectral_basis(N, m)
    D0 = assemble_system_chebyshev(K0, basis, a).D0
    D1 = assemble_system_chebyshev(K1, basis, a).D0
    D2 = assemble_system_chebyshev(K2, basis, a).D0

    METRICSSystem(D0, D1, D2, N, m, a)
end
