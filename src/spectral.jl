# Chebyshev and associated Legendre spectral bases with linearization coefficients.
# Eqs. 20, 29 of arxiv:2312.08435.

# ═══════════════════════════════════════════════════════════════════════════════
#  Chebyshev basis on z ∈ [-1, +1]
# ═══════════════════════════════════════════════════════════════════════════════

"""
    ChebyshevBasis(N)

Precomputed Chebyshev T_n operators truncated at order N.
Provides derivative and multiplication-by-z^δ as sparse matrices
acting on coefficient vectors in the T_n basis.
"""
struct ChebyshevBasis
    N::Int
    D::SparseMatrixCSC{Float64,Int}           # dT_n/dz → T coefficients
    Z::Vector{SparseMatrixCSC{Float64,Int}}    # Z[δ+1]: z^δ · T_n → T coefficients
end

"""
    chebyshev_basis(N) → ChebyshevBasis

Build Chebyshev operators for n = 0, …, N.

The derivative matrix encodes dT_n/dz = 2n ∑_{k<n, k+n odd} T_k / c_k
where c_0 = 2, c_k = 1 for k ≥ 1.
"""
function chebyshev_basis(N::Int)
    # ── Derivative matrix D: (dT_n/dz)[k] ────────────────────────────────────
    # dT_n/dz = 2n ∑_{k<n, k+n odd} T_k / c_k   where c_0=2, c_k=1
    D = spzeros(N + 1, N + 1)  # rows = output index k, cols = input index n
    for n in 1:N
        # k + n must be odd, i.e., k and n have opposite parity
        for k in (iseven(n) ? 1 : 0):2:(n-1)
            c_k = k == 0 ? 2.0 : 1.0
            D[k+1, n+1] = 2n / c_k
        end
    end

    # ── Multiplication matrices Z[δ+1]: z^δ · T_n → T coefficients ───────────
    # z · T_n = (T_{n+1} + T_{n-1}) / 2,  with T_{-1} = T_1
    Z1 = spzeros(N + 1, N + 1)
    for n in 0:N
        if n + 1 ≤ N
            Z1[n+2, n+1] = 0.5   # T_{n+1} contribution
        end
        if n - 1 ≥ 0
            Z1[n, n+1] = 0.5     # T_{n-1} contribution
        else
            Z1[2, n+1] += 0.5    # T_{-1} = T_1
        end
    end

    # Build higher powers by repeated multiplication: z^δ = Z1 * z^{δ-1}
    max_delta = 12  # enough for any practical PDE
    Zs = Vector{SparseMatrixCSC{Float64,Int}}(undef, max_delta + 1)
    Zs[1] = sparse(I, N + 1, N + 1)  # z^0 = identity
    Zs[2] = Z1
    for δ in 2:max_delta
        Zs[δ+1] = Z1 * Zs[δ]
    end

    ChebyshevBasis(N, D, Zs)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Associated Legendre basis on χ ∈ [-1, +1]
# ═══════════════════════════════════════════════════════════════════════════════

"""
    LegendreBasis(N, m)

Precomputed associated Legendre P_l^|m| operators.
Index l runs from |m| to |m| + N, giving N + 1 basis functions.
"""
struct LegendreBasis
    N::Int
    m::Int
    D::SparseMatrixCSC{Float64,Int}           # dP_l^m/dχ → P coefficients
    X::Vector{SparseMatrixCSC{Float64,Int}}    # X[σ+1]: χ^σ · P_l^m → P coefficients
end

"""
    legendre_basis(N, m) → LegendreBasis

Build associated Legendre operators for l = |m|, …, |m| + N.

Recurrences:  χ P_l^m = [α⁺_l P_{l+1}^m + α⁻_l P_{l-1}^m]
with α⁺_l = (l - m + 1)/(2l + 1), α⁻_l = (l + m)/(2l + 1).

Derivative:  dP_l^m/dχ = 1/(1-χ²) [-l χ P_l^m + (l+m) P_{l-1}^m]
But since 1/(1-χ²) is singular, we use the equivalent recurrence that
maps dP_l^m/dχ to a linear combination of P_{l'}^m:

  (1-χ²) dP_l^m/dχ = -l χ P_l^m + (l+m) P_{l-1}^m

We fold this into the spectral projection: the PDE's (1-χ²) factors
from second-derivative elimination (Eq. 29) cancel the singular denominator.

For the first derivative, we use the three-term relation:
  dP_l^m/dχ = β⁺_l P_{l+1}^m + β⁻_l P_{l-1}^m
where:
  β⁺_l = -(l + 1)(l - m + 1) / [(2l + 1)(l + 1)]  ... wait, let me use the correct form.

Actually, the recurrence for the derivative of associated Legendre is:
  (1 - χ²) dP_l^m/dχ = (l+m) P_{l-1}^m - l χ P_l^m

And χ P_l^m = α⁺_l P_{l+1}^m + α⁻_l P_{l-1}^m,  so:
  (1 - χ²) dP_l^m/dχ = (l+m) P_{l-1}^m - l [α⁺_l P_{l+1}^m + α⁻_l P_{l-1}^m]
                       = [(l+m) - l α⁻_l] P_{l-1}^m - l α⁺_l P_{l+1}^m
                       = [(l+m) - l(l+m)/(2l+1)] P_{l-1}^m - l(l-m+1)/(2l+1) P_{l+1}^m
                       = (l+m)(l+1)/(2l+1) P_{l-1}^m  - l(l-m+1)/(2l+1) P_{l+1}^m

This gives (1-χ²) dP_l^m/dχ as a combination of P_{l±1}^m.
We store this as the "derivative" operator, with the understanding that
the PDE formulation already accounts for the (1-χ²) factor.
"""
function legendre_basis(N::Int, m::Int)
    am = abs(m)
    sz = N + 1  # number of basis functions: l = am, am+1, ..., am+N

    # ── χ-multiplication: χ P_l^m = α⁺ P_{l+1}^m + α⁻ P_{l-1}^m ────────────
    X1 = spzeros(sz, sz)
    for i in 1:sz
        l = am + i - 1
        α_plus  = (l - am + 1) / (2l + 1)
        α_minus = (l + am) / (2l + 1)
        if i + 1 ≤ sz
            X1[i+1, i] = α_plus    # P_{l+1}^m coefficient
        end
        if i - 1 ≥ 1
            X1[i-1, i] = α_minus   # P_{l-1}^m coefficient
        end
    end

    # Higher powers by repeated multiplication
    max_sigma = 12
    Xs = Vector{SparseMatrixCSC{Float64,Int}}(undef, max_sigma + 1)
    Xs[1] = sparse(I, sz, sz)
    Xs[2] = X1
    for σ in 2:max_sigma
        Xs[σ+1] = X1 * Xs[σ]
    end

    # ── Derivative: (1-χ²) dP_l^m/dχ → P coefficients ────────────────────────
    # (1-χ²) dP_l^m/dχ = (l+m)(l+1)/(2l+1) P_{l-1}^m - l(l-m+1)/(2l+1) P_{l+1}^m
    D = spzeros(sz, sz)
    for i in 1:sz
        l = am + i - 1
        c_minus = (l + am) * (l + 1) / (2l + 1)
        c_plus  = -l * (l - am + 1) / (2l + 1)
        if i - 1 ≥ 1
            D[i-1, i] = c_minus
        end
        if i + 1 ≤ sz
            D[i+1, i] = c_plus
        end
    end

    LegendreBasis(N, m, D, Xs)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Combined spectral basis
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SpectralBasis(N, m)

Combined Chebyshev × Legendre basis for the spectral expansion
u_k(z, χ) = ∑_n ∑_l v_k^{nl} T_n(z) P_l^{|m|}(χ).
"""
struct SpectralBasis
    N::Int
    m::Int
    cheb::ChebyshevBasis
    leg::LegendreBasis
    block_size::Int    # (N+1)^2 = number of (n,l) pairs per field
end

function spectral_basis(N::Int, m::Int)
    cheb = chebyshev_basis(N)
    leg  = legendre_basis(N, m)
    SpectralBasis(N, m, cheb, leg, (N + 1)^2)
end

"""
    nl_index(n, l, N, m) → Int

Linear index for the (n, l) spectral mode in a block of size (N+1)².
n = 0, …, N;  l = |m|, …, |m| + N.
"""
nl_index(n::Int, l::Int, N::Int, m::Int) = n * (N + 1) + (l - abs(m)) + 1
