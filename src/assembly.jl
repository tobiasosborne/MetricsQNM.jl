# Spectral matrix assembly: K coefficients × spectral bases → D̃ matrices.
# Eq. 35: D̃(ω) = D̃₀ + D̃₁ω + D̃₂ω²

"""
    METRICSSystem

The assembled spectral system for a given spin and truncation order.
D̃(ω) = D0 + D1·ω + D2·ω² is a rectangular matrix:
  rows = 10·(N+1)², cols = 6·(N+1)².
"""
struct METRICSSystem
    D0::Matrix{ComplexF64}
    D1::Matrix{ComplexF64}
    D2::Matrix{ComplexF64}
    N::Int
    m::Int
    a::Float64
end

Dtilde(sys::METRICSSystem, ω) = sys.D0 + sys.D1 * ω + sys.D2 * ω^2
Dtilde_deriv(sys::METRICSSystem, ω) = sys.D1 + 2 * sys.D2 * ω

# ═══════════════════════════════════════════════════════════════════════════════
#  Second-derivative elimination (Eq. 29)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Chebyshev: (1−z²) d²T_n/dz² = z dT_n/dz − n² T_n
Returns the (1−z²) d²/dz² operator in the T basis.
"""
function _cheb_d2(cb::ChebyshevBasis)
    N = cb.N
    n2 = spdiagm(0 => [Float64(n^2) for n in 0:N])
    cb.Z[2] * cb.D - n2   # z·D - n²·I
end

"""
Legendre: (1−χ²) d²P/dχ² = 2χ dP/dχ − l(l+1) P − m²/(1−χ²) P
Multiply through by (1−χ²):
  (1−χ²)² d²P/dχ² = 2χ·(1−χ²)dP/dχ − l(l+1)(1−χ²)P + m²P
Where (1−χ²)dP/dχ is what leg.D already encodes.
"""
function _leg_d2(lb::LegendreBasis)
    am = abs(lb.m)
    sz = lb.N + 1
    ll1 = spdiagm(0 => [Float64((am + i - 1) * (am + i)) for i in 1:sz])
    one_m_χ2 = sparse(I, sz, sz) - lb.X[3]   # (1 − χ²) operator via I − χ²·
    2 * lb.X[2] * lb.D - ll1 * one_m_χ2 + am^2 * sparse(I, sz, sz)
end

"""
Third angular derivative: (1−χ²)³ d³P/dχ³.

Derived by differentiating the identity for (1−χ²)² d²P/dχ² and using
the product rule. Let D = (1−χ²)d/dχ (what lb.D encodes), then:

  d/dχ[(1−χ²)² d²P] = (1−χ²)³ d³P − 4χ(1−χ²)² d²P

So: (1−χ²)³ d³P = d/dχ[(1−χ²)² d²P] + 4χ(1−χ²)² d²P

The first term d/dχ[f] where f = d2 · P is computed as D·(d2·P)/(1−χ²),
but since D encodes (1−χ²)d/dχ, we have D·(d2) = (1−χ²) d/dχ[(1−χ²)² d²P].
So d/dχ[(1−χ²)² d²P] needs to account for the (1−χ²) factor.

Actually, we use: (1−χ²)³ d³P = (1−χ²) D·(d2·P) + 4χ (1−χ²)² d²P
where D·(d2) is the composition of the D operator with d2.
Wait, more carefully: D encodes (1−χ²)d/dχ, so D applied to the coefficient
vector of f gives the coefficient vector of (1−χ²)df/dχ. If f = d2·c where
c is the coefficient vector of P, then D·d2·c gives coefficients of
(1−χ²) d/dχ[(1−χ²)²d²P/dχ²].

So: (1−χ²)·d/dχ[(1−χ²)²d²P] = D·d2 applied to c.
And: (1−χ²)³ d³P = D·d2 + 4χ·(1−χ²)·d2 ... but we have extra (1−χ²) factors.

Let's be precise. We want operator O3 such that O3·c = coefficients of (1−χ²)³ d³P.
Start from: (1−χ²)² d²P = d2·c (our d2 operator)
Differentiate: d/dχ[(1−χ²)² d²P] = (1−χ²)² d³P − 4χ(1−χ²) d²P

Multiply by (1−χ²):
(1−χ²)·d/dχ[(1−χ²)²d²P] = (1−χ²)³ d³P − 4χ(1−χ²)² d²P

So: (1−χ²)³ d³P = (1−χ²)·d/dχ[(1−χ²)²d²P] + 4χ·(1−χ²)² d²P
                  = D·d2 + 4χ·(1−χ²)·d2  ... no, 4χ(1−χ²)² d²P = 4χ · d2.

Aha: since d2 already encodes (1−χ²)² d²P, we have:
  4χ(1−χ²)² d²P ↔ 4 χ · d2   (the χ multiplication operator times d2)

And D·d2 gives (1−χ²)·d/dχ[(1−χ²)²d²P].

So: (1−χ²)³ d³P ↔ D * d2 + 4 * X[2] * d2
"""
function _leg_d3(lb::LegendreBasis, d2)
    lb.D * d2 + 4 * lb.X[2] * d2
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Operator composition for one PDE term
# ═══════════════════════════════════════════════════════════════════════════════

# z^δ · ∂_z^α  acting on T coefficients
function _z_operator(cb::ChebyshevBasis, α::Int, δ::Int, d2)
    α == 0 && return cb.Z[δ + 1]
    α == 1 && return cb.Z[δ + 1] * cb.D
    α == 2 && return cb.Z[δ + 1] * d2
    error("α = $α > 2 not supported")
end

# χ^σ · ∂_χ^β  acting on P coefficients
function _χ_operator(lb::LegendreBasis, β::Int, σ::Int, d2, d3=nothing)
    β == 0 && return lb.X[σ + 1]
    β == 1 && return lb.X[σ + 1] * lb.D
    β == 2 && return lb.X[σ + 1] * d2
    β == 3 && d3 !== nothing && return lb.X[σ + 1] * d3
    error("β = $β not supported (max 3 with d3 operator)")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Scatter one term into the global D matrix
# ═══════════════════════════════════════════════════════════════════════════════

function _scatter!(D::Matrix{ComplexF64}, K_val, Oz, Oχ,
                   row_off::Int, col_off::Int, sz::Int)
    Oz_s = sparse(Oz)
    Oχ_s = sparse(Oχ)
    cp_z = SparseArrays.getcolptr(Oz_s)
    rv_z = SparseArrays.getrowval(Oz_s)
    nz_z = nonzeros(Oz_s)
    cp_χ = SparseArrays.getcolptr(Oχ_s)
    rv_χ = SparseArrays.getrowval(Oχ_s)
    nz_χ = nonzeros(Oχ_s)

    @inbounds for np in 1:sz                              # source n'
        for iz in cp_z[np]:(cp_z[np+1] - 1)
            n  = rv_z[iz]                                  # target n (1-based)
            vz = nz_z[iz]
            Kvz = K_val * vz
            for lp in 1:sz                                 # source l'
                col = col_off + (np - 1) * sz + lp
                for iχ in cp_χ[lp]:(cp_χ[lp+1] - 1)
                    l  = rv_χ[iχ]                          # target l (1-based)
                    D[row_off + (n - 1) * sz + l, col] += Kvz * nz_χ[iχ]
                end
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Main assembly
# ═══════════════════════════════════════════════════════════════════════════════

"""
    assemble_system(K, basis, a) → METRICSSystem

Build D̃₀, D̃₁, D̃₂ from PDE coefficient tables K and the spectral basis.
Each term K_{k,γ,δ,σ,α,β,j} contributes:
  D_γ[(k,n,l),(j,n',l')] += K_val · [z^δ ∂_z^α T_{n'}]_n · [χ^σ ∂_χ^β P_{l'}]_l
"""
function assemble_system(K::PDECoefficients, basis::SpectralBasis, a::Float64)
    N  = basis.N
    bs = basis.block_size
    sz = N + 1

    D0 = zeros(ComplexF64, 10bs, 6bs)
    D1 = zeros(ComplexF64, 10bs, 6bs)
    D2 = zeros(ComplexF64, 10bs, 6bs)
    Ds = (D0, D1, D2)

    cheb_d2 = _cheb_d2(basis.cheb)
    leg_d2  = _leg_d2(basis.leg)

    # Build d3 operator only if needed (any β=3 terms)
    needs_d3 = any(any(β == 3 for ((_, _, _, _, β, _), _) in eq) for eq in K.equations)
    leg_d3 = needs_d3 ? _leg_d3(basis.leg, leg_d2) : nothing

    for k in 1:10
        isempty(K.equations[k]) && continue
        row_off = (k - 1) * bs

        for ((γ, δ, σ, α, β, j), K_val) in K.equations[k]
            iszero(K_val) && continue
            Oz = _z_operator(basis.cheb, α, δ, cheb_d2)
            Oχ = _χ_operator(basis.leg, β, σ, leg_d2, leg_d3)
            _scatter!(Ds[γ + 1], ComplexF64(K_val), Oz, Oχ,
                      row_off, (j - 1) * bs, sz)
        end
    end

    METRICSSystem(D0, D1, D2, N, basis.m, a)
end

assemble_system(a::Float64, N::Int, m::Int, K::PDECoefficients) =
    assemble_system(K, spectral_basis(N, m), a)
