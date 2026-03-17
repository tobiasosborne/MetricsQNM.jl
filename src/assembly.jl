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
function _χ_operator(lb::LegendreBasis, β::Int, σ::Int, d2)
    β == 0 && return lb.X[σ + 1]
    β == 1 && return lb.X[σ + 1] * lb.D
    β == 2 && return lb.X[σ + 1] * d2
    error("β = $β > 2 not supported")
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

    for k in 1:10
        isempty(K.equations[k]) && continue
        row_off = (k - 1) * bs

        for ((γ, δ, σ, α, β, j), K_val) in K.equations[k]
            iszero(K_val) && continue
            Oz = _z_operator(basis.cheb, α, δ, cheb_d2)
            Oχ = _χ_operator(basis.leg, β, σ, leg_d2)
            _scatter!(Ds[γ + 1], ComplexF64(K_val), Oz, Oχ,
                      row_off, (j - 1) * bs, sz)
        end
    end

    METRICSSystem(D0, D1, D2, N, basis.m, a)
end

assemble_system(a::Float64, N::Int, m::Int, K::PDECoefficients) =
    assemble_system(K, spectral_basis(N, m), a)
