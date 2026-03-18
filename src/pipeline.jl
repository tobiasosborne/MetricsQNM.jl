# Complete Galerkin pipeline: compiled evaluator → polynomial G → z-space K → D̃.
#
# This is the correct implementation of the paper's method:
# 1. Extract coefficients, factorize A_k, fit polynomial in (r, χ)
# 2. Transform r → z = 2r₊/r − 1, clearing (1+z) denominators
# 3. Build PDECoefficients in z-space
# 4. Spectral projection via assembly.jl

export build_system

"""
    build_system(cfe, a, N, m; h_term_strings, P=3, Q=1, S=0, dr=12, dχ=6) → METRICSSystem

The full pipeline from compiled field equations to constant D̃₀, D̃₁, D̃₂.
"""
function build_system(cfe::CompiledFieldEquations, a::Float64, N::Int, m::Int;
                      h_term_strings::Vector{String},
                      P::Int=3, Q::Int=1, S::Int=0, dr::Int=12, dχ::Int=6,
                      nr::Int=25, nχ::Int=15)
    h_map = _parse_h_terms(h_term_strings)
    params = KerrParams(a, m)
    rp = r_plus(a)
    rm = r_minus(a)
    n_h = length(h_map)

    # ── Step 1: Polynomial fit of factored coefficients in (r, χ) ─────────
    r_grid = range(rp + 0.05, 20.0, length=nr)
    χ_grid = range(-0.95, 0.95, length=nχ)
    n_pts = nr * nχ

    # Vandermonde matrix for r-χ polynomial fitting
    n_mono = (dr + 1) * (dχ + 1)
    V = zeros(n_pts, n_mono)
    denoms = zeros(n_pts)
    pt = 0
    for r in r_grid, χ in χ_grid
        pt += 1
        Σv = r^2 + a^2 * χ^2
        Δv = (r - rp) * (r - rm)
        s2 = 1 - χ^2
        denoms[pt] = Σv^P * Δv^Q * s2^S
        col = 0
        for δ in 0:dr, σ in 0:dχ
            col += 1
            V[pt, col] = r^δ * χ^σ
        end
    end
    Vp = pinv(V)

    # For each ω-power γ ∈ {0,1,2}: extract and fit polynomial G_γ
    G_poly = [Dict{Tuple{Int,Int,Int,Int,Int}, Vector{ComplexF64}}() for _ in 1:3]
    # G_poly[γ][(k, j, α, β)] = vector of G_{δσ} coefficients, flattened

    for γ_idx in 1:3
        ω_eval = [0.0+0im, 1.0+0im, -1.0+0im][γ_idx == 1 ? 1 : γ_idx]

        # For the ω-separation, we precompute C̃ at all points for all 3 ω values
        # (same as before)
    end

    # Actually, let's separate ω first, then fit each γ-component
    # Precompute C̃₀, C̃₁, C̃₂ at all grid points
    # The factored C̃(r,χ,ω) = C̃₀ + C̃₁ω + C̃₂ω² is polynomial in ω.
    # To separate: evaluate the FACTORED coefficient at 3 ω values and decompose.
    # The h→u transform must use the SAME ω as the coefficient evaluation.
    C̃_γ = [zeros(ComplexF64, 10, n_h, n_pts) for _ in 1:3]
    ω_eval = [0.0+0im, 1.0+0im, -1.0+0im]
    pt = 0
    for r in r_grid, χ in χ_grid
        pt += 1
        C̃_at_ω = map(ω_eval) do ωe
            C = extract_coefficients_complex(cfe, r, χ, ωe, a)
            _transform_h_to_u(C, h_map, r, ωe, params)
        end
        # ω-polynomial separation
        C̃_γ[1][:, :, pt] = C̃_at_ω[1]                               # C̃₀
        C̃_γ[2][:, :, pt] = (C̃_at_ω[2] - C̃_at_ω[3]) / 2          # C̃₁
        C̃_γ[3][:, :, pt] = (C̃_at_ω[2] + C̃_at_ω[3]) / 2 - C̃_at_ω[1]  # C̃₂
    end

    # Fit polynomials: for each (γ, k, d), fit G_{δσ} in r^δ χ^σ
    G_r = [[Dict{NTuple{6,Int}, ComplexF64}() for _ in 1:10] for _ in 1:3]

    for γ in 1:3, k in 1:10, d in 1:n_h
        j_d, α_d, β_d = h_map[d]

        vals_cleared = [C̃_γ[γ][k, d, pt] * denoms[pt] for pt in 1:n_pts]
        maximum(abs, vals_cleared) < 1e-14 && continue

        # Fit real and imaginary parts
        c_re = Vp * real.(vals_cleared)
        c_im = Vp * imag.(vals_cleared)

        col = 0
        for δ in 0:dr, σ in 0:dχ
            col += 1
            g = complex(c_re[col], c_im[col])
            abs(g) < 1e-12 && continue
            # γ_val = 0 always — the ω-power is handled by storing in K[γ]
            G_r[γ][k][(0, δ, σ, α_d, β_d, j_d)] = g
        end
    end

    # ── Step 2: Transform r → z ───────────────────────────────────────────
    # r^δ = (2rp)^δ (1+z)^{-δ}
    # Multiply equation by (1+z)^dr to clear all denominators
    # (1+z)^{dr-δ} = Σ_{j=0}^{dr-δ} binom(dr-δ, j) z^j
    # Also account for denominators: Σ^P Δ^Q (1-χ²)^S was already multiplied into G,
    # so K coefficients are just the z-transformed G with (1+z)^dr factor.

    dz_max = dr  # maximum z-degree after transform
    K = [PDECoefficients([Dict{NTuple{6,Int}, ComplexF64}() for _ in 1:10],
                         fill(dz_max, 10), fill(dχ, 10)) for _ in 1:3]

    for γ in 1:3, k in 1:10
        for ((_, δ, σ, α, β, j), G_val) in G_r[γ][k]
            coeff_r = (2rp)^δ * G_val
            for jz in 0:(dr - δ)
                K_val = coeff_r * binomial(dr - δ, jz)
                abs(K_val) < 1e-14 && continue
                # γ=0 in key: assembly.jl routes all entries to D₀
                key = (0, jz, σ, α, β, j)
                K[γ].equations[k][key] = get(K[γ].equations[k], key, 0.0im) + K_val
            end
        end
    end

    # ── Step 3: Spectral projection ──────────────────────────────────────
    basis = spectral_basis(N, m)
    D0 = assemble_system(K[1], basis, a).D0
    D1 = assemble_system(K[2], basis, a).D0
    D2 = assemble_system(K[3], basis, a).D0

    METRICSSystem(D0, D1, D2, N, m, a)
end
