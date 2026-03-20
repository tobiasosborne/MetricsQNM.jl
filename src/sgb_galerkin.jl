# Numerical Galerkin assembly for sGB correction D̃⁽¹⁾.
#
# Strategy: evaluate correction coefficients C[k,d] on a Chebyshev grid,
# clear known denominators to get a polynomial, fit monomial coefficients,
# and feed into the standard _r_to_z + assemble_system pipeline.

# ═══════════════════════════════════════════════════════════════════════════════
#  Single-point evaluation with clearing
# ═══════════════════════════════════════════════════════════════════════════════

"""
    eval_cleared_sgb(csc, bg, a, r, χ, ω, P, Q, S) → Matrix{ComplexF64}

Evaluate the 10×n_h correction coefficient matrix at a single (r, χ) point,
multiplied by the clearing factor Σ^P Δ^Q (1-χ²)^S to make the result polynomial.
"""
function eval_cleared_sgb(csc::CompiledSGBCorrection,
                           bg, a::Float64, r::Float64, χ::Float64,
                           ω::ComplexF64, P::Int, Q::Int, S::Int)
    hp = sgb_H_params(bg, r, χ)
    C = extract_sgb_coefficients_complex(csc, r, χ, ω, a, hp)

    Sig = r^2 + a^2 * χ^2
    Del = r^2 - 2r + a^2
    s2  = 1 - χ^2
    clearing = Sig^P * Del^Q * s2^S

    return C * clearing
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Grid evaluation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    chebyshev_nodes(N) → Vector{Float64}

N Chebyshev-Gauss-Lobatto nodes on [-1, 1]: cos(πi/(N-1)) for i=0..N-1.
"""
chebyshev_nodes(N::Int) = [cos(π * i / (N - 1)) for i in 0:(N-1)]

"""
    chebyshev_nodes_mapped(N, a, b) → Vector{Float64}

N Chebyshev nodes mapped to [a, b].
"""
function chebyshev_nodes_mapped(N::Int, a::Float64, b::Float64)
    nodes = chebyshev_nodes(N)
    return @. (b + a) / 2 + (b - a) / 2 * nodes
end

"""
    evaluate_grid(csc, bg, a, ω, r_nodes, χ_nodes, P, Q, S; verbose)

Evaluate cleared correction coefficients on the full r × χ grid.
Returns Array{ComplexF64, 4} of size (10, n_h, N_r, N_χ).
"""
function evaluate_grid(csc::CompiledSGBCorrection, bg,
                        a::Float64, ω::ComplexF64,
                        r_nodes::Vector{Float64},
                        χ_nodes::Vector{Float64},
                        P::Int, Q::Int, S::Int;
                        verbose::Bool=false)
    N_r = length(r_nodes)
    N_χ = length(χ_nodes)
    n_h = csc.n_h

    grid = zeros(ComplexF64, 10, n_h, N_r, N_χ)

    total = N_r * N_χ
    done = Threads.Atomic{Int}(0)
    t_start = time()

    # Thread over (i_r, i_χ) pairs
    tasks = [(i_r, i_χ) for i_r in 1:N_r for i_χ in 1:N_χ]
    Threads.@threads for (i_r, i_χ) in tasks
        r = r_nodes[i_r]
        χ = χ_nodes[i_χ]
        grid[:, :, i_r, i_χ] = eval_cleared_sgb(csc, bg, a, r, χ, ω, P, Q, S)

        dc = Threads.atomic_add!(done, 1) + 1
        if verbose && (dc % 50 == 0 || dc == total)
            elapsed = time() - t_start
            @printf("    Grid eval: %d/%d (%.1fs, %.1f pts/s)\n",
                    dc, total, elapsed, dc / elapsed)
            flush(stdout)
        end
    end

    return grid
end

# ═══════════════════════════════════════════════════════════════════════════════
#  2D polynomial fitting: values on (r, χ) grid → monomial coefficients
# ═══════════════════════════════════════════════════════════════════════════════

"""
    fit_r_chi_polynomial(values, r_nodes, χ_nodes, d_r, d_χ)

Fit function values on a (r, χ) tensor product grid as a polynomial:
    f(r, χ) = Σ_{δ=0}^{d_r} Σ_{σ=0}^{d_χ} c[δ,σ] r^δ χ^σ

Returns c as a (d_r+1) × (d_χ+1) complex matrix.

Strategy: for each r_i, fit in χ direction; then for each σ, fit in r direction.
"""
function fit_r_chi_polynomial(values::Matrix{ComplexF64},
                               r_nodes::Vector{Float64},
                               χ_nodes::Vector{Float64},
                               d_r::Int, d_χ::Int)
    N_r = length(r_nodes)
    N_χ = length(χ_nodes)
    @assert size(values) == (N_r, N_χ) "Grid size mismatch"
    @assert N_r >= d_r + 1 "Need at least $(d_r+1) r-nodes for degree $d_r"
    @assert N_χ >= d_χ + 1 "Need at least $(d_χ+1) χ-nodes for degree $d_χ"

    # Step 1: fit χ-direction at each r-node via Vandermonde
    # V_χ[j, σ+1] = χ_nodes[j]^σ
    V_χ = zeros(N_χ, d_χ + 1)
    for j in 1:N_χ, σ in 0:d_χ
        V_χ[j, σ + 1] = χ_nodes[j]^σ
    end

    # c_σ[i_r, σ+1] = coefficient of χ^σ at r = r_nodes[i_r]
    c_σ = zeros(ComplexF64, N_r, d_χ + 1)
    for i_r in 1:N_r
        c_σ[i_r, :] = V_χ \ values[i_r, :]
    end

    # Step 2: fit r-direction for each σ
    V_r = zeros(N_r, d_r + 1)
    for i in 1:N_r, δ in 0:d_r
        V_r[i, δ + 1] = r_nodes[i]^δ
    end

    coeffs = zeros(ComplexF64, d_r + 1, d_χ + 1)
    for σ in 0:d_χ
        coeffs[:, σ + 1] = V_r \ c_σ[:, σ + 1]
    end

    return coeffs
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Convert fitted polynomials to PDECoefficients
# ═══════════════════════════════════════════════════════════════════════════════

"""
    grid_to_pde_coefficients(grid, r_nodes, χ_nodes, h_map, d_r, d_χ; tol)

Convert grid evaluation data to PDECoefficients in r-space.

Arguments:
- grid: (10, n_h, N_r, N_χ) array from evaluate_grid
- h_map: Vector{(j, α_r, β_χ)} from parse_h_term_map
- d_r, d_χ: polynomial degrees for fitting

Returns PDECoefficients ready for _r_to_z.
"""
function grid_to_pde_coefficients(grid::Array{ComplexF64, 4},
                                   r_nodes::Vector{Float64},
                                   χ_nodes::Vector{Float64},
                                   h_map::Vector{Tuple{Int,Int,Int}},
                                   d_r::Int, d_χ::Int;
                                   tol::Float64=1e-12)
    n_eqs = size(grid, 1)
    n_h   = size(grid, 2)

    K = PDECoefficients([Dict{NTuple{6,Int}, ComplexF64}() for _ in 1:n_eqs],
                         fill(d_r, n_eqs), fill(d_χ, n_eqs))

    for k in 1:n_eqs
        for d in 1:n_h
            # Check if this (k, d) pair has any nonzero values
            vals = grid[k, d, :, :]
            maximum(abs, vals) < tol && continue

            j, α, β = h_map[d]

            # Fit polynomial in (r, χ)
            coeffs = fit_r_chi_polynomial(vals, r_nodes, χ_nodes, d_r, d_χ)

            # Store as PDECoefficients entries
            for δ in 0:d_r, σ in 0:d_χ
                c = coeffs[δ + 1, σ + 1]
                abs(c) < tol && continue
                key = (0, δ, σ, α, β, j)  # γ=0 (no ω separation at this stage)
                K.equations[k][key] = get(K.equations[k], key, 0.0im) + c
            end
        end
    end

    return K
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_sgb_galerkin(a, N, m, bg, ω₀; kwargs...) → METRICSSystem

Build the sGB correction matrix D̃⁽¹⁾(ω₀) via numerical Galerkin assembly.

Pipeline:
1. Evaluate correction coefficients on a Chebyshev grid in (r, χ)
2. Clear denominators (multiply by Σ^P Δ^Q (1-χ²)^S)
3. Fit as polynomial in (r, χ) → PDECoefficients in r-space
4. Apply _r_to_z binomial transform → z-space
5. Assemble via assemble_system → D̃⁽¹⁾(ω₀)

Keyword arguments:
- csc: pre-compiled CompiledSGBCorrection (default: compile fresh)
- P, Q, S: clearing powers (default: 4, 2, 2)
- N_r, N_χ: grid sizes (default: 32, 20)
- d_r, d_χ: polynomial degrees for fitting (default: 25, 15)
- R_max_factor: R_max = R_max_factor * r₊ (default: 10.0)
- verbose: print progress (default: false)
"""
function build_sgb_galerkin(a::Float64, N::Int, m::Int,
                             bg, ω₀::ComplexF64;
                             csc=nothing,
                             P::Int=4, Q::Int=2, S::Int=2,
                             N_r::Int=32, N_χ::Int=20,
                             d_r::Int=25, d_χ::Int=15,
                             R_max_factor::Float64=10.0,
                             verbose::Bool=false)
    rp = r_plus(a)
    R_max = R_max_factor * rp

    # Step 0: compile correction evaluator
    if csc === nothing
        verbose && (println("Compiling sGB correction evaluator..."); flush(stdout))
        csc = compile_sgb_correction(m; verbose=verbose)
    end

    # Get h-derivative map: (j, α_r, β_χ) for each of the n_h terms
    h_map = parse_h_term_map(csc.h_deriv_names)

    # Step 1: set up evaluation grid
    verbose && (println("Setting up grid: $(N_r)×$(N_χ) points in r∈[$rp, $R_max] × χ∈[-1,1]"); flush(stdout))
    r_nodes = chebyshev_nodes_mapped(N_r, rp, R_max)
    # Avoid χ = ±1 exactly (where (1-χ²)^S = 0 if S > 0)
    χ_raw = chebyshev_nodes(N_χ)
    χ_nodes = clamp.(χ_raw, -0.9999, 0.9999)

    # Step 2: evaluate on grid
    verbose && (println("Evaluating correction coefficients on grid..."); flush(stdout))
    grid = evaluate_grid(csc, bg, a, ω₀, r_nodes, χ_nodes, P, Q, S; verbose=verbose)

    # Step 3: fit polynomials → PDECoefficients in r-space
    verbose && (println("Fitting polynomials (d_r=$d_r, d_χ=$d_χ)..."); flush(stdout))
    K_r = grid_to_pde_coefficients(grid, r_nodes, χ_nodes, h_map, d_r, d_χ)

    n_terms = sum(length(d) for d in K_r.equations)
    verbose && (println("  $n_terms terms extracted in r-space"); flush(stdout))

    # Step 4: r → z transform
    d_max = 0
    for k in 1:10
        for ((γ, δ, σ, α, β, j), _) in K_r.equations[k]
            d_max = max(d_max, δ)
        end
    end
    verbose && (println("  d_max = $d_max, transforming r→z..."); flush(stdout))

    if d_max > 25
        @warn "d_max=$d_max exceeds max_delta=25 in ChebyshevBasis. " *
              "Consider reducing d_r or increasing clearing powers."
    end

    K_z = _r_to_z(K_r, rp, d_max)

    # Step 5: assemble
    verbose && (println("Assembling D̃⁽¹⁾ correction matrix (N=$N)..."); flush(stdout))
    basis = spectral_basis(N, m)
    D_corr = assemble_system(K_z, basis, a)

    n_rows, n_cols = size(D_corr.D0)
    verbose && (println("  D̃⁽¹⁾ size: $(n_rows) × $(n_cols)"); flush(stdout))

    # Return as METRICSSystem with D0 only (evaluated at fixed ω₀)
    sys = METRICSSystem(D_corr.D0,
                         zeros(ComplexF64, n_rows, n_cols),
                         zeros(ComplexF64, n_rows, n_cols),
                         N, m, a)
    normalize_system!(sys)
    verbose && (println("  Per-equation normalization applied"); flush(stdout))
    return sys
end
