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
#  Fit validation: check polynomial accuracy at test points
# ═══════════════════════════════════════════════════════════════════════════════

"""
    validate_fit(csc, bg, a, ω, K_r, h_map, P, Q, S; n_test=5)

Evaluate the fitted polynomial and the actual C[k,d] at random test points
not on the original grid. Returns the maximum relative error.
"""
function validate_fit(csc::CompiledSGBCorrection, bg, a::Float64, ω::ComplexF64,
                       K_r::PDECoefficients, h_map::Vector{Tuple{Int,Int,Int}},
                       P::Int, Q::Int, S::Int;
                       n_test::Int=5, verbose::Bool=false)
    rp = r_plus(a)
    max_err = 0.0
    max_val = 0.0

    for _ in 1:n_test
        r = rp + (10rp - rp) * rand()
        χ = -0.99 + 1.98 * rand()

        C_actual = eval_cleared_sgb(csc, bg, a, r, χ, ω, P, Q, S)

        # Evaluate fitted polynomial
        for k in 1:10
            for d in 1:length(h_map)
                j, α, β = h_map[d]
                fitted_val = 0.0im
                for ((γ, δ, σ, α2, β2, j2), c) in K_r.equations[k]
                    (α2 == α && β2 == β && j2 == j) || continue
                    fitted_val += c * r^δ * χ^σ
                end
                actual_val = C_actual[k, d]
                err = abs(fitted_val - actual_val)
                max_val = max(max_val, abs(actual_val))
                max_err = max(max_err, err)
            end
        end
    end

    rel_err = max_val > 0 ? max_err / max_val : max_err
    verbose && @printf("  Fit validation: max |err| = %.2e, max |val| = %.2e, rel = %.2e\n",
                        max_err, max_val, rel_err)
    return rel_err
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_sgb_galerkin(a, N, m, bg, ω₀; kwargs...) → METRICSSystem

Build the sGB correction matrix D̃⁽¹⁾(ω₀) via numerical Galerkin assembly.

Pipeline:
1. Evaluate correction coefficients C[k,d] on a Chebyshev grid in (z, χ)
2. C[k,d] are already O(1) — no denominator clearing needed
3. Fit as polynomial in (z, χ) via Chebyshev interpolation
4. Store as PDECoefficients in z-space → assemble_system → D̃⁽¹⁾(ω₀)

Keyword arguments:
- csc: pre-compiled CompiledSGBCorrection (default: compile fresh)
- N_z, N_χ: grid sizes (default: 24, 16)
- R_max_factor: R_max = R_max_factor * r₊ (default: 10.0)
- verbose: print progress (default: false)
"""
function build_sgb_galerkin(a::Float64, N::Int, m::Int,
                             bg, ω₀::ComplexF64;
                             csc=nothing,
                             N_z::Int=24, N_χ::Int=16,
                             # Legacy clearing params (now unused, kept for compat)
                             P::Int=0, Q::Int=0, S::Int=0,
                             N_r::Int=0, N_χ_old::Int=0,
                             d_r::Int=0, d_χ::Int=0,
                             R_max_factor::Float64=10.0,
                             verbose::Bool=false)
    rp = r_plus(a)

    # Step 0: compile correction evaluator
    if csc === nothing
        verbose && (println("Compiling sGB correction evaluator..."); flush(stdout))
        csc = compile_sgb_correction(m; verbose=verbose)
    end

    # Get h-derivative map: (j, α_r, β_χ) for each of the n_h terms
    h_map = parse_h_term_map(csc.h_deriv_names)

    # Step 1: set up evaluation grid in z-space
    z_nodes = chebyshev_nodes(N_z)
    # Clamp z to avoid exact endpoints (horizon z=1, infinity z=-1)
    z_nodes = clamp.(z_nodes, -0.9999, 0.9999)
    r_nodes = [r_of_z(z, rp) for z in z_nodes]

    χ_nodes = chebyshev_nodes(N_χ)
    χ_nodes = clamp.(χ_nodes, -0.9999, 0.9999)

    verbose && (@printf("Setting up grid: %d×%d in z∈[-1,1] (r∈[%.1f, %.1f]) × χ∈[-1,1]\n",
                         N_z, N_χ, minimum(r_nodes), maximum(r_nodes)); flush(stdout))

    # Step 2: evaluate C[k,d] on grid (NO clearing — raw values are O(1))
    verbose && (println("Evaluating correction coefficients on grid..."); flush(stdout))
    grid = evaluate_grid(csc, bg, a, ω₀, r_nodes, χ_nodes, 0, 0, 0; verbose=verbose)

    # Step 3: fit as polynomial in (z, χ) via Chebyshev interpolation
    d_z = N_z - 1  # max polynomial degree in z
    d_χ_fit = N_χ - 1  # max polynomial degree in χ
    verbose && (@printf("Fitting polynomials in z-space (d_z=%d, d_χ=%d)...\n", d_z, d_χ_fit); flush(stdout))

    K_z = _fit_zspace_coefficients(grid, z_nodes, χ_nodes, h_map, d_z, d_χ_fit;
                                     tol=1e-12, verbose=verbose)

    n_terms = sum(length(d) for d in K_z.equations)
    verbose && (println("  $n_terms terms in z-space"); flush(stdout))

    # Validate fit quality
    if verbose
        _validate_fit_zspace(csc, bg, a, ω₀, K_z, h_map, rp;
                              n_test=8, verbose=true)
    end

    # Step 4: assemble directly from z-space coefficients
    verbose && (println("Assembling D̃⁽¹⁾ correction matrix (N=$N)..."); flush(stdout))
    basis = spectral_basis(N, m)
    D_corr = assemble_system(K_z, basis, a)

    n_rows, n_cols = size(D_corr.D0)
    verbose && (println("  D̃⁽¹⁾ size: $(n_rows) × $(n_cols)"); flush(stdout))

    sys = METRICSSystem(D_corr.D0,
                         zeros(ComplexF64, n_rows, n_cols),
                         zeros(ComplexF64, n_rows, n_cols),
                         N, m, a)
    normalize_system!(sys)
    verbose && (println("  Per-equation normalization applied"); flush(stdout))
    return sys
end

# ═══════════════════════════════════════════════════════════════════════════════
#  z-space Chebyshev interpolation
# ═══════════════════════════════════════════════════════════════════════════════

"""
Fit grid values as TRUNCATED polynomial in (z, χ) via least-squares,
storing as PDECoefficients in z-space.

Uses Chebyshev-Gauss-Lobatto nodes for stability and least-squares fitting
to truncate at degree (d_z, d_χ), which regularizes the non-polynomial
tail of C[k,d] at large r (z → -1).
"""
function _fit_zspace_coefficients(grid::Array{ComplexF64, 4},
                                   z_nodes::Vector{Float64},
                                   χ_nodes::Vector{Float64},
                                   h_map::Vector{Tuple{Int,Int,Int}},
                                   d_z::Int, d_χ::Int;
                                   tol::Float64=1e-12,
                                   verbose::Bool=false)
    n_eqs = size(grid, 1)
    n_h = size(grid, 2)
    N_z = length(z_nodes)
    N_χ = length(χ_nodes)

    dz_fit = min(d_z, N_z - 1)
    dχ_fit = min(d_χ, N_χ - 1)

    K = PDECoefficients([Dict{NTuple{6,Int}, ComplexF64}() for _ in 1:n_eqs],
                         fill(dz_fit, n_eqs), fill(dχ_fit, n_eqs))

    # ── Chebyshev interpolation in z, then convert to z-monomials ──
    # Step A: build the Chebyshev evaluation matrix at the z-nodes
    # T_cheb[i, n+1] = T_n(z_nodes[i])
    T_cheb = zeros(N_z, N_z)
    for i in 1:N_z
        T_cheb[i, 1] = 1.0
        if N_z > 1; T_cheb[i, 2] = z_nodes[i]; end
        for n in 2:(N_z-1)
            T_cheb[i, n+1] = 2 * z_nodes[i] * T_cheb[i, n] - T_cheb[i, n-1]
        end
    end

    # Step B: Chebyshev → z-monomial conversion matrix (dz_fit+1 × dz_fit+1)
    # T_n(z) = Σ_{δ} M[δ+1, n+1] z^δ  →  z^δ coefficients of T_n
    M_cheb_to_mono = zeros(dz_fit + 1, dz_fit + 1)
    # Build recursively: T_0 = 1, T_1 = z, T_{n+1} = 2z T_n - T_{n-1}
    # as polynomial coefficient vectors
    if dz_fit >= 0
        p_prev = zeros(dz_fit + 1); p_prev[1] = 1.0  # T_0 = 1
        M_cheb_to_mono[:, 1] = p_prev
    end
    if dz_fit >= 1
        p_curr = zeros(dz_fit + 1); p_curr[2] = 1.0  # T_1 = z
        M_cheb_to_mono[:, 2] = p_curr
    end
    for n in 2:dz_fit
        p_next = zeros(dz_fit + 1)
        # T_{n+1} = 2z T_n - T_{n-1}: shift p_curr up by 1 (multiply by z), scale by 2, subtract p_prev
        for δ in 1:dz_fit
            p_next[δ + 1] = 2.0 * p_curr[δ]
        end
        p_next .-= p_prev
        M_cheb_to_mono[:, n + 1] = p_next
        p_prev = p_curr
        p_curr = p_next
    end

    # χ-direction: simple Vandermonde (χ range is [-1,1], moderate degree)
    V_χ = zeros(N_χ, dχ_fit + 1)
    for j in 1:N_χ, σ in 0:dχ_fit
        V_χ[j, σ + 1] = χ_nodes[j]^σ
    end

    n_fitted = 0
    for k in 1:n_eqs
        for d in 1:n_h
            vals = grid[k, d, :, :]  # N_z × N_χ
            maximum(abs, vals) < tol && continue

            j, α, β = h_map[d]

            # Step 1: fit χ at each z-node (Vandermonde, low degree)
            c_σ = zeros(ComplexF64, N_z, dχ_fit + 1)
            for i in 1:N_z
                c_σ[i, :] = V_χ \ vals[i, :]
            end

            # Step 2: for each σ, Chebyshev interpolation in z → monomial coefficients
            for σ in 0:dχ_fit
                # Get Chebyshev coefficients from values at Chebyshev nodes
                cheb_coeffs = T_cheb \ c_σ[:, σ + 1]

                # Truncate to dz_fit and convert to z-monomials
                cheb_trunc = cheb_coeffs[1:min(end, dz_fit + 1)]
                if length(cheb_trunc) < dz_fit + 1
                    resize!(cheb_trunc, dz_fit + 1)
                end
                mono_coeffs = M_cheb_to_mono * cheb_trunc

                for δ in 0:dz_fit
                    c = mono_coeffs[δ + 1]
                    abs(c) < tol && continue
                    key = (0, δ, σ, α, β, j)
                    K.equations[k][key] = get(K.equations[k], key, 0.0im) + c
                    n_fitted += 1
                end
            end
        end
    end

    verbose && println("  $n_fitted nonzero coefficients fitted (dz=$dz_fit, dχ=$dχ_fit)")
    return K
end

"""
Validate z-space fit at random test points.
"""
function _validate_fit_zspace(csc::CompiledSGBCorrection, bg, a::Float64,
                                ω::ComplexF64, K_z::PDECoefficients,
                                h_map::Vector{Tuple{Int,Int,Int}},
                                rp::Float64;
                                n_test::Int=5, verbose::Bool=false)
    max_err = 0.0
    max_val = 0.0

    for _ in 1:n_test
        z = -0.999 + 1.998 * rand()
        χ = -0.99 + 1.98 * rand()
        r = r_of_z(z, rp)

        hp = sgb_H_params(bg, r, χ)
        C_actual = extract_sgb_coefficients_complex(csc, r, χ, ω, a, hp)

        for k in 1:10
            for d in 1:length(h_map)
                j, α, β = h_map[d]
                fitted_val = 0.0im
                for ((γ, δ, σ, α2, β2, j2), c) in K_z.equations[k]
                    (α2 == α && β2 == β && j2 == j) || continue
                    fitted_val += c * z^δ * χ^σ
                end
                actual_val = C_actual[k, d]
                err = abs(fitted_val - actual_val)
                max_val = max(max_val, abs(actual_val))
                max_err = max(max_err, err)
            end
        end
    end

    rel_err = max_val > 0 ? max_err / max_val : max_err
    verbose && @printf("  Fit validation: max|err|=%.2e, max|val|=%.2e, rel=%.2e\n",
                        max_err, max_val, rel_err)
    return rel_err
end
