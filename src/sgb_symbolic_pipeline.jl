# Exact K^(η=1) coefficient extraction for sGB gravity via SparsePoly CAS.
#
# Extends the GR pipeline (symbolic_pipeline.jl → extract_G_bespoke) to
# extract correction coefficients for the D̃⁽¹⁾ matrix.
#
# The sGB correction equations are LINEAR in both h-derivatives and H-parameters:
#   eq_k = Σ_d Σ_p c_{k,d,p}(r,χ,ω,a) × H_param_p × h_deriv_d
#
# Strategy: double probing (h_d=1, H_p=1) to extract c_{k,d,p} as RatPoly,
# multiply by H_p(r,χ) as RatPoly, sum over p → exact K^(η=1) coefficients.

using Symbolics
using Printf

export build_sgb_system_bespoke

# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2: Extract c_{k,d,p} via double probing
# ═══════════════════════════════════════════════════════════════════════════════

"""
    extract_sgb_coefficients_symbolic(a; verbose=false)
    → (coeffs, h_map, n_eqs)

Probe the sGB correction equations to extract c_{k,d,p} as RatPolys.

Returns:
  - coeffs: Dict{Tuple{Int,Int,Int}, RatPoly} mapping (k,d,p) → c_{k,d,p}
  - h_map: Vector of (j, α_r, β_χ) for each h-derivative index d
  - n_eqs: number of equations (10)
"""
function extract_sgb_coefficients_symbolic(a::Float64; verbose::Bool=false)
    verbose && (println("Phase 2: Extracting c_{k,d,p} via double probing..."); flush(stdout))
    t0 = time()

    # Compute sGB correction equations (symbolic, with abstract H-params)
    verbose && (println("  Computing sGB correction equations..."); flush(stdout))
    eqs, coords, params, hfuncs, omega_vars, Hp =
        compute_sgb_correction_equations(2; verbose=false)
    verbose && (@printf("  Equations computed: %.1fs\n", time() - t0); flush(stdout))

    r, chi = coords[2], coords[3]
    a_s = params[1]
    omega_re, omega_im, iu_sym = omega_vars

    # Discover h-derivative terms (same as extract_G_bespoke)
    all_vars = Set{Any}()
    for eq in eqs, v in Symbolics.get_variables(eq)
        any(occursin("h$j", string(v)) for j in 1:6) && push!(all_vars, v)
    end
    h_terms = sort(collect(all_vars), by=string)
    h_map = _parse_h_terms(string.(h_terms))
    sub_zero_h = Dict{Any,Any}(t => Num(0) for t in h_terms)
    n_h = length(h_terms)
    n_eqs = length(eqs)

    # H-parameter substitution dicts
    sub_zero_H = Dict{Any,Any}(Symbolics.unwrap(hp) => Num(0) for hp in _H_PARAMS)

    # SparsePoly context
    var_list = Num[r, chi, omega_re, omega_im, iu_sym]
    ctx = SymToPolyCtx(var_list, a)

    verbose && (println("  $n_h h-derivatives × $n_eqs equations × 24 H-params"); flush(stdout))
    verbose && (println("  Stage A: h-probing..."); flush(stdout))

    coeffs = Dict{Tuple{Int,Int,Int}, RatPoly}()
    n_nonzero_kd = 0
    n_nonzero_kdp = 0

    for k in 1:n_eqs
        for d in 1:n_h
            # Stage A: substitute h_d = 1, all other h = 0
            sub_h = copy(sub_zero_h)
            sub_h[h_terms[d]] = Num(1)
            intermediate = Symbolics.substitute(eqs[k], sub_h)
            isequal(intermediate, Num(0)) && continue
            n_nonzero_kd += 1

            # Stage B: for each H-parameter, substitute H_p = 1, others = 0
            for p in 1:24
                sub_H = copy(sub_zero_H)
                sub_H[Symbolics.unwrap(_H_PARAMS[p])] = Num(1)
                c_expr = Symbolics.substitute(intermediate, sub_H)
                isequal(c_expr, Num(0)) && continue

                # Substitute a → numerical
                c_expr_a = Symbolics.substitute(c_expr, Dict(a_s => a))

                # Convert to RatPoly
                rp = symexpr_to_poly(c_expr_a, ctx)
                coeffs[(k, d, p)] = rp
                n_nonzero_kdp += 1
            end

            # Release Symbolics intermediate
            intermediate = nothing
        end

        verbose && (print("  eq $k: $(n_nonzero_kdp) non-zero (k,d,p) so far\r"); flush(stdout))
    end

    verbose && (println("\n  Non-zero: $n_nonzero_kd (k,d) pairs, $n_nonzero_kdp (k,d,p) triples"))
    verbose && (@printf("  Phase 2 total: %.1fs\n", time() - t0); flush(stdout))

    return coeffs, h_map, n_eqs
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 3: Multiply c_{k,d,p} × H_p, accumulate → PDECoefficients
# ═══════════════════════════════════════════════════════════════════════════════

"""
    combine_sgb_K(coeffs, H_ratpolys, h_map, n_eqs, ctx; P, Q, S, verbose)
    → (K0_r, K1_r, K2_r)

Multiply c_{k,d,p} × H_p, accumulate over p, ω-decompose, store in PDECoefficients.
Memory-safe: processes one (k,d) at a time, releases intermediates.
"""
function combine_sgb_K(coeffs::Dict{Tuple{Int,Int,Int}, RatPoly},
                       H_ratpolys::Vector{RatPoly},
                       h_map::Vector{Tuple{Int,Int,Int}},
                       n_eqs::Int,
                       ctx::SymToPolyCtx;
                       P::Int=3, Q::Int=1, S::Int=1,
                       verbose::Bool=false)
    verbose && (println("Phase 3: Multiply c_{k,d,p} × H_p → PDECoefficients..."); flush(stdout))
    t0 = time()

    Ks = [PDECoefficients([Dict{NTuple{6,Int}, ComplexF64}() for _ in 1:n_eqs],
                          fill(200, n_eqs), fill(50, n_eqs)) for _ in 1:3]

    # Group by (k, d)
    kd_pairs = sort(unique([(k, d) for (k, d, p) in keys(coeffs)]))
    n_kd = length(kd_pairs)
    verbose && (println("  Processing $n_kd non-zero (k,d) pairs"); flush(stdout))

    for (idx_kd, (k, d)) in enumerate(kd_pairs)
        # Collect all p that contribute to this (k,d)
        active_p = [p for p in 1:24 if haskey(coeffs, (k, d, p))]

        # Accumulate: K = Σ_p c_{k,d,p} × H_p (in RatPoly land)
        terms = RatPoly[]
        for p in active_p
            product = mul_ratpoly(coeffs[(k, d, p)], H_ratpolys[p])
            push!(terms, product)
        end

        # Sum all terms via LCD clearing
        if length(terms) == 1
            accumulated = terms[1]
        else
            accumulated = add_ratpolys(terms, ctx)
        end
        cleanup!(accumulated.num; tol=1e-15, relative=true)

        # Clear all denominators
        poly, actual = clear_denominators(accumulated, P, Q, S, ctx)

        # ω-decompose and store in PDECoefficients
        j_d, α_d, β_d = h_map[d]
        n_stored = 0
        for (exps, coeff_val) in poly.terms
            abs(coeff_val) < 1e-15 && continue
            δ, σ, p_ω, q_ω, s_iu = exps
            G0, G1, G2 = _omega_monomial_to_G(p_ω, q_ω, s_iu, coeff_val)

            for (γ, Gval) in enumerate((G0, G1, G2))
                abs(Gval) < 1e-15 && continue
                key = (γ - 1, δ, σ, α_d, β_d, j_d)
                Ks[γ].equations[k][key] =
                    get(Ks[γ].equations[k], key, 0.0im) + Gval
                n_stored += 1
            end
        end

        if verbose && (idx_kd % 20 == 0 || idx_kd == n_kd)
            @printf("  [%d/%d] (k=%d,d=%d): %d p-terms, %d poly terms, clearing (P=%d,Q=%d,S=%d,T=%d)\n",
                    idx_kd, n_kd, k, d, length(active_p), length(poly.terms),
                    actual.p, actual.q, actual.s, actual.t)
            flush(stdout)
        end

        # Release intermediates
        terms = nothing
        accumulated = nothing
        poly = nothing
        (idx_kd % 50 == 0) && GC.gc(false)
    end

    # Count total K entries
    for (γ, name) in enumerate(["K0", "K1", "K2"])
        total = sum(length(Ks[γ].equations[k]) for k in 1:n_eqs)
        verbose && println("  $name: $total terms (r-space)")
    end

    verbose && (@printf("  Phase 3 total: %.1fs\n", time() - t0); flush(stdout))
    return Ks[1], Ks[2], Ks[3]
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 4: Wire into pipeline → METRICSSystem
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_sgb_system_bespoke(a, N, m; epsilon, verbose) → (METRICSSystem, d_max)

Build D̃⁽¹⁾ correction matrices via per-a-order exact SparsePoly extraction.

The H_i background is a power series in a²: H_i = Σ_k a^{2k} H_i^{(2k)}(r,χ).
Each a-order has moderate r-denominators (den.t ≈ 2k+5). Processing per-order
avoids the LCD d_max inflation (80→~15) that occurs when summing all orders.

Each a-order is processed through: c_{k,d,p} × H_p^{(2k)} → K_r → K_z → D̃.
The resulting D̃ matrices are summed with a^{2k} weights.

Returns `(sys, d_max)` where d_max is the shared d_max used across all orders.
The caller must build the GR system with the same d_max for perturbation theory.
"""
function build_sgb_system_bespoke(a::Float64, N::Int, m::Int;
                                   epsilon::Float64=1e-14,
                                   verbose::Bool=false)
    verbose && (println("Building sGB D̃⁽¹⁾ via per-a-order exact SparsePoly..."); flush(stdout))
    t0_total = time()

    rp = r_plus(a)

    # Phase 1: Load H_i as per-a-order RatPoly
    H_per_order, a_powers = load_H_ratpolys_per_order(; verbose)
    n_orders_total = length(H_per_order)

    # Truncate a-series: skip orders where a^{a_pow} < epsilon
    n_a_max = n_orders_total
    for (oi, a_pow) in enumerate(a_powers)
        if a_pow > 0 && abs(a)^a_pow < epsilon
            n_a_max = oi - 1
            break
        end
    end
    if verbose
        trunc_pow = n_a_max < n_orders_total ? a_powers[n_a_max + 1] : a_powers[end]
        trunc_val = n_a_max < n_orders_total ? abs(a)^trunc_pow : 0.0
        @printf("  Retaining %d/%d a-orders (first dropped: a^%d ≈ %.1e)\n",
                n_a_max, n_orders_total, trunc_pow, trunc_val)
        flush(stdout)
    end

    # Verification: check per-order splitting matches original
    if verbose
        print("  Verifying per-order splitting... "); flush(stdout)
        err = verify_H_ratpolys_per_order(H_per_order, a_powers, a; verbose=false)
        @printf("max rel err = %.2e %s\n", err, err < 1e-10 ? "✓" : "✗ WARNING")
        flush(stdout)
    end

    # Phase 2: Extract c_{k,d,p} via double probing (numerical a, same as before)
    coeffs, h_map, n_eqs = extract_sgb_coefficients_symbolic(a; verbose)

    # Build context for Phase 3
    @variables r chi omega_re omega_im iu_sym
    var_list = Num[r, chi, omega_re, omega_im, iu_sym]
    ctx = SymToPolyCtx(var_list, a)

    # Phase 3: Process each a-order through combine_sgb_K
    verbose && (println("Phase 3: Processing $n_a_max a-orders..."); flush(stdout))
    K_r_per_order = Vector{NTuple{3, PDECoefficients}}(undef, n_a_max)

    for ao = 1:n_a_max
        t_ao = time()
        K0_ao, K1_ao, K2_ao = combine_sgb_K(coeffs, H_per_order[ao], h_map,
                                              n_eqs, ctx; verbose=false)
        K_r_per_order[ao] = (K0_ao, K1_ao, K2_ao)

        if verbose
            d_ao = 0
            for K in (K0_ao, K1_ao, K2_ao), k in 1:n_eqs
                for ((γ, δ, σ, α, β, j), _) in K.equations[k]
                    d_ao = max(d_ao, δ)
                end
            end
            n_terms = sum(length(K.equations[k]) for K in (K0_ao, K1_ao, K2_ao) for k in 1:n_eqs)
            @printf("  a^%d: d_max=%d, %d terms, %.1fs\n",
                    a_powers[ao], d_ao, n_terms, time() - t_ao)
            flush(stdout)
        end
    end

    # Determine shared d_max and s_max across all retained a-orders
    d_max = 0
    s_max = 0
    for ao = 1:n_a_max
        for K in K_r_per_order[ao], k in 1:n_eqs
            for ((γ, δ, σ, α, β, j), _) in K.equations[k]
                d_max = max(d_max, δ)
                s_max = max(s_max, σ)
            end
        end
    end
    verbose && (@printf("  Shared d_max=%d, s_max=%d\n", d_max, s_max); flush(stdout))

    # Phase 4-5: r→z transform + assembly (shared basis for all orders)
    verbose && (println("Phase 4-5: r→z + assembly (shared d_max=$d_max, N=$N)..."); flush(stdout))
    basis = spectral_basis(N, m; max_delta=d_max, max_sigma=max(s_max, 25))
    bs = (N + 1)^2

    D0_total = zeros(ComplexF64, 10 * bs, 6 * bs)
    D1_total = zeros(ComplexF64, 10 * bs, 6 * bs)
    D2_total = zeros(ComplexF64, 10 * bs, 6 * bs)

    for ao = 1:n_a_max
        K0_ao_r, K1_ao_r, K2_ao_r = K_r_per_order[ao]

        K0_ao_z = _r_to_z(K0_ao_r, rp, d_max)
        K1_ao_z = _r_to_z(K1_ao_r, rp, d_max)
        K2_ao_z = _r_to_z(K2_ao_r, rp, d_max)

        D0_ao = assemble_system(K0_ao_z, basis, a).D0
        D1_ao = assemble_system(K1_ao_z, basis, a).D0
        D2_ao = assemble_system(K2_ao_z, basis, a).D0

        weight = a_powers[ao] == 0 ? 1.0 : a^a_powers[ao]
        D0_total .+= weight .* D0_ao
        D1_total .+= weight .* D1_ao
        D2_total .+= weight .* D2_ao

        if verbose
            @printf("  a^%d: weight=%.2e, ‖D₀‖=%.2e, ‖D₁‖=%.2e, ‖D₂‖=%.2e\n",
                    a_powers[ao], weight, norm(D0_ao), norm(D1_ao), norm(D2_ao))
            flush(stdout)
        end
    end

    sys = METRICSSystem(D0_total, D1_total, D2_total, N, m, a)

    verbose && (@printf("Total sGB D̃⁽¹⁾ build: %.1fs\n", time() - t0_total); flush(stdout))
    return sys, d_max
end
