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
    build_sgb_system_bespoke(a, N, m; verbose=false) → METRICSSystem

Build D̃⁽¹⁾ correction matrices via exact symbolic K^(η=1) extraction.
Uses the SparsePoly CAS to extract exact polynomial coefficients from
the sGB correction equations, replacing the failed numerical Galerkin approach.

Returns a METRICSSystem with D₀⁽¹⁾, D₁⁽¹⁾, D₂⁽¹⁾ matrices.
"""
function build_sgb_system_bespoke(a::Float64, N::Int, m::Int;
                                   verbose::Bool=false)
    verbose && (println("Building sGB D̃⁽¹⁾ via exact SparsePoly extraction..."); flush(stdout))
    t0_total = time()

    rp = r_plus(a)

    # Phase 1: Load H_i as RatPoly
    H_ratpolys = load_H_ratpolys(a; verbose)

    # Phase 2: Extract c_{k,d,p} via double probing
    coeffs, h_map, n_eqs = extract_sgb_coefficients_symbolic(a; verbose)

    # Build context for Phase 3 (same as used in Phase 2)
    @variables r chi omega_re omega_im iu_sym
    var_list = Num[r, chi, omega_re, omega_im, iu_sym]
    ctx = SymToPolyCtx(var_list, a)

    # Phase 3: Multiply and accumulate → K0, K1, K2 in r-space
    K0_r, K1_r, K2_r = combine_sgb_K(coeffs, H_ratpolys, h_map, n_eqs, ctx;
                                       verbose)

    # Phase 4: r → z transform
    d_max_0 = maximum(δ for k in 1:n_eqs for ((γ, δ, σ, α, β, j), _) in K0_r.equations[k]; init=0)
    d_max_1 = maximum(δ for k in 1:n_eqs for ((γ, δ, σ, α, β, j), _) in K1_r.equations[k]; init=0)
    d_max_2 = maximum(δ for k in 1:n_eqs for ((γ, δ, σ, α, β, j), _) in K2_r.equations[k]; init=0)
    d_max = max(d_max_0, d_max_1, d_max_2)
    verbose && (println("  Max r-degree: $d_max"); flush(stdout))

    verbose && (println("Transforming r → z (d_max=$d_max)..."); flush(stdout))
    K0_z = _r_to_z(K0_r, rp, d_max)
    K1_z = _r_to_z(K1_r, rp, d_max)
    K2_z = _r_to_z(K2_r, rp, d_max)

    # Phase 5: Assemble D̃ matrices
    # Compute max z-degree and chi-degree from z-space K coefficients
    s_max = 0
    for K in (K0_z, K1_z, K2_z), k in 1:n_eqs
        for ((γ, δ, σ, α, β, j), _) in K.equations[k]
            σ > s_max && (s_max = σ)
        end
    end
    verbose && (println("Assembling D̃⁽¹⁾ matrices (N=$N, max_delta=$d_max, max_sigma=$s_max)..."); flush(stdout))
    basis = spectral_basis(N, m; max_delta=d_max, max_sigma=max(s_max, 25))
    D0 = assemble_system(K0_z, basis, a).D0
    D1 = assemble_system(K1_z, basis, a).D0
    D2 = assemble_system(K2_z, basis, a).D0

    sys = METRICSSystem(D0, D1, D2, N, m, a)
    # NOTE: Do NOT normalize here. The caller must apply the GR normalization
    # factors via normalize_system!(sys, gr_factors) to ensure D̃⁽¹⁾ uses the
    # same per-equation scaling as D̃⁽⁰⁾ (required for perturbation theory).

    verbose && (@printf("Total sGB D̃⁽¹⁾ build: %.1fs\n", time() - t0_total); flush(stdout))
    return sys
end
