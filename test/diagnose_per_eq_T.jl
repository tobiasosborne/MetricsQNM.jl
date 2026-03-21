#!/usr/bin/env julia
# Diagnostic: per-equation denominator distribution for sGB K coefficients.
# Determines which equations have large r-denominators (T) and which are mild.

using MetricsQNM
using Printf
using Symbolics

function diagnose_T(a::Float64)
    println("=" ^ 70)
    @printf("Per-equation denominator diagnostic: a=%.3f\n", a)
    println("=" ^ 70); flush(stdout)

    # ── Phase 1-2: Extract sGB coefficients as RatPolys ──
    println("\n[1] Loading H_i..."); flush(stdout)
    H_ratpolys = load_H_ratpolys(a; verbose=true)

    println("\n[2] Extracting c_{k,d,p}..."); flush(stdout)
    coeffs, h_map, n_eqs = MetricsQNM.extract_sgb_coefficients_symbolic(a; verbose=true)

    @variables r chi omega_re omega_im iu_sym
    var_list = Num[r, chi, omega_re, omega_im, iu_sym]
    ctx = MetricsQNM.SymToPolyCtx(var_list, a)

    # ── Phase 3 variant: accumulate but DON'T clear, inspect denominators ──
    println("\n[3] Accumulating RatPolys (no clearing)..."); flush(stdout)

    kd_pairs = sort(unique([(k, d) for (k, d, p) in keys(coeffs)]))

    # Track per-equation statistics
    eq_stats = Dict{Int, @NamedTuple{
        n_pairs::Int, max_p::Int, max_q::Int, max_s::Int, max_t::Int,
        total_terms::Int, pairs::Vector{Tuple{Int,Int,Int,Int,Int}}
    }}()

    for k in 1:n_eqs
        eq_stats[k] = (n_pairs=0, max_p=0, max_q=0, max_s=0, max_t=0,
                        total_terms=0, pairs=Tuple{Int,Int,Int,Int,Int}[])
    end

    for (idx, (k, d)) in enumerate(kd_pairs)
        active_p = [p for p in 1:24 if haskey(coeffs, (k, d, p))]

        # Multiply c_{k,d,p} × H_p and accumulate
        terms = MetricsQNM.RatPoly[]
        for p in active_p
            product = MetricsQNM.mul_ratpoly(coeffs[(k, d, p)], H_ratpolys[p])
            push!(terms, product)
        end

        if length(terms) == 1
            accumulated = terms[1]
        else
            accumulated = MetricsQNM.add_ratpolys(terms, ctx)
        end
        MetricsQNM.cleanup!(accumulated.num; tol=1e-15, relative=true)

        # Reduce (but don't clear)
        reduced = MetricsQNM.reduce_ratpoly(accumulated, ctx)
        dp, dq, ds, dt = reduced.den.p, reduced.den.q, reduced.den.s, reduced.den.t
        n_terms = length(reduced.num.terms)

        # Update per-equation stats
        st = eq_stats[k]
        push!(st.pairs, (d, dp, dq, ds, dt))
        eq_stats[k] = (
            n_pairs = st.n_pairs + 1,
            max_p = max(st.max_p, dp),
            max_q = max(st.max_q, dq),
            max_s = max(st.max_s, ds),
            max_t = max(st.max_t, dt),
            total_terms = st.total_terms + n_terms,
            pairs = st.pairs
        )

        if idx % 20 == 0 || idx == length(kd_pairs)
            @printf("  [%d/%d] (k=%d,d=%d): den=Σ^%d Δ^%d s2^%d r^%d, %d terms\n",
                    idx, length(kd_pairs), k, d, dp, dq, ds, dt, n_terms)
            flush(stdout)
        end
    end

    # ── Summary ──
    println("\n" * "=" ^ 70)
    println("PER-EQUATION DENOMINATOR SUMMARY")
    println("=" ^ 70)
    @printf("%-4s  %-6s  %-6s  %-6s  %-6s  %-6s  %-8s\n",
            "Eq", "pairs", "max_P", "max_Q", "max_S", "max_T", "terms")
    println("-" ^ 50)

    total_high_T = 0
    for k in 1:n_eqs
        st = eq_stats[k]
        marker = st.max_t > 20 ? " ← HIGH T" : ""
        @printf("%-4d  %-6d  %-6d  %-6d  %-6d  %-6d  %-8d%s\n",
                k, st.n_pairs, st.max_p, st.max_q, st.max_s, st.max_t,
                st.total_terms, marker)
        st.max_t > 20 && (total_high_T += 1)
    end

    println("-" ^ 50)
    @printf("Equations with T > 20: %d / %d\n", total_high_T, n_eqs)

    # Also compute what GR d_max would be
    println("\nGR comparison:")
    K0_gr, K1_gr, K2_gr = extract_G_bespoke(a; verbose=false)
    d_max_gr = 0
    for K in (K0_gr, K1_gr, K2_gr), k in 1:10
        for ((γ, δ, σ, α, β, j), _) in K.equations[k]
            d_max_gr = max(d_max_gr, δ)
        end
    end
    @printf("  GR d_max (P=3,Q=1,S=1): %d\n", d_max_gr)

    # Estimate unified d_max per equation
    println("\nEstimated unified d_max per equation (GR P=3,Q=1 + sGB max clearing):")
    for k in 1:n_eqs
        st = eq_stats[k]
        # After clearing: r-degree ≈ original_num_degree + 2*max_p + 2*max_q + max_t
        # The original numerator has some r-degree, then clearing adds powers
        est_d_max = 2 * st.max_p + 2 * st.max_q + st.max_t + 10  # rough estimate
        @printf("  Eq %d: est d_max_k ≈ %d (P=%d, Q=%d, T=%d)\n",
                k, est_d_max, st.max_p, st.max_q, st.max_t)
    end

    println("\n" * "=" ^ 70); flush(stdout)
    return eq_stats
end

if abspath(PROGRAM_FILE) == @__FILE__
    a = parse(Float64, get(ARGS, 1, "0.3"))
    println("Julia threads: $(Threads.nthreads())"); flush(stdout)
    diagnose_T(a)
end
