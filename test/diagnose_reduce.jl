#!/usr/bin/env julia
"""
Diagnostic: test whether reduce_ratpoly is missing Σ/Δ cancellations.
"""

using Printf

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MetricsQNM
using MetricsQNM: compute_sgb_correction_equations, _H_PARAMS, _parse_h_terms,
                  symexpr_to_poly, reduce_ratpoly, SymToPolyCtx, RatPoly,
                  SparsePoly, DenomSig, _poly_divmod, cleanup!, N_VARS,
                  var_poly
import Symbolics
import Symbolics: Num

function test_extra_divisibility(rp_reduced, ctx)
    num = rp_reduced.num
    isempty(num.terms) && return Dict("Σ"=>0, "Δ"=>0, "(1-χ²)"=>0)

    results = Dict{String, Int}()

    for (name, fac) in [("Σ", ctx.Σ_poly), ("Δ", ctx.Δ_poly), ("(1-χ²)", ctx.s2_poly)]
        trial = num
        extra = 0
        for _ in 1:5
            q, rem = _poly_divmod(trial, fac)
            scale = isempty(trial.terms) ? 1.0 : maximum(abs, values(trial.terms))
            cleanup!(rem; tol=1e-10 * scale)
            if q !== nothing && iszero(rem)
                extra += 1
                cleanup!(q)
                trial = q
            else
                break
            end
        end
        results[name] = extra
    end
    return results
end

function main()
    println("=" ^ 70)
    println("DIAGNOSTIC: reduce_ratpoly cancellation quality")
    println("=" ^ 70)
    flush(stdout)

    println("\n>>> Building sGB correction equations...")
    flush(stdout)

    eqs_corr, coords, params, hfuncs, freq_vars, H_params_struct =
        compute_sgb_correction_equations(2; verbose=true)
    r, chi = coords[2], coords[3]
    a_s = params[1]
    omega_re, omega_im, iu_sym = freq_vars

    all_vars = Set{Any}()
    for eq in eqs_corr, v in Symbolics.get_variables(eq)
        any(occursin("h$j", string(v)) for j in 1:6) && push!(all_vars, v)
    end
    h_terms = sort(collect(all_vars), by=string)
    h_map = _parse_h_terms(string.(h_terms))
    sub_zero_h = Dict{Any,Any}(t => Num(0) for t in h_terms)
    n_h = length(h_terms)

    H_param_syms = _H_PARAMS
    n_H = length(H_param_syms)
    sub_zero_H = Dict{Any,Any}(Symbolics.unwrap(H_param_syms[p]) => Num(0) for p in 1:n_H)

    var_list = Num[r, chi, omega_re, omega_im, iu_sym, a_s]
    ctx = SymToPolyCtx(var_list)

    println("  Ready. n_h=$n_h, n_H=$n_H\n")
    flush(stdout)

    # ─────────────────────────────────────────────────────────────────
    # Probe: for each equation, sample 5 nonzero triples, test extra divisibility
    # ─────────────────────────────────────────────────────────────────
    println(">>> Probing (k,d,p) triples for missed cancellations\n")
    flush(stdout)

    total_probed = 0
    total_missed = 0
    missed_details = Tuple{Int,Int,Int,DenomSig,Dict{String,Int}}[]

    for k in 1:10
        n_probed_k = 0
        for d in 1:n_h
            n_probed_k >= 5 && break
            for p in 1:n_H
                n_probed_k >= 5 && break

                sub_h = copy(sub_zero_h)
                sub_h[h_terms[d]] = Num(1)
                intermediate = Symbolics.substitute(eqs_corr[k], sub_h)
                isequal(intermediate, Num(0)) && continue

                sub_H = copy(sub_zero_H)
                sub_H[Symbolics.unwrap(H_param_syms[p])] = Num(1)
                c_expr = Symbolics.substitute(intermediate, sub_H)
                isequal(c_expr, Num(0)) && continue

                rp = symexpr_to_poly(c_expr, ctx)
                rp_reduced = reduce_ratpoly(rp, ctx)
                den = rp_reduced.den
                n_terms = length(rp_reduced.num.terms)

                extra = test_extra_divisibility(rp_reduced, ctx)
                total_probed += 1
                n_probed_k += 1

                missed = extra["Σ"] + extra["Δ"] + extra["(1-χ²)"]
                total_missed += missed

                marker = missed > 0 ? " *** MISSED ***" : ""
                @printf("  (k=%d, d=%2d, p=%2d): reduced Σ^%d Δ^%d s2^%d r^%d (%3d terms)",
                        k, d, p, den.p, den.q, den.s, den.t, n_terms)
                if missed > 0
                    @printf("  extra: Σ^%d Δ^%d s2^%d", extra["Σ"], extra["Δ"], extra["(1-χ²)"])
                    push!(missed_details, (k, d, p, den, extra))
                end
                println(marker)
                flush(stdout)
            end
        end
    end

    @printf("\n>>> Summary: probed %d triples, %d had missed cancellations\n\n",
            total_probed, length(missed_details))

    if !isempty(missed_details)
        println("  Missed cancellation details:")
        for (k, d, p, den, extra) in missed_details
            could_reduce_p = den.p - extra["Σ"]
            could_reduce_q = den.q - extra["Δ"]
            could_reduce_s = den.s - extra["(1-χ²)"]
            @printf("    (k=%d,d=%d,p=%d): current Σ^%d Δ^%d s2^%d → could be Σ^%d Δ^%d s2^%d (saves %d r-deg)\n",
                    k, d, p, den.p, den.q, den.s,
                    could_reduce_p, could_reduce_q, could_reduce_s,
                    2*extra["Σ"] + 2*extra["Δ"])
        end
    end
    flush(stdout)

    # ─────────────────────────────────────────────────────────────────
    # Full scan: equation 1, ALL nonzero (d,p) triples
    # ─────────────────────────────────────────────────────────────────
    println("\n>>> Full scan: eq 1, all nonzero (d,p) triples\n")
    flush(stdout)

    n_scanned = 0
    n_with_missed = 0
    max_missed_sigma = 0
    max_missed_delta = 0
    max_missed_s2 = 0

    for d in 1:n_h
        for p in 1:n_H
            sub_h = copy(sub_zero_h)
            sub_h[h_terms[d]] = Num(1)
            intermediate = Symbolics.substitute(eqs_corr[1], sub_h)
            isequal(intermediate, Num(0)) && continue

            sub_H = copy(sub_zero_H)
            sub_H[Symbolics.unwrap(H_param_syms[p])] = Num(1)
            c_expr = Symbolics.substitute(intermediate, sub_H)
            isequal(c_expr, Num(0)) && continue

            rp = symexpr_to_poly(c_expr, ctx)
            rp_reduced = reduce_ratpoly(rp, ctx)

            extra = test_extra_divisibility(rp_reduced, ctx)
            n_scanned += 1

            if extra["Σ"] > 0 || extra["Δ"] > 0 || extra["(1-χ²)"] > 0
                n_with_missed += 1
                max_missed_sigma = max(max_missed_sigma, extra["Σ"])
                max_missed_delta = max(max_missed_delta, extra["Δ"])
                max_missed_s2 = max(max_missed_s2, extra["(1-χ²)"])
                @printf("  (1,%2d,%2d): Σ^%d Δ^%d s2^%d r^%d → extra Σ^%d Δ^%d s2^%d\n",
                        d, p,
                        rp_reduced.den.p, rp_reduced.den.q, rp_reduced.den.s, rp_reduced.den.t,
                        extra["Σ"], extra["Δ"], extra["(1-χ²)"])
                flush(stdout)
            end
        end
    end

    @printf("\n  Eq 1: scanned %d triples, %d had missed cancellations\n",
            n_scanned, n_with_missed)
    @printf("  Max missed: Σ^%d Δ^%d (1-χ²)^%d\n",
            max_missed_sigma, max_missed_delta, max_missed_s2)

    if n_with_missed > 0
        # What would the per-eq clearing be if we fixed reduce_ratpoly?
        current_max_p = 0
        current_max_q = 0
        improved_max_p = 0
        improved_max_q = 0
        for d in 1:n_h, p in 1:n_H
            sub_h = copy(sub_zero_h)
            sub_h[h_terms[d]] = Num(1)
            intermediate = Symbolics.substitute(eqs_corr[1], sub_h)
            isequal(intermediate, Num(0)) && continue
            sub_H = copy(sub_zero_H)
            sub_H[Symbolics.unwrap(H_param_syms[p])] = Num(1)
            c_expr = Symbolics.substitute(intermediate, sub_H)
            isequal(c_expr, Num(0)) && continue

            rp = symexpr_to_poly(c_expr, ctx)
            rp_reduced = reduce_ratpoly(rp, ctx)
            extra = test_extra_divisibility(rp_reduced, ctx)

            current_max_p = max(current_max_p, rp_reduced.den.p)
            current_max_q = max(current_max_q, rp_reduced.den.q)
            improved_max_p = max(improved_max_p, rp_reduced.den.p - extra["Σ"])
            improved_max_q = max(improved_max_q, rp_reduced.den.q - extra["Δ"])
        end
        @printf("\n  Eq 1 clearing: current Σ^%d Δ^%d → improved Σ^%d Δ^%d\n",
                current_max_p, current_max_q, improved_max_p, improved_max_q)
        saved_rdeg = 2*(current_max_p - improved_max_p) + 2*(current_max_q - improved_max_q)
        @printf("  Saved r-degrees: %d\n", saved_rdeg)
    end

    flush(stdout)
    println("\n" * "=" ^ 70)
    println("DONE")
    println("=" ^ 70)
end

main()
