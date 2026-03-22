#!/usr/bin/env julia
#
# Diagnose the 1.5% verification error in per-a-order H_i splitting.
#
# Strategy: isolate the error to one of:
#   Stage 1: symexpr_to_poly with symbolic a vs numerical a (before split)
#   Stage 2: _split_ratpoly_by_var (split fidelity)
#   Stage 3: cleanup! dropping different terms
#   Stage 4: derivative computation
#
using Printf

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using MetricsQNM

import MetricsQNM: load_H_ratpolys, load_H_ratpolys_per_order, verify_H_ratpolys_per_order,
    _eval_ratpoly_rchi, _split_ratpoly_by_var, _load_nb_sections, _box_to_str,
    symexpr_to_poly, SymToPolyCtx, RatPoly, SparsePoly, DenomSig, cleanup!,
    _differentiate_ratpoly_r, _differentiate_ratpoly_chi, N_VARS, ExpVec
using Symbolics

const a_test = 0.3
const test_points = [(5.0, 0.5), (3.0, 0.8), (10.0, 0.1), (2.5, 0.99), (50.0, 0.0)]

# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: evaluate RatPoly with 3 variables (r, χ, a)
# ═══════════════════════════════════════════════════════════════════════════════
function eval_ratpoly_rchi_a(rp::RatPoly, r_val, chi_val, a_val)
    @assert rp.den.p == 0 && rp.den.q == 0 && rp.den.s == 0
    num_val = 0.0
    for (e, c) in rp.num.terms
        num_val += c * r_val^e[1] * chi_val^e[2] * a_val^e[3]
    end
    den_val = r_val^rp.den.t
    return num_val / den_val
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1: Compare symexpr_to_poly outputs (before split, before derivatives)
# ═══════════════════════════════════════════════════════════════════════════════
function diagnose_stage1()
    println("=" ^ 80)
    println("STAGE 1: symexpr_to_poly — symbolic-a vs numerical-a (H_i base functions)")
    println("=" ^ 80); flush(stdout)

    sections = _load_nb_sections(; verbose=false)

    # --- Path A: numerical a ---
    @variables _r_A _χ_A _dA3 _dA4 _dA5
    var_list_A = Num[_r_A, _χ_A, _dA3, _dA4, _dA5]
    ctx_A = SymToPolyCtx(var_list_A, a_test)

    # --- Path B: symbolic a ---
    @variables _r_B _χ_B _a_B _dB4 _dB5
    var_list_B = Num[_r_B, _χ_B, _a_B, _dB4, _dB5]
    ctx_B = SymToPolyCtx(var_list_B, 1.0)

    for i in 1:4
        @printf("\n--- H%d ---\n", i); flush(stdout)
        expr_str = _box_to_str(sections["H$(i)"])
        body = Meta.parse(expr_str)

        # Path A: numerical a
        print("  Path A (a=$a_test numerical): "); flush(stdout)
        sym_A = Base.invokelatest(eval, :(let _r_ = $(Num(_r_A)), _χ_ = $(Num(_χ_A)),
                                              _a_ = $a_test, _M_ = 1, _α_ = 1
                                              Num($body)
                                          end))
        rp_A = symexpr_to_poly(sym_A, ctx_A)
        cleanup!(rp_A.num; tol=1e-15, relative=true)
        @printf("%d terms, den.t=%d\n", length(rp_A.num.terms), rp_A.den.t); flush(stdout)

        # Path B: symbolic a
        print("  Path B (a symbolic):          "); flush(stdout)
        sym_B = Base.invokelatest(eval, :(let _r_ = $(Num(_r_B)), _χ_ = $(Num(_χ_B)),
                                              _a_ = $(Num(_a_B)), _M_ = 1, _α_ = 1
                                              Num($body)
                                          end))
        rp_B = symexpr_to_poly(sym_B, ctx_B)
        # Do NOT cleanup yet — we want to compare raw first
        rp_B_raw_nterms = length(rp_B.num.terms)
        cleanup!(rp_B.num; tol=1e-15, relative=true)
        @printf("%d terms (raw %d), den.t=%d\n",
                length(rp_B.num.terms), rp_B_raw_nterms, rp_B.den.t); flush(stdout)

        # Compare at test points
        for (r, χ) in test_points
            val_A = _eval_ratpoly_rchi(rp_A, r, χ)
            val_B = eval_ratpoly_rchi_a(rp_B, r, χ, a_test)
            err = abs(val_A - val_B)
            rel = val_A == 0 ? err : err / abs(val_A)
            flag = rel > 1e-10 ? " *** ERROR ***" : ""
            @printf("  (r=%.1f, χ=%.2f): A=%.10e  B=%.10e  rel=%.2e%s\n",
                    r, χ, val_A, val_B, rel, flag)
        end
        flush(stdout)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2: _split_ratpoly_by_var fidelity
# ═══════════════════════════════════════════════════════════════════════════════
function diagnose_stage2()
    println("\n" * "=" ^ 80)
    println("STAGE 2: _split_ratpoly_by_var — split-and-sum vs unsplit")
    println("=" ^ 80); flush(stdout)

    sections = _load_nb_sections(; verbose=false)

    @variables _r_S _χ_S _a_S _dS4 _dS5
    var_list = Num[_r_S, _χ_S, _a_S, _dS4, _dS5]
    ctx = SymToPolyCtx(var_list, 1.0)

    for i in 1:4
        @printf("\n--- H%d ---\n", i); flush(stdout)
        expr_str = _box_to_str(sections["H$(i)"])
        body = Meta.parse(expr_str)

        sym_expr = Base.invokelatest(eval, :(let _r_ = $(Num(_r_S)), _χ_ = $(Num(_χ_S)),
                                                  _a_ = $(Num(_a_S)), _M_ = 1, _α_ = 1
                                                  Num($body)
                                              end))
        rp_full = symexpr_to_poly(sym_expr, ctx)
        cleanup!(rp_full.num; tol=1e-15, relative=true)

        # Split
        split = _split_ratpoly_by_var(rp_full, 3)
        a_orders = sort(collect(keys(split)))
        @printf("  %d a-orders: %s\n", length(a_orders),
                join(["a^$k" for k in a_orders[1:min(6,end)]], ", ") *
                (length(a_orders) > 6 ? "..." : "")); flush(stdout)

        # Compare: unsplit rp_full(r, χ, a) vs Σ a^k × split_k(r, χ)
        for (r, χ) in test_points
            val_full = eval_ratpoly_rchi_a(rp_full, r, χ, a_test)
            val_sum = 0.0
            for k in a_orders
                val_sum += a_test^k * _eval_ratpoly_rchi(split[k], r, χ)
            end
            err = abs(val_full - val_sum)
            rel = val_full == 0 ? err : err / abs(val_full)
            flag = rel > 1e-12 ? " *** SPLIT ERROR ***" : ""
            @printf("  (r=%.1f, χ=%.2f): full=%.10e  sum=%.10e  rel=%.2e%s\n",
                    r, χ, val_full, val_sum, rel, flag)
        end
        flush(stdout)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 3: Per-component verification (which of the 24 H-params has the error?)
# ═══════════════════════════════════════════════════════════════════════════════
function diagnose_stage3()
    println("\n" * "=" ^ 80)
    println("STAGE 3: Full per-order pipeline — which H-params have error?")
    println("=" ^ 80); flush(stdout)

    print("Loading per-order H_i... "); flush(stdout)
    H_per_order, a_powers = load_H_ratpolys_per_order(; verbose=false)
    println("done ($(length(a_powers)) orders)"); flush(stdout)

    print("Loading reference H_i (a=$a_test)... "); flush(stdout)
    H_ref = load_H_ratpolys(a_test; verbose=false)
    println("done"); flush(stdout)

    h_names = String[]
    for i in 1:4
        push!(h_names, "H$(i)")
        push!(h_names, "H$(i)_r")
        push!(h_names, "H$(i)_χ")
        push!(h_names, "H$(i)_rr")
        push!(h_names, "H$(i)_χχ")
        push!(h_names, "H$(i)_rχ")
    end

    r_test, χ_test = 5.0, 0.5
    println("\nTest point: (r=$r_test, χ=$χ_test)")
    println("-" ^ 80)
    @printf("%-10s  %16s  %16s  %12s\n", "H-param", "Reference", "Per-order sum", "Rel error")
    println("-" ^ 80)

    max_err = 0.0
    worst_j = 0
    for j in 1:24
        ref_val = _eval_ratpoly_rchi(H_ref[j], r_test, χ_test)
        sum_val = 0.0
        for (oi, a_pow) in enumerate(a_powers)
            sum_val += a_test^a_pow * _eval_ratpoly_rchi(H_per_order[oi][j], r_test, χ_test)
        end
        err = abs(sum_val - ref_val)
        rel = ref_val == 0 ? err : err / abs(ref_val)
        flag = rel > 1e-10 ? " ***" : ""
        @printf("%-10s  %16.10e  %16.10e  %12.2e%s\n",
                h_names[j], ref_val, sum_val, rel, flag)
        if rel > max_err
            max_err = rel
            worst_j = j
        end
    end
    println("-" ^ 80)
    @printf("Worst: %s with rel error %.2e\n", h_names[worst_j], max_err)
    flush(stdout)

    # For the worst component, show per-order contributions
    if worst_j > 0
        println("\nPer-order breakdown for $(h_names[worst_j]):")
        ref_val = _eval_ratpoly_rchi(H_ref[worst_j], r_test, χ_test)
        running_sum = 0.0
        for (oi, a_pow) in enumerate(a_powers)
            contrib = a_test^a_pow * _eval_ratpoly_rchi(H_per_order[oi][worst_j], r_test, χ_test)
            running_sum += contrib
            partial_rel = ref_val == 0 ? abs(running_sum) : abs(running_sum - ref_val) / abs(ref_val)
            @printf("  a^%2d: contrib=%+.6e  cumsum=%+.10e  partial_rel=%.2e  (den.t=%d, %d terms)\n",
                    a_pow, contrib, running_sum, partial_rel,
                    H_per_order[oi][worst_j].den.t,
                    length(H_per_order[oi][worst_j].num.terms))
        end
        @printf("  Reference: %.10e\n", ref_val)
        flush(stdout)
    end

    # Also check at multiple points for the worst component
    if worst_j > 0
        println("\n$(h_names[worst_j]) at multiple test points:")
        for (r, χ) in test_points
            ref_val = _eval_ratpoly_rchi(H_ref[worst_j], r, χ)
            sum_val = 0.0
            for (oi, a_pow) in enumerate(a_powers)
                sum_val += a_test^a_pow * _eval_ratpoly_rchi(H_per_order[oi][worst_j], r, χ)
            end
            rel = ref_val == 0 ? abs(sum_val) : abs(sum_val - ref_val) / abs(ref_val)
            @printf("  (r=%5.1f, χ=%.2f): ref=%+.10e  sum=%+.10e  rel=%.2e\n",
                    r, χ, ref_val, sum_val, rel)
        end
        flush(stdout)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 4: cleanup! sensitivity — does tightening/loosening change the error?
# ═══════════════════════════════════════════════════════════════════════════════
function diagnose_stage4()
    println("\n" * "=" ^ 80)
    println("STAGE 4: cleanup! tolerance sensitivity")
    println("=" ^ 80); flush(stdout)

    sections = _load_nb_sections(; verbose=false)

    @variables _r_T _χ_T _a_T _dT4 _dT5
    var_list = Num[_r_T, _χ_T, _a_T, _dT4, _dT5]
    ctx = SymToPolyCtx(var_list, 1.0)

    # Just do H1 for speed
    expr_str = _box_to_str(sections["H1"])
    body = Meta.parse(expr_str)

    sym_expr = Base.invokelatest(eval, :(let _r_ = $(Num(_r_T)), _χ_ = $(Num(_χ_T)),
                                              _a_ = $(Num(_a_T)), _M_ = 1, _α_ = 1
                                              Num($body)
                                          end))

    # Reference
    @variables _r_R _χ_R _dR3 _dR4 _dR5
    var_list_R = Num[_r_R, _χ_R, _dR3, _dR4, _dR5]
    ctx_R = SymToPolyCtx(var_list_R, a_test)

    sym_ref = Base.invokelatest(eval, :(let _r_ = $(Num(_r_R)), _χ_ = $(Num(_χ_R)),
                                             _a_ = $a_test, _M_ = 1, _α_ = 1
                                             Num($body)
                                         end))
    rp_ref = symexpr_to_poly(sym_ref, ctx_R)
    cleanup!(rp_ref.num; tol=1e-15, relative=true)

    r_test, χ_test = 5.0, 0.5
    ref_val = _eval_ratpoly_rchi(rp_ref, r_test, χ_test)

    for tol_exp in [0, -5, -10, -12, -13, -14, -15, -16]
        tol = tol_exp == 0 ? 0.0 : 10.0^tol_exp

        # Re-convert (need fresh copy since cleanup! mutates)
        rp_B = symexpr_to_poly(sym_expr, ctx)
        nterms_before = length(rp_B.num.terms)

        if tol > 0
            cleanup!(rp_B.num; tol=tol, relative=true)
        end
        nterms_after = length(rp_B.num.terms)

        val_B = eval_ratpoly_rchi_a(rp_B, r_test, χ_test, a_test)
        rel = abs(val_B - ref_val) / abs(ref_val)
        @printf("  tol=%.0e: %d→%d terms, val=%.10e, rel_vs_ref=%.2e\n",
                tol, nterms_before, nterms_after, val_B, rel)
    end
    flush(stdout)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
println("Diagnosing per-a-order H_i verification error")
println("a_test = $a_test")
println()

diagnose_stage1()
diagnose_stage2()
diagnose_stage4()
diagnose_stage3()  # slowest — runs last

println("\n\nDone.")
