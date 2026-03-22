#!/usr/bin/env julia
#
# Term-by-term comparison of H4 from numerical-a vs symbolic-a paths.
# Identifies exactly which monomials (r^i * χ^j) differ.
#
using Printf

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using MetricsQNM

import MetricsQNM: _load_nb_sections, _box_to_str,
    symexpr_to_poly, SymToPolyCtx, RatPoly, SparsePoly, DenomSig, cleanup!,
    N_VARS, ExpVec
using Symbolics

const a_test = 0.3

println("Loading H4 expression from notebook..."); flush(stdout)
sections = _load_nb_sections(; verbose=false)
expr_str = _box_to_str(sections["H4"])
body = Meta.parse(expr_str)

# ═══════════════════════════════════════════════════════════════════════════════
#  Path A: numerical a → RatPoly in (r, χ)
# ═══════════════════════════════════════════════════════════════════════════════
println("Path A: numerical a=$a_test..."); flush(stdout)
@variables _r_A _χ_A _dA3 _dA4 _dA5
var_list_A = Num[_r_A, _χ_A, _dA3, _dA4, _dA5]
ctx_A = SymToPolyCtx(var_list_A, a_test)

sym_A = Base.invokelatest(eval, :(let _r_ = $(Num(_r_A)), _χ_ = $(Num(_χ_A)),
                                       _a_ = $a_test, _M_ = 1, _α_ = 1
                                       Num($body)
                                   end))
rp_A = symexpr_to_poly(sym_A, ctx_A)
# Do NOT cleanup — compare raw
@printf("  %d terms, den.t=%d\n", length(rp_A.num.terms), rp_A.den.t); flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════════
#  Path B: symbolic a → RatPoly in (r, χ, a)
# ═══════════════════════════════════════════════════════════════════════════════
println("Path B: symbolic a..."); flush(stdout)
@variables _r_B _χ_B _a_B _dB4 _dB5
var_list_B = Num[_r_B, _χ_B, _a_B, _dB4, _dB5]
ctx_B = SymToPolyCtx(var_list_B, 1.0)

sym_B = Base.invokelatest(eval, :(let _r_ = $(Num(_r_B)), _χ_ = $(Num(_χ_B)),
                                       _a_ = $(Num(_a_B)), _M_ = 1, _α_ = 1
                                       Num($body)
                                   end))
rp_B = symexpr_to_poly(sym_B, ctx_B)
@printf("  %d terms, den.t=%d\n", length(rp_B.num.terms), rp_B.den.t); flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════════
#  Collapse Path B to (r, χ) by evaluating a-variable at a_test
# ═══════════════════════════════════════════════════════════════════════════════
println("\nCollapsing Path B by evaluating a=$a_test..."); flush(stdout)

collapsed_B = Dict{Tuple{Int,Int}, Float64}()
for (e, c) in rp_B.num.terms
    r_exp, chi_exp, a_exp = e[1], e[2], e[3]
    key = (r_exp, chi_exp)
    collapsed_B[key] = get(collapsed_B, key, 0.0) + c * a_test^a_exp
end
@printf("  Collapsed to %d unique (r,χ) monomials\n", length(collapsed_B)); flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════════
#  Compare term-by-term
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 90)
println("TERM-BY-TERM COMPARISON (sorted by |difference|, showing top 30)")
println("=" ^ 90)

all_keys = union(Set((e[1], e[2]) for (e, _) in rp_A.num.terms),
                 Set(keys(collapsed_B)))

diffs = []
for (r_exp, chi_exp) in all_keys
    c_A = 0.0
    for (e, c) in rp_A.num.terms
        if e[1] == r_exp && e[2] == chi_exp
            c_A = c
            break
        end
    end
    c_B = get(collapsed_B, (r_exp, chi_exp), 0.0)
    push!(diffs, (r_exp, chi_exp, c_A, c_B, abs(c_A - c_B)))
end
sort!(diffs, by=x->x[5], rev=true)

@printf("%-8s %-8s  %20s  %20s  %16s  %12s\n",
        "r_exp", "χ_exp", "coeff_A (num a)", "coeff_B (sym a)", "abs_diff", "rel_diff")
println("-" ^ 90)
for (r_exp, chi_exp, c_A, c_B, d) in diffs[1:min(30, end)]
    scale = max(abs(c_A), abs(c_B), 1e-30)
    rel = d / scale
    flag = rel > 1e-10 ? " ***" : ""
    @printf("r^%-4d  χ^%-4d  %20.12e  %20.12e  %16.4e  %12.2e%s\n",
            r_exp, chi_exp, c_A, c_B, d, rel, flag)
end
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════════
#  Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 90)
println("SUMMARY")
println("=" ^ 90)

n_only_A = count(d -> d[3] != 0 && d[4] == 0, diffs)
n_only_B = count(d -> d[3] == 0 && d[4] != 0, diffs)
n_both_exact = count(d -> d[3] != 0 && d[4] != 0 && d[5] < 1e-15 * max(abs(d[3]), abs(d[4])), diffs)
n_both_differ = count(d -> d[3] != 0 && d[4] != 0 && d[5] >= 1e-15 * max(abs(d[3]), abs(d[4])), diffs)
n_chi0_differ = count(d -> d[2] == 0 && d[5] > 1e-10 * max(abs(d[3]), abs(d[4]), 1e-30), diffs)
n_chi_nonzero_differ = count(d -> d[2] > 0 && d[5] > 1e-10 * max(abs(d[3]), abs(d[4]), 1e-30), diffs)

println("Total (r,χ) monomials: $(length(diffs))")
println("  Only in A (numerical): $n_only_A")
println("  Only in B (symbolic):  $n_only_B")
println("  In both, match:        $n_both_exact")
println("  In both, differ:       $n_both_differ")
println("  Differing with χ^0:    $n_chi0_differ")
println("  Differing with χ^k>0:  $n_chi_nonzero_differ")
flush(stdout)

# Check: are all differing terms χ-independent?
if n_chi_nonzero_differ == 0 && n_chi0_differ > 0
    println("\n*** CONFIRMED: All errors are in χ-independent terms (χ^0) ***")
elseif n_chi_nonzero_differ > 0
    println("\n*** SOME errors are in χ-dependent terms too ***")
    # Show the χ-dependent ones
    for (r_exp, chi_exp, c_A, c_B, d) in diffs
        chi_exp == 0 && continue
        scale = max(abs(c_A), abs(c_B), 1e-30)
        d / scale < 1e-10 && continue
        @printf("  r^%-4d χ^%-4d: A=%.6e B=%.6e diff=%.2e\n", r_exp, chi_exp, c_A, c_B, d)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  For the worst monomial, show a-order contributions from Path B
# ═══════════════════════════════════════════════════════════════════════════════
worst = diffs[1]
r_exp_w, chi_exp_w = worst[1], worst[2]
println("\n\nPer-a-order breakdown for worst monomial r^$(r_exp_w) χ^$(chi_exp_w):")
println("-" ^ 60)

a_contribs = Tuple{Int, Float64}[]
for (e, c) in rp_B.num.terms
    if e[1] == r_exp_w && e[2] == chi_exp_w
        push!(a_contribs, (e[3], c))
    end
end
sort!(a_contribs, by=first)

running = 0.0
for (a_exp, c) in a_contribs
    contrib = c * a_test^a_exp
    running += contrib
    @printf("  a^%2d: raw_coeff=%+.10e  × a^%d=%+.10e  cumsum=%+.14e\n",
            a_exp, c, a_exp, contrib, running)
end
@printf("\n  Path B collapsed: %+.14e\n", running)
@printf("  Path A direct:    %+.14e\n", worst[3])
@printf("  Difference:       %+.4e\n", running - worst[3])

println("\nDone.")
