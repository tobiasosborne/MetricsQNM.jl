#!/usr/bin/env julia
#
# Identify exactly which terms cleanup! drops from H4 (symbolic a path)
# and verify they cause the 1.5% error.
#
using Printf

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using MetricsQNM

import MetricsQNM: _load_nb_sections, _box_to_str,
    symexpr_to_poly, SymToPolyCtx, RatPoly, SparsePoly, DenomSig, cleanup!,
    N_VARS, ExpVec
using Symbolics

const a_test = 0.3

println("Loading H4..."); flush(stdout)
sections = _load_nb_sections(; verbose=false)
expr_str = _box_to_str(sections["H4"])
body = Meta.parse(expr_str)

# Path B: symbolic a
@variables _r_B _χ_B _a_B _dB4 _dB5
var_list_B = Num[_r_B, _χ_B, _a_B, _dB4, _dB5]
ctx_B = SymToPolyCtx(var_list_B, 1.0)

sym_B = Base.invokelatest(eval, :(let _r_ = $(Num(_r_B)), _χ_ = $(Num(_χ_B)),
                                       _a_ = $(Num(_a_B)), _M_ = 1, _α_ = 1
                                       Num($body)
                                   end))
rp_B = symexpr_to_poly(sym_B, ctx_B)

# Save raw terms before cleanup
raw_terms = copy(rp_B.num.terms)
println("Raw: $(length(raw_terms)) terms"); flush(stdout)

# Find max coefficient (this determines cleanup threshold)
max_coeff = maximum(abs, values(raw_terms))
@printf("Max coefficient: %.6e\n", max_coeff); flush(stdout)
@printf("Cleanup threshold (tol=1e-15, relative): %.6e\n", 1e-15 * max_coeff); flush(stdout)

# Apply cleanup
cleanup!(rp_B.num; tol=1e-15, relative=true)
cleaned_terms = rp_B.num.terms
println("After cleanup: $(length(cleaned_terms)) terms"); flush(stdout)

# Find dropped terms
dropped = Dict{ExpVec, Float64}()
for (e, c) in raw_terms
    if !haskey(cleaned_terms, e)
        dropped[e] = c
    end
end
println("\nDropped $(length(dropped)) terms:")
println("-" ^ 80)
@printf("%-30s  %20s  %12s\n", "exponent (r,χ,a,_,_)", "coefficient", "vs threshold")
println("-" ^ 80)
threshold = 1e-15 * max_coeff
for (e, c) in sort(collect(dropped), by=kv->abs(kv[2]), rev=true)
    @printf("r^%-3d χ^%-3d a^%-3d            %20.10e  %12.4e\n",
            e[1], e[2], e[3], c, abs(c) / threshold)
end
flush(stdout)

# What's the evaluation impact at test points?
println("\n\nEvaluation impact of dropped terms:")
println("-" ^ 80)

# Path A: numerical a (reference)
@variables _r_A _χ_A _dA3 _dA4 _dA5
var_list_A = Num[_r_A, _χ_A, _dA3, _dA4, _dA5]
ctx_A = SymToPolyCtx(var_list_A, a_test)

sym_A = Base.invokelatest(eval, :(let _r_ = $(Num(_r_A)), _χ_ = $(Num(_χ_A)),
                                       _a_ = $a_test, _M_ = 1, _α_ = 1
                                       Num($body)
                                   end))
rp_A = symexpr_to_poly(sym_A, ctx_A)
# Don't cleanup reference either (for fair comparison)

den_t = rp_B.den.t

for (r_test, chi_test) in [(5.0, 0.5), (3.0, 0.8), (2.5, 0.99)]
    ref_val = sum(c * r_test^e[1] * chi_test^e[2] for (e, c) in rp_A.num.terms) / r_test^den_t

    # Raw B value
    raw_val = sum(c * r_test^e[1] * chi_test^e[2] * a_test^e[3] for (e, c) in raw_terms) / r_test^den_t

    # Cleaned B value
    clean_val = sum(c * r_test^e[1] * chi_test^e[2] * a_test^e[3] for (e, c) in cleaned_terms) / r_test^den_t

    # Dropped contribution
    drop_val = sum(c * r_test^e[1] * chi_test^e[2] * a_test^e[3] for (e, c) in dropped) / r_test^den_t

    @printf("(r=%.1f, χ=%.2f):\n", r_test, chi_test)
    @printf("  Reference (A, no cleanup):   %+.14e\n", ref_val)
    @printf("  Raw B (no cleanup):          %+.14e  rel_vs_A=%.2e\n", raw_val, abs(raw_val - ref_val) / abs(ref_val))
    @printf("  Cleaned B:                   %+.14e  rel_vs_A=%.2e\n", clean_val, abs(clean_val - ref_val) / abs(ref_val))
    @printf("  Dropped contribution:        %+.14e\n", drop_val)
    @printf("  raw_B - cleaned_B:           %+.14e  (should = dropped)\n", raw_val - clean_val)
    println()
end
flush(stdout)

println("\nDone.")
