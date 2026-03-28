#!/usr/bin/env julia
"""
Diagnostic: per-equation denominator structure for sGB correction equations.

Goal: see the ACTUAL max_denom_per_k values and understand what factorization
the paper does that we're missing.
"""

using Printf

# Load the module
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MetricsQNM
using MetricsQNM: extract_sgb_correction_symbolic_a, DenomSig, RatPoly, SparsePoly

println("=" ^ 70)
println("DIAGNOSTIC: sGB per-equation denominator structure")
println("=" ^ 70)
flush(stdout)

# ─────────────────────────────────────────────────────────────────────
# Step 1: Run extraction with verbose=true to see per-equation clearing
# ─────────────────────────────────────────────────────────────────────
println("\n>>> Step 1: Extract sGB correction coefficients (verbose)")
println("    This takes ~85s. Watching per-equation DenomSig values.\n")
flush(stdout)

c_kdp_per_a, a_powers, max_denom_per_k =
    extract_sgb_correction_symbolic_a(; P=3, Q=1, S=1, verbose=true)

# ─────────────────────────────────────────────────────────────────────
# Step 2: Analyze per-equation clearing vs global clearing
# ─────────────────────────────────────────────────────────────────────
println("\n>>> Step 2: Per-equation vs global clearing analysis\n")
flush(stdout)

# Global LCD
global_P = maximum(d.p for d in values(max_denom_per_k))
global_Q = maximum(d.q for d in values(max_denom_per_k))
global_S = maximum(d.s for d in values(max_denom_per_k))
global_T = maximum(d.t for d in values(max_denom_per_k))

@printf("  Global LCD: Σ^%d Δ^%d (1-χ²)^%d r^%d\n", global_P, global_Q, global_S, global_T)
@printf("  GR baseline: Σ^3 Δ^1 (1-χ²)^1 r^0\n\n")

# Per-equation analysis
println("  Per-equation clearing targets:")
for k in sort(collect(keys(max_denom_per_k)))
    d = max_denom_per_k[k]
    excess_p = d.p - 3
    excess_q = d.q - 1
    excess_s = d.s - 1
    # Extra r-degree from clearing beyond GR: 2*excess_p + 2*excess_q
    extra_rdeg = 2 * excess_p + 2 * excess_q
    @printf("    eq %2d: Σ^%d Δ^%d (1-χ²)^%d r^%d  |  excess: Σ^%d Δ^%d (1-χ²)^%d  |  extra r-deg: %d\n",
            k, d.p, d.q, d.s, d.t, excess_p, excess_q, excess_s, extra_rdeg)
end
flush(stdout)

# ─────────────────────────────────────────────────────────────────────
# Step 3: Per-equation d_max analysis (r-degree after clearing)
# ─────────────────────────────────────────────────────────────────────
println("\n>>> Step 3: Per-equation max r-degree after clearing\n")
flush(stdout)

for a_exp in a_powers
    println("  a^$a_exp:")
    # Collect max r-degree per equation
    eq_dmax = Dict{Int, Int}()
    eq_terms = Dict{Int, Int}()
    for ((k, d, p), sp) in c_kdp_per_a[a_exp]
        for (e, _) in sp.terms
            eq_dmax[k] = max(get(eq_dmax, k, 0), e[1])
        end
        eq_terms[k] = get(eq_terms, k, 0) + length(sp.terms)
    end
    for k in sort(collect(keys(eq_dmax)))
        @printf("    eq %2d: max_r_deg=%2d, terms=%d\n", k, eq_dmax[k], eq_terms[k])
    end
    global_dmax = isempty(eq_dmax) ? 0 : maximum(values(eq_dmax))
    @printf("    → global d_max for a^%d = %d\n", a_exp, global_dmax)
    flush(stdout)
end

# ─────────────────────────────────────────────────────────────────────
# Step 4: What would per-equation clearing save?
# ─────────────────────────────────────────────────────────────────────
println("\n>>> Step 4: Savings from per-equation clearing\n")
flush(stdout)

# If each equation used only its own max_denom (already done),
# vs if all equations used the global LCD
for a_exp in a_powers
    println("  a^$a_exp:")
    eq_dmax_pereq = Dict{Int, Int}()
    for ((k, d, p), sp) in c_kdp_per_a[a_exp]
        for (e, _) in sp.terms
            eq_dmax_pereq[k] = max(get(eq_dmax_pereq, k, 0), e[1])
        end
    end
    per_eq_max = isempty(eq_dmax_pereq) ? 0 : maximum(values(eq_dmax_pereq))

    # What would global LCD add?
    # Each eq would need to clear to global (P,Q,S) instead of its own
    # Extra clearing per eq = (global_P - eq_P)*2 + (global_Q - eq_Q)*2
    for k in sort(collect(keys(eq_dmax_pereq)))
        d = max_denom_per_k[k]
        extra_from_global = 2*(global_P - d.p) + 2*(global_Q - d.q)
        global_dmax_k = eq_dmax_pereq[k] + extra_from_global
        @printf("    eq %2d: per-eq d_max=%2d, if global LCD: d_max=%2d (+%d)\n",
                k, eq_dmax_pereq[k], global_dmax_k, extra_from_global)
    end
    flush(stdout)
end

# ─────────────────────────────────────────────────────────────────────
# Step 5: Detailed (k,d,p) triple denominator breakdown
# ─────────────────────────────────────────────────────────────────────
println("\n>>> Step 5: Denominator variance WITHIN equations (is per-eq uniform enough?)\n")
flush(stdout)

# Re-run pass 1 logic to see individual (k,d,p) denoms before max
# We can infer from the reduced RatPolys what each triple needs
# But we don't have them anymore. Instead, analyze the spread of
# max_denom_per_k vs hypothetical per-(k,d,p) clearing.

# For now, report the per-equation clearing vs GR baseline
println("  Summary: which equations exceed GR baseline (P=3,Q=1,S=1)?")
for k in sort(collect(keys(max_denom_per_k)))
    d = max_denom_per_k[k]
    exceeds = d.p > 3 || d.q > 1 || d.s > 1 || d.t > 0
    if exceeds
        @printf("    eq %2d: EXCEEDS GR — Σ^%d Δ^%d (1-χ²)^%d r^%d\n",
                k, d.p, d.q, d.s, d.t)
    else
        @printf("    eq %2d: fits GR baseline\n", k)
    end
end
flush(stdout)

println("\n" * "=" ^ 70)
println("DONE")
println("=" ^ 70)
