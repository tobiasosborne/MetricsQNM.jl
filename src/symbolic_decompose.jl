# Symbolic G coefficient extraction — the paper's method done properly.
#
# For each equation k and each h-derivative (j, α, β):
# 1. Extract the symbolic coefficient by substitution
# 2. Multiply by Σ^P Δ^Q (1-χ²)^S to clear ALL denominators
# 3. Expand to polynomial in (r, χ, ω_re, ω_im, iu, a_s)
# 4. Collect by powers of r^δ χ^σ → G_{k,γ,δ,σ,α,β,j}(a_s)
#
# This is the EXACT approach from the paper, done symbolically.

export extract_G_symbolic

"""
    extract_G_symbolic(; P=3, Q=1, S=1, verbose=false)

Extract the EXACT polynomial G coefficients from the symbolic field equations.
Returns (G_eqs, h_map, coords, params, freq_vars) where G_eqs[k] is a dictionary
mapping (γ,δ,σ,α,β,j) → symbolic coefficient (function of a_s only).

The denominator Σ^P Δ^Q (1-χ²)^S is cleared, so all G values are polynomial
in (r, χ) and all divisions have been eliminated.
"""
function extract_G_symbolic(; P::Int=3, Q::Int=1, S::Int=1, verbose::Bool=false)
    verbose && println("Computing symbolic field equations...")
    eqs, coords, params, hfuncs, freq_vars = compute_field_equations(2)
    r, chi = coords[2], coords[3]
    a_s = params[1]
    omega_re, omega_im, iu = freq_vars

    # Common denominator (polynomial in r, chi, a_s)
    Sig = r^2 + a_s^2 * chi^2
    Del = r^2 - 2r + a_s^2    # Δ = r² - 2r + a² (M=1)
    s2  = 1 - chi^2
    denom = Sig^P * Del^Q * s2^S

    # Discover all h-derivative terms
    all_vars = Set{Any}()
    for eq in eqs, v in Symbolics.get_variables(eq)
        any(occursin("h$j", string(v)) for j in 1:6) && push!(all_vars, v)
    end
    h_terms = sort(collect(all_vars), by=string)
    h_map = _parse_h_terms(string.(h_terms))
    sub_zero = Dict{Any,Any}(t => Num(0) for t in h_terms)

    n_h = length(h_terms)
    n_eqs = length(eqs)

    verbose && println("Extracting coefficients for $n_h terms × $n_eqs equations...")

    # For each (k, d): extract coefficient, clear denominator, expand
    cleared_coeffs = Vector{Any}(undef, n_eqs * n_h)
    for k in 1:n_eqs
        for (d, ht) in enumerate(h_terms)
            sub = copy(sub_zero)
            sub[ht] = Num(1)
            coeff = Symbolics.substitute(eqs[k], sub)

            # Multiply by denominator and expand
            if !isequal(coeff, Num(0))
                cleared = Symbolics.expand(coeff * denom)
                # Simplify to clean up any remaining fractions
                cleared = Symbolics.simplify(cleared; expand=true)
            else
                cleared = Num(0)
            end

            cleared_coeffs[(k-1)*n_h + d] = cleared

            if verbose && d % 10 == 0
                println("  eq $k, term $d/$n_h: $(length(string(cleared))) chars")
            end
        end
        verbose && println("  Equation $k done")
    end

    return cleared_coeffs, h_map, h_terms, coords, params, freq_vars, denom
end
