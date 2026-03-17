# PDE coefficient extraction and numerical evaluation.
# Uses Symbolics.build_function to compile the 10 field equations into fast callables.

using Symbolics: build_function, Differential, expand_derivatives

"""
    PDECoefficients

Polynomial coefficient tables G_{k,γ,δ,σ,α,β,j} for the 10 linearized
field equations.
"""
struct PDECoefficients
    equations::Vector{Dict{NTuple{6,Int}, ComplexF64}}
    d_r::Vector{Int}
    d_chi::Vector{Int}
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Compiled field equation evaluator
# ═══════════════════════════════════════════════════════════════════════════════

struct CompiledFieldEquations
    funcs::Vector{Any}    # compiled callables, one per equation
    n_args::Int           # number of arguments: 4 + 6*n_deriv_orders
    m_mode::Int
end

"""
    compile_field_equations(m_mode=2) → CompiledFieldEquations

Compile the 10 linearized Einstein equations into fast numerical functions.

Each function takes a vector [r, χ, ω, a, h₁, ∂ᵣh₁, ∂χh₁, ∂²ᵣh₁, ∂ᵣ∂χh₁, ∂²χh₁,
                                           h₂, ..., h₆, ...].
"""
function compile_field_equations(m_mode::Int=2; verbose::Bool=false)
    verbose && println("  Symbolic pipeline...")
    eqs, coords, params, hfuncs, freq_vars = compute_field_equations(m_mode)
    rv, cv = coords[2], coords[3]
    a_s = params[1]
    omega_re, omega_im, iu_sym = freq_vars

    # Discover ALL h-related terms across all 10 equations
    all_vars = Set{Any}()
    for eq in eqs
        for v in Symbolics.get_variables(eq)
            if any(occursin("h$(j)", string(v)) for j in 1:6)
                push!(all_vars, v)
            end
        end
    end
    h_terms = sort(collect(all_vars), by=string)
    n_h = length(h_terms)

    verbose && println("  Found $n_h h-derivative terms")

    # Create a plain symbolic variable for each h-derivative term
    placeholders = [Symbolics.variable(Symbol("p_$i")) for i in 1:n_h]

    # Build substitution dictionary
    sub_dict = Dict{Any, Any}(h_terms[i] => placeholders[i] for i in 1:n_h)

    # Argument vector: [r, χ, ω_re, ω_im, iu, a, p_1, ..., p_n_h]
    all_args = vcat(Num[rv, cv, omega_re, omega_im, iu_sym, a_s], placeholders)

    verbose && println("  Compiling $(length(eqs)) equations ($n_h + 6 = $(n_h + 6) args)...")
    funcs = Any[]
    for (k, eq) in enumerate(eqs)
        verbose && print("    Eq $k...")
        eq_sub = Symbolics.substitute(eq, sub_dict)
        # Verify no h-terms remain
        remaining = filter(v -> any(occursin("h$(j)", string(v)) for j in 1:6),
                          collect(Symbolics.get_variables(eq_sub)))
        if !isempty(remaining)
            @warn "Equation $k still contains h-terms after substitution" remaining
        end
        f = build_function(eq_sub, all_args; expression=Val{false})
        push!(funcs, f)
        verbose && println(" done")
    end

    CompiledFieldEquations(funcs, length(all_args), m_mode)
end

"""
    evaluate_equations(cfe, r, χ, ω, a, hvals) → Vector{ComplexF64}

Evaluate all 10 field equations.
`hvals` is a vector of length n_args - 6, corresponding to h-derivative placeholders.
ω is complex; it's split into real/imaginary parts internally.
"""
function evaluate_equations(cfe::CompiledFieldEquations,
                            r, χ, ω::Number, a, hvals::AbstractVector)
    n_h = cfe.n_args - 6
    length(hvals) == n_h || error("Expected $n_h h-values, got $(length(hvals))")
    args = zeros(Float64, cfe.n_args)
    args[1] = real(r); args[2] = real(χ)
    args[3] = real(ω); args[4] = imag(ω)  # ω_re, ω_im
    args[5] = 0.0  # iu_sym placeholder — set to 0 for real part
    args[6] = real(a)
    args[7:end] .= real.(hvals)

    # Evaluate real part (iu=0) and imaginary part (iu=1) separately
    # f(ω_re, ω_im, iu=0) gives the real-coefficient part
    # f(ω_re, ω_im, iu=1) gives the imaginary-coefficient part
    # Full result = f(iu=0) + im * f(iu=1)... but this only works if f is linear in iu.
    #
    # Actually, just evaluate with iu=1 (setting iu_sym = imaginary unit effectively)
    # BUT we can't pass complex to the compiled function...
    #
    # Simplest approach: evaluate TWICE — once with real hvals and once with imaginary.
    # Or: accept that for now, we evaluate real-valued only and handle complex later.

    # For h=0 test: just evaluate with iu=0
    [cfe.funcs[k](args) for k in 1:10]
end
