# Symbolic G coefficient extraction — fast implementation.
#
# Uses direct SymbolicUtils Add.dict / Mul.dict tree walking for O(n)
# monomial coefficient extraction, bypassing the slow polynomial_coeffs API.
#
# Pipeline:
# 1. compute_field_equations → 10 symbolic equations
# 2. For each (k, d): substitute h-terms → extract coefficient (~0.25s)
# 3. Multiply by Σ^P Δ^Q (1-χ²)^S, expand (~0.04s)
# 4. Substitute a = numerical value
# 5. Walk Add.dict tree to extract ALL (r^δ, χ^σ, ω_re^p, ω_im^q, iu^s) monomials in one O(n) pass
# 6. Reconstruct complex ω coefficients via iu→i trick
# 7. Transform r → z via binomial expansion
# 8. Feed to assembly.jl → exact D̃₀, D̃₁, D̃₂
#
# With threading: ~20-30s total for 400 coefficients.

export extract_G_exact, build_system_symbolic

using Symbolics
using Symbolics: SymbolicUtils

# ═══════════════════════════════════════════════════════════════════════════════
#  Fast monomial extraction via Add.dict / Mul.dict tree walking
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _walk_expanded_poly(expr_num, var_syms) → Dict{NTuple{N,Int}, Float64}

Given an expanded Symbolics expression and a list of symbolic variables,
extract ALL monomial coefficients in a single O(n_terms) pass.

Returns Dict mapping exponent tuples to Float64 coefficients.
All variables in `var_syms` must be simple Sym nodes (not functions).
The expression must be fully expanded and numerical (no remaining symbolic params).
"""
function _walk_expanded_poly(expr_num::Num, var_syms::Vector)
    expr = Symbolics.unwrap(expr_num)
    n = length(var_syms)
    # Build identity map: BasicSymbolic → index
    var_idx = Dict{Any,Int}()
    for (i, v) in enumerate(var_syms)
        var_idx[Symbolics.unwrap(v)] = i
    end

    result = Dict{NTuple{n,Int}, Float64}()

    function add_term!(exps_tuple::NTuple{N,Int}, coeff::Float64) where N
        result[exps_tuple] = get(result, exps_tuple, 0.0) + coeff
    end

    # Parse a single Mul or Sym or Pow or Number into (exponent_tuple, numeric_coeff)
    function parse_monomial(term, multiplier::Float64)
        exps = zeros(Int, n)

        if term isa Number
            add_term!(NTuple{n,Int}(exps), multiplier * Float64(term))
            return
        end

        if SymbolicUtils.issym(term)
            idx = get(var_idx, term, 0)
            if idx > 0
                exps[idx] = 1
                add_term!(NTuple{n,Int}(exps), multiplier)
            else
                # Unknown symbol — treat as constant (shouldn't happen after full substitution)
                @warn "Unknown symbol in polynomial: $term"
                add_term!(NTuple{n,Int}(exps), multiplier)
            end
            return
        end

        if SymbolicUtils.ispow(term)
            args = SymbolicUtils.arguments(term)
            base, exp_val = args[1], args[2]
            idx = get(var_idx, base, 0)
            if idx > 0 && exp_val isa Number
                exps[idx] = Int(exp_val)
                add_term!(NTuple{n,Int}(exps), multiplier)
            else
                # Treat as constant
                add_term!(NTuple{n,Int}(exps), multiplier)
            end
            return
        end

        if SymbolicUtils.ismul(term)
            coeff = Float64(term.coeff) * multiplier
            for (base, exp_val) in term.dict
                idx = get(var_idx, base, 0)
                if idx > 0 && exp_val isa Number
                    exps[idx] = Int(exp_val)
                elseif base isa Number
                    # Numeric base left over from substitution — fold into coefficient
                    coeff *= Float64(base)^Float64(exp_val)
                elseif SymbolicUtils.ispow(base) && exp_val isa Number
                    pargs = SymbolicUtils.arguments(base)
                    pb, pe = pargs[1], pargs[2]
                    pidx = get(var_idx, pb, 0)
                    if pidx > 0 && pe isa Number
                        exps[pidx] = Int(pe * exp_val)
                    else
                        # Numeric pow — fold into coefficient
                        try; coeff *= Float64(pb)^Float64(pe * exp_val)
                        catch; @warn "Unhandled pow base in Mul: $base ^ $exp_val"; end
                    end
                elseif SymbolicUtils.issym(base)
                    bidx = get(var_idx, base, 0)
                    if bidx > 0
                        exps[bidx] = Int(exp_val)
                    end
                else
                    # Last resort: try numeric conversion
                    try; coeff *= Float64(base)^Float64(exp_val)
                    catch; @warn "Unhandled factor in Mul: $(typeof(base)) ^ $exp_val"; end
                end
            end
            add_term!(NTuple{n,Int}(exps), coeff)
            return
        end

        # Fallback: treat as constant — try multiple conversion paths
        try
            add_term!(NTuple{n,Int}(exps), multiplier * Float64(term))
            return
        catch; end
        try
            add_term!(NTuple{n,Int}(exps), multiplier * Float64(Symbolics.value(Num(term))))
            return
        catch; end
        @warn "Could not parse term: $(typeof(term))"
    end

    # Main dispatch
    if SymbolicUtils.isadd(expr)
        # Constant term
        if !iszero(expr.coeff)
            parse_monomial(expr.coeff, 1.0)
        end
        # Sum terms: each key in dict is a term, value is its multiplier
        for (term, multiplier) in expr.dict
            parse_monomial(term, Float64(multiplier))
        end
    elseif expr isa Number
        exps = ntuple(_ -> 0, n)
        add_term!(exps, Float64(expr))
    else
        parse_monomial(expr, 1.0)
    end

    return result
end

# ═══════════════════════════════════════════════════════════════════════════════
#  ω-decomposition from (ω_re, ω_im, iu) monomials
# ═══════════════════════════════════════════════════════════════════════════════

"""
Given monomial exponents (p_ωre, q_ωim, s_iu) and a Float64 coefficient,
compute the contribution to G₀, G₁, G₂ where D̃(ω) = D₀ + D₁ω + D₂ω².

Uses the identity: physical ω = ω_re + i·ω_im, iu → i.
Evaluates the monomial ω_re^p · ω_im^q · iu^s at ω = 0, 1, -1
(with ω_im = 0 for real ω) and applies the iu→i trick.
"""
function _omega_monomial_to_G(p_ωre::Int, q_ωim::Int, s_iu::Int, coeff::Float64)
    # For ω = real value w: ω_re = w, ω_im = 0
    # f(w) = w^p · 0^q · iu^s = (q==0 ? w^p : 0) · iu^s
    # For q > 0: the monomial vanishes at real ω (ω_im = 0)
    if q_ωim > 0
        return (0.0im, 0.0im, 0.0im)
    end

    # iu^s at iu = 0, 1, -1:
    iu_0  = s_iu == 0 ? 1.0 : 0.0
    iu_1  = 1.0  # 1^s = 1
    iu_m1 = iseven(s_iu) ? 1.0 : -1.0  # (-1)^s

    # f(ω, iu) = ω^p · iu^s · coeff, evaluated at real ω values
    # At ω=0: f = 0^p · iu^s · coeff = (p==0 ? coeff : 0) · iu^s
    f0_iu0  = (p_ωre == 0 ? coeff : 0.0) * iu_0
    f0_iu1  = (p_ωre == 0 ? coeff : 0.0) * iu_1
    f0_ium1 = (p_ωre == 0 ? coeff : 0.0) * iu_m1

    # At ω=1: f = 1 · iu^s · coeff
    f1_iu0  = coeff * iu_0
    f1_iu1  = coeff * iu_1
    f1_ium1 = coeff * iu_m1

    # At ω=-1: f = (-1)^p · iu^s · coeff
    mp = iseven(p_ωre) ? coeff : -coeff
    fm1_iu0  = mp * iu_0
    fm1_iu1  = mp * iu_1
    fm1_ium1 = mp * iu_m1

    # Complex reconstruction: f(iu→i) = (f_iu0 - f_iu2) + i·f_iu1
    # where f_iu2 = (f_iu1 + f_ium1)/2 - f_iu0
    function to_complex(f0, f1, fm1)
        a0 = f0
        a1 = (f1 - fm1) / 2
        a2 = (f1 + fm1) / 2 - f0
        return complex(a0 - a2, a1)
    end

    val_at_0  = to_complex(f0_iu0,  f0_iu1,  f0_ium1)
    val_at_1  = to_complex(f1_iu0,  f1_iu1,  f1_ium1)
    val_at_m1 = to_complex(fm1_iu0, fm1_iu1, fm1_ium1)

    # ω-separation: G₀ + G₁ω + G₂ω²
    G0 = val_at_0
    G1 = (val_at_1 - val_at_m1) / 2
    G2 = (val_at_1 + val_at_m1) / 2 - val_at_0

    return (G0, G1, G2)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  r → z transform via binomial expansion
# ═══════════════════════════════════════════════════════════════════════════════

function _r_to_z(K_r::PDECoefficients, rp::Float64, d_max::Int)
    K_z = PDECoefficients([Dict{NTuple{6,Int}, ComplexF64}() for _ in 1:10],
                          fill(d_max, 10), fill(20, 10))
    # Precompute binomial coefficients as Float64 via BigInt to avoid Int64 overflow
    # (sGB correction has d_max up to 80; binomial(77,22) > typemax(Int64))
    binom = [Float64(binomial(big(n), big(k))) for n in 0:d_max, k in 0:d_max]
    for k in 1:10
        for ((γ, δ, σ, α, β, j), G_val) in K_r.equations[k]
            δ > d_max && continue
            coeff = (2rp)^δ * G_val
            for jz in 0:(d_max - δ)
                K_val = coeff * binom[d_max - δ + 1, jz + 1]
                abs(K_val) < 1e-15 && continue
                key = (0, jz, σ, α, β, j)
                K_z.equations[k][key] = get(K_z.equations[k], key, 0.0im) + K_val
            end
        end
    end
    return K_z
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Main extraction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    extract_G_exact(a; P=3, Q=1, S=1, verbose=false) → (K₀, K₁, K₂)

Extract exact polynomial G coefficients using direct tree-walking.
Returns three PDECoefficients in r-space for ω⁰, ω¹, ω² terms.
"""
function extract_G_exact(a::Float64; P::Int=3, Q::Int=1, S::Int=1,
                         verbose::Bool=false)
    verbose && println("  Computing field equations...")
    eqs, coords, params, hfuncs, freq_vars = compute_field_equations(2)
    r, chi = coords[2], coords[3]
    a_s = params[1]
    omega_re, omega_im, iu_sym = freq_vars

    # Discover h-derivative terms
    all_vars = Set{Any}()
    for eq in eqs, v in Symbolics.get_variables(eq)
        any(occursin("h$j", string(v)) for j in 1:6) && push!(all_vars, v)
    end
    h_terms = sort(collect(all_vars), by=string)
    h_map = _parse_h_terms(string.(h_terms))
    sub_zero = Dict{Any,Any}(t => Num(0) for t in h_terms)
    n_h = length(h_terms)
    n_eqs = length(eqs)

    # Common denominator
    Sig = r^2 + a_s^2 * chi^2
    Del = r^2 - 2r + a_s^2
    s2  = 1 - chi^2
    denom = Sig^P * Del^Q * s2^S

    verbose && println("  Extracting $n_h × $n_eqs = $(n_h * n_eqs) coefficients...")

    # Variables for monomial extraction (after substituting a_s)
    var_list = Num[r, chi, omega_re, omega_im, iu_sym]

    # Result: three PDECoefficients for ω⁰, ω¹, ω²
    Ks = [PDECoefficients([Dict{NTuple{6,Int}, ComplexF64}() for _ in 1:10],
                          fill(30, 10), fill(20, 10)) for _ in 1:3]

    t_start = time()

    # Flatten (k, d) pairs for parallel processing
    tasks = [(k, d) for k in 1:n_eqs for d in 1:n_h]
    n_tasks = length(tasks)

    # Per-task results: Vector of (k, j, α, β, δ, σ, G0, G1, G2) tuples
    task_results = Vector{Vector{Tuple{Int,Int,Int,Int,Int,Int,ComplexF64,ComplexF64,ComplexF64}}}(undef, n_tasks)

    n_threads = Threads.nthreads()
    verbose && println("  Using $n_threads threads for $n_tasks tasks...")

    done_count = Threads.Atomic{Int}(0)

    Threads.@threads for idx in 1:n_tasks
        k, d = tasks[idx]
        ht = h_terms[d]
        local_results = Tuple{Int,Int,Int,Int,Int,Int,ComplexF64,ComplexF64,ComplexF64}[]

        # Extract coefficient of this h-derivative
        sub = copy(sub_zero)
        sub[ht] = Num(1)
        coeff = Symbolics.substitute(eqs[k], sub)
        if isequal(coeff, Num(0))
            task_results[idx] = local_results
            Threads.atomic_add!(done_count, 1)
            dc = done_count[]
            if verbose && (dc % 25 == 0 || dc == n_tasks)
                elapsed = time() - t_start
                rate = dc / elapsed
                eta = (n_tasks - dc) / max(rate, 0.01)
                println("    [$dc/$n_tasks] $(round(elapsed,digits=1))s elapsed, ~$(round(eta,digits=0))s remaining ($(round(rate,digits=1)) tasks/s)")
                flush(stdout)
            end
            continue
        end

        # Substitute a_s, simplify fractions (cancel Σ/Δ denominators), then expand
        # The field equations are rational in r,χ — simplify_fractions clears inner
        # denominators that expand() alone cannot handle.
        raw = Symbolics.substitute(coeff * denom, Dict(a_s => a))
        cleared_num = Symbolics.expand(Symbolics.simplify_fractions(raw))

        # Extract ALL monomial coefficients via fast tree walking (~6x faster than polynomial_coeffs)
        mono_coeffs = _walk_expanded_poly(cleared_num, var_list)

        j_d, α_d, β_d = h_map[d]

        for (exps, coeff_val) in mono_coeffs
            abs(coeff_val) < 1e-15 && continue

            δ, σ, p_ω, q_ω, s_iu = exps
            G0, G1, G2 = _omega_monomial_to_G(p_ω, q_ω, s_iu, coeff_val)

            for (γ, Gval) in enumerate((G0, G1, G2))
                abs(Gval) < 1e-15 && continue
                push!(local_results, (k, j_d, α_d, β_d, δ, σ,
                      γ == 1 ? Gval : 0.0im,
                      γ == 2 ? Gval : 0.0im,
                      γ == 3 ? Gval : 0.0im))
            end
        end
        task_results[idx] = local_results

        Threads.atomic_add!(done_count, 1)
        dc = done_count[]
        if verbose && (dc % 25 == 0 || dc == n_tasks)
            elapsed = time() - t_start
            rate = dc / elapsed
            eta = (n_tasks - dc) / max(rate, 0.01)
            println("    [$dc/$n_tasks] $(round(elapsed,digits=1))s elapsed, ~$(round(eta,digits=0))s remaining ($(round(rate,digits=1)) tasks/s)")
            flush(stdout)
        end
    end

    # Merge results into Ks
    for idx in 1:n_tasks
        for (k, j_d, α_d, β_d, δ, σ, g0, g1, g2) in task_results[idx]
            for (γ, Gval) in enumerate((g0, g1, g2))
                abs(Gval) < 1e-15 && continue
                key = (γ - 1, δ, σ, α_d, β_d, j_d)
                Ks[γ].equations[k][key] =
                    get(Ks[γ].equations[k], key, 0.0im) + Gval
            end
        end
    end

    verbose && println("  Total extraction: $(round(time() - t_start, digits=1))s")
    return Ks[1], Ks[2], Ks[3]
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Bespoke extraction via SparsePoly (replaces simplify_fractions pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

export extract_G_bespoke, build_system_bespoke, normalize_system!, sweep_N

"""
    extract_G_bespoke(a; P=3, Q=1, S=1, verbose=false) → (K₀, K₁, K₂)

Extract polynomial G coefficients using the bespoke SparsePoly CAS.
Single-pass tree walk with structural denominator tracking — no
simplify_fractions, no expand, no GCD computation.

Returns three PDECoefficients in r-space for ω⁰, ω¹, ω² terms.
"""
function extract_G_bespoke(a::Float64; P::Int=3, Q::Int=1, S::Int=1,
                            verbose::Bool=false)
    verbose && (println("  Computing field equations..."); flush(stdout))
    eqs, coords, params, hfuncs, freq_vars = compute_field_equations(2)
    r, chi = coords[2], coords[3]
    a_s = params[1]
    omega_re, omega_im, iu_sym = freq_vars

    # Discover h-derivative terms
    all_vars = Set{Any}()
    for eq in eqs, v in Symbolics.get_variables(eq)
        any(occursin("h$j", string(v)) for j in 1:6) && push!(all_vars, v)
    end
    h_terms = sort(collect(all_vars), by=string)
    h_map = _parse_h_terms(string.(h_terms))
    sub_zero = Dict{Any,Any}(t => Num(0) for t in h_terms)
    n_h = length(h_terms)
    n_eqs = length(eqs)

    var_list = Num[r, chi, omega_re, omega_im, iu_sym]
    ctx = SymToPolyCtx(var_list, a)

    verbose && (println("  Extracting $n_h × $n_eqs = $(n_h * n_eqs) coefficients via SparsePoly..."); flush(stdout))

    # Result: three PDECoefficients for ω⁰, ω¹, ω²
    Ks = [PDECoefficients([Dict{NTuple{6,Int}, ComplexF64}() for _ in 1:10],
                          fill(30, 10), fill(20, 10)) for _ in 1:3]

    t_start = time()
    tasks = [(k, d) for k in 1:n_eqs for d in 1:n_h]
    n_tasks = length(tasks)

    task_results = Vector{Vector{Tuple{Int,Int,Int,Int,Int,Int,ComplexF64,ComplexF64,ComplexF64}}}(undef, n_tasks)

    done_count = Threads.Atomic{Int}(0)

    # Pass 1: convert all tasks to RatPoly (serial — Symbolics is not thread-safe)
    ratpolys = Vector{Union{Nothing, RatPoly}}(undef, n_tasks)

    for idx in 1:n_tasks
        k, d = tasks[idx]
        sub = copy(sub_zero)
        sub[h_terms[d]] = Num(1)
        coeff = Symbolics.substitute(eqs[k], sub)
        if isequal(coeff, Num(0))
            ratpolys[idx] = nothing
            continue
        end
        coeff_a = Symbolics.substitute(coeff, Dict(a_s => a))
        ratpolys[idx] = symexpr_to_poly(coeff_a, ctx)
    end

    n_nonzero = count(!isnothing, ratpolys)
    verbose && (println("  $n_nonzero non-zero tasks, per-task clearing (P≥$P, Q≥$Q, S≥$S)"); flush(stdout))

    # Pass 2: reduce + per-task clear + extract monomials (threaded)
    Threads.@threads for idx in 1:n_tasks
        k, d = tasks[idx]
        local_results = Tuple{Int,Int,Int,Int,Int,Int,ComplexF64,ComplexF64,ComplexF64}[]

        rp = ratpolys[idx]
        if rp === nothing
            task_results[idx] = local_results
            Threads.atomic_add!(done_count, 1)
            continue
        end

        # Per-task: reduce (cancel common factors) then clear to minimum
        poly, _ = clear_denominators(rp, P, Q, S, ctx)
        mono_coeffs = poly.terms

        j_d, α_d, β_d = h_map[d]

        for (exps, coeff_val) in mono_coeffs
            abs(coeff_val) < 1e-15 && continue

            δ, σ, p_ω, q_ω, s_iu = exps
            G0, G1, G2 = _omega_monomial_to_G(p_ω, q_ω, s_iu, coeff_val)

            for (γ, Gval) in enumerate((G0, G1, G2))
                abs(Gval) < 1e-15 && continue
                push!(local_results, (k, j_d, α_d, β_d, δ, σ,
                      γ == 1 ? Gval : 0.0im,
                      γ == 2 ? Gval : 0.0im,
                      γ == 3 ? Gval : 0.0im))
            end
        end
        task_results[idx] = local_results

        Threads.atomic_add!(done_count, 1)
        dc = done_count[]
        if verbose && (dc % 50 == 0 || dc == n_tasks)
            elapsed = time() - t_start
            @printf("    [%d/%d] %.1fs elapsed (%.1f tasks/s)\n", dc, n_tasks, elapsed, dc/elapsed)
            flush(stdout)
        end
    end

    # Merge results into Ks
    for idx in 1:n_tasks
        for (k, j_d, α_d, β_d, δ, σ, g0, g1, g2) in task_results[idx]
            for (γ, Gval) in enumerate((g0, g1, g2))
                abs(Gval) < 1e-15 && continue
                key = (γ - 1, δ, σ, α_d, β_d, j_d)
                Ks[γ].equations[k][key] =
                    get(Ks[γ].equations[k], key, 0.0im) + Gval
            end
        end
    end

    verbose && (println("  Total extraction: $(round(time() - t_start, digits=1))s"); flush(stdout))
    return Ks[1], Ks[2], Ks[3]
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Per-equation normalization
# ═══════════════════════════════════════════════════════════════════════════════

"""
    normalize_system!(sys::METRICSSystem) → (sys, factors)

Per-equation normalization: for each of the 10 field equations, scale the
corresponding rows of D₀, D₁, D₂ so the largest coefficient has modulus 1.
Mandatory for numerical stability (cf. arxiv:2312.08435, lines 491-492).

Returns the system and the 10-element vector of scale factors used.
"""
function normalize_system!(sys::METRICSSystem)
    N = sys.N
    bs = (N + 1)^2
    factors = Vector{Float64}(undef, 10)
    for k in 1:10
        rows = ((k-1)*bs+1):(k*bs)
        scale = max(
            maximum(abs, view(sys.D0, rows, :); init=0.0),
            maximum(abs, view(sys.D1, rows, :); init=0.0),
            maximum(abs, view(sys.D2, rows, :); init=0.0)
        )
        factors[k] = scale
        if scale > 0
            sys.D0[rows,:] ./= scale
            sys.D1[rows,:] ./= scale
            sys.D2[rows,:] ./= scale
        end
    end
    return sys, factors
end

"""
    normalize_system!(sys::METRICSSystem, factors::Vector{Float64}) → sys

Apply pre-computed per-equation normalization factors to a system.
Use this to ensure D̃⁽¹⁾ uses the SAME normalization as D̃⁽⁰⁾ in perturbation theory.
"""
function normalize_system!(sys::METRICSSystem, factors::Vector{Float64})
    N = sys.N
    bs = (N + 1)^2
    for k in 1:10
        factors[k] > 0 || continue
        rows = ((k-1)*bs+1):(k*bs)
        sys.D0[rows,:] ./= factors[k]
        sys.D1[rows,:] ./= factors[k]
        sys.D2[rows,:] ./= factors[k]
    end
    return sys
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Full pipelines
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_system_bespoke(a, N, m; P=3, Q=1, S=1, d_max_override=nothing, verbose=false)

Build D̃₀, D̃₁, D̃₂ via bespoke SparsePoly G extraction + r→z transform + assembly.
Fast: ~5s total for extraction (vs minutes with simplify_fractions).

If `d_max_override` is provided, use it instead of the naturally computed d_max.
This is needed for sGB perturbation theory: D̃⁽⁰⁾ and D̃⁽¹⁾ must share the
same d_max (same polynomial-space weighting in the Galerkin projection).
"""
function build_system_bespoke(a::Float64, N::Int, m::Int;
                               P::Int=3, Q::Int=1, S::Int=1,
                               d_max_override::Union{Nothing,Int}=nothing,
                               verbose::Bool=false)
    rp = r_plus(a)

    verbose && (println("Extracting G coefficients (bespoke)..."); flush(stdout))
    K0_r, K1_r, K2_r = extract_G_bespoke(a; P, Q, S, verbose)

    if verbose
        for (name, K) in [("K0", K0_r), ("K1", K1_r), ("K2", K2_r)]
            n = sum(length(d) for d in K.equations)
            println("  $name: $n terms (r-space)")
        end
        flush(stdout)
    end

    d_max = 0
    for K in (K0_r, K1_r, K2_r), k in 1:10
        for ((γ, δ, σ, α, β, j), _) in K.equations[k]
            d_max = max(d_max, δ)
        end
    end

    if d_max_override !== nothing
        d_max_override < d_max && @warn "d_max_override=$d_max_override < natural d_max=$d_max"
        d_max = max(d_max, d_max_override)
    end
    verbose && (println("  Max r-degree: $d_max$(d_max_override !== nothing ? " (override)" : "")"); flush(stdout))

    verbose && (println("Transforming r → z (d_max=$d_max)..."); flush(stdout))
    K0_z = _r_to_z(K0_r, rp, d_max)
    K1_z = _r_to_z(K1_r, rp, d_max)
    K2_z = _r_to_z(K2_r, rp, d_max)

    verbose && (println("Assembling D̃ matrices (N=$N)..."); flush(stdout))
    basis = spectral_basis(N, m; max_delta=d_max)
    D0 = assemble_system(K0_z, basis, a).D0
    D1 = assemble_system(K1_z, basis, a).D0
    D2 = assemble_system(K2_z, basis, a).D0

    sys = METRICSSystem(D0, D1, D2, N, m, a)
    sys, norm_factors = normalize_system!(sys)
    verbose && (println("  Per-equation normalization applied"); flush(stdout))
    return sys, norm_factors
end

"""
    sweep_N(a, m, N_range; P=3, Q=1, S=1, verbose=false) → Dict{Int, METRICSSystem}

Build systems for multiple N values efficiently: extract G coefficients once,
then assemble + normalize for each N. ~5s extraction + ~0.1s per N.
"""
function sweep_N(a::Float64, m::Int, N_range;
                  P::Int=3, Q::Int=1, S::Int=1, verbose::Bool=false)
    rp = r_plus(a)

    verbose && (println("Extracting G coefficients (a=$a)..."); flush(stdout))
    K0_r, K1_r, K2_r = extract_G_bespoke(a; P, Q, S, verbose)

    d_max = 0
    for K in (K0_r, K1_r, K2_r), k in 1:10
        for ((γ, δ, σ, α, β, j), _) in K.equations[k]
            d_max = max(d_max, δ)
        end
    end
    verbose && (println("  d_max=$d_max, transforming r→z..."); flush(stdout))

    K0_z = _r_to_z(K0_r, rp, d_max)
    K1_z = _r_to_z(K1_r, rp, d_max)
    K2_z = _r_to_z(K2_r, rp, d_max)

    systems = Dict{Int, METRICSSystem}()
    for N in N_range
        basis = spectral_basis(N, m)
        D0 = assemble_system(K0_z, basis, a).D0
        D1 = assemble_system(K1_z, basis, a).D0
        D2 = assemble_system(K2_z, basis, a).D0
        sys = METRICSSystem(D0, D1, D2, N, m, a)
        sys, _ = normalize_system!(sys)
        systems[N] = sys
    end
    verbose && (println("  Built $(length(systems)) systems (N=$(first(N_range))..$(last(N_range)))"); flush(stdout))
    return systems
end

"""
    build_system_symbolic(a, N, m; P=3, Q=1, S=1, verbose=false) → METRICSSystem

Build exact D̃₀, D̃₁, D̃₂ via symbolic G extraction + r→z transform + assembly.
"""
function build_system_symbolic(a::Float64, N::Int, m::Int;
                               P::Int=3, Q::Int=1, S::Int=1,
                               verbose::Bool=false)
    rp = r_plus(a)

    verbose && println("Extracting G coefficients...")
    K0_r, K1_r, K2_r = extract_G_exact(a; P, Q, S, verbose)

    if verbose
        for (name, K) in [("K0", K0_r), ("K1", K1_r), ("K2", K2_r)]
            n = sum(length(d) for d in K.equations)
            println("  $name: $n terms (r-space)")
        end
    end

    # Determine maximum r-degree across all coefficients
    d_max = 0
    for K in (K0_r, K1_r, K2_r), k in 1:10
        for ((γ, δ, σ, α, β, j), _) in K.equations[k]
            d_max = max(d_max, δ)
        end
    end
    verbose && println("  Max r-degree: $d_max")

    verbose && println("Transforming r → z (d_max=$d_max)...")
    K0_z = _r_to_z(K0_r, rp, d_max)
    K1_z = _r_to_z(K1_r, rp, d_max)
    K2_z = _r_to_z(K2_r, rp, d_max)

    verbose && println("Assembling D̃ matrices (N=$N)...")
    basis = spectral_basis(N, m)
    D0 = assemble_system(K0_z, basis, a).D0
    D1 = assemble_system(K1_z, basis, a).D0
    D2 = assemble_system(K2_z, basis, a).D0

    METRICSSystem(D0, D1, D2, N, m, a)
end
