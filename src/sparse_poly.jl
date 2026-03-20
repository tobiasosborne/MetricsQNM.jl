# Bespoke sparse multivariate polynomial algebra for coefficient extraction.
#
# Replaces the Symbolics.jl pipeline: simplify_fractions → expand → tree-walk
# with a single-pass conversion: Symbolics tree → SparsePoly (Dict{exponent→coeff}).
#
# The key insight: we KNOW the denominator structure is always
#   Σ^p × Δ^q × (1-χ²)^s
# where Σ = r² + a²χ², Δ = r² - 2r + a².  So we track denominator
# powers structurally and clear them by polynomial multiplication —
# no GCD computation needed.
#
# Variables: (r, χ, ω_re, ω_im, iu) — 5 variables, fixed order.

using Symbolics: SymbolicUtils

export SparsePoly, DenomSig, RatPoly
export symexpr_to_poly, clear_denominators

# ═══════════════════════════════════════════════════════════════════════════════
#  SparsePoly: Dict{NTuple{5,Int}, Float64}
# ═══════════════════════════════════════════════════════════════════════════════

const N_VARS = 5
const ExpVec = NTuple{N_VARS, Int}
const ZERO_EXP = ntuple(_ -> 0, N_VARS)

struct SparsePoly
    terms::Dict{ExpVec, Float64}
end

SparsePoly() = SparsePoly(Dict{ExpVec, Float64}())

function SparsePoly(c::Real)
    iszero(c) && return SparsePoly()
    SparsePoly(Dict{ExpVec, Float64}(ZERO_EXP => Float64(c)))
end

function var_poly(i::Int)
    e = ntuple(k -> k == i ? 1 : 0, N_VARS)
    SparsePoly(Dict{ExpVec, Float64}(e => 1.0))
end

Base.iszero(p::SparsePoly) = isempty(p.terms)
Base.length(p::SparsePoly) = length(p.terms)

function Base.:+(a::SparsePoly, b::SparsePoly)
    result = copy(a.terms)
    for (e, c) in b.terms
        result[e] = get(result, e, 0.0) + c
    end
    SparsePoly(result)
end

function Base.:-(a::SparsePoly, b::SparsePoly)
    result = copy(a.terms)
    for (e, c) in b.terms
        result[e] = get(result, e, 0.0) - c
    end
    SparsePoly(result)
end

function Base.:-(p::SparsePoly)
    SparsePoly(Dict{ExpVec, Float64}(e => -c for (e, c) in p.terms))
end

function Base.:*(a::SparsePoly, b::SparsePoly)
    result = Dict{ExpVec, Float64}()
    sizehint!(result, length(a.terms) * length(b.terms))
    for (ea, ca) in a.terms, (eb, cb) in b.terms
        e_new = ea .+ eb
        result[e_new] = get(result, e_new, 0.0) + ca * cb
    end
    SparsePoly(result)
end

function Base.:*(c::Real, p::SparsePoly)
    iszero(c) && return SparsePoly()
    SparsePoly(Dict{ExpVec, Float64}(e => Float64(c) * v for (e, v) in p.terms))
end
Base.:*(p::SparsePoly, c::Real) = c * p

function Base.:^(p::SparsePoly, n::Int)
    n == 0 && return SparsePoly(1.0)
    n == 1 && return p
    n < 0 && error("SparsePoly: negative exponent $n (use RatPoly)")
    # Binary exponentiation
    if iseven(n)
        half = p ^ (n ÷ 2)
        return half * half
    else
        return p * (p ^ (n - 1))
    end
end

function cleanup!(p::SparsePoly; tol=1e-15, relative=false)
    if relative && !isempty(p.terms)
        scale = maximum(abs, values(p.terms))
        scale > 0 && (tol *= scale)
    end
    filter!(kv -> abs(kv.second) > tol, p.terms)
    p
end

# ═══════════════════════════════════════════════════════════════════════════════
#  SparsePoly differentiation
# ═══════════════════════════════════════════════════════════════════════════════

"""
    differentiate(p::SparsePoly, var_idx::Int) → SparsePoly

Exact symbolic differentiation of SparsePoly with respect to variable `var_idx`.
For monomial c × x₁^e₁ × ... × xₖ^eₖ: d/dxᵢ = c·eᵢ × x₁^e₁ × ... × xᵢ^(eᵢ-1) × ...
"""
function differentiate(p::SparsePoly, var_idx::Int)
    result = Dict{ExpVec, Float64}()
    for (e, c) in p.terms
        e[var_idx] == 0 && continue
        new_e = ntuple(i -> i == var_idx ? e[i] - 1 : e[i], N_VARS)
        coeff = c * e[var_idx]
        result[new_e] = get(result, new_e, 0.0) + coeff
    end
    SparsePoly(result)
end

export differentiate

# ═══════════════════════════════════════════════════════════════════════════════
#  DenomSig: tracks known denominator factors structurally
# ═══════════════════════════════════════════════════════════════════════════════

struct DenomSig
    p::Int   # power of Σ = r² + a²χ²
    q::Int   # power of Δ = r² - 2r + a²
    s::Int   # power of (1-χ²)
    t::Int   # power of r
end

DenomSig(p, q, s) = DenomSig(p, q, s, 0)   # backward compat: GR never has r-denom
DenomSig() = DenomSig(0, 0, 0, 0)

function Base.:+(a::DenomSig, b::DenomSig)
    DenomSig(a.p + b.p, a.q + b.q, a.s + b.s, a.t + b.t)
end

function lcd(sigs::AbstractVector{DenomSig})
    DenomSig(maximum(s.p for s in sigs),
             maximum(s.q for s in sigs),
             maximum(s.s for s in sigs),
             maximum(s.t for s in sigs))
end

# ═══════════════════════════════════════════════════════════════════════════════
#  RatPoly: numerator SparsePoly + tracked denominator
# ═══════════════════════════════════════════════════════════════════════════════

struct RatPoly
    num::SparsePoly
    den::DenomSig
end

RatPoly(c::Real) = RatPoly(SparsePoly(c), DenomSig())
RatPoly(p::SparsePoly) = RatPoly(p, DenomSig())

function mul_ratpoly(a::RatPoly, b::RatPoly)
    RatPoly(a.num * b.num, a.den + b.den)
end

function scale_ratpoly(c::Real, rp::RatPoly)
    RatPoly(c * rp.num, rp.den)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Symbolics expression tree → RatPoly conversion
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SymToPolyCtx

Context for the tree-walking conversion.  Holds variable→index map and
precomputed factor polynomials for Σ, Δ, (1-χ²), r.
"""
struct SymToPolyCtx
    var_idx::Dict{Any, Int}       # unwrapped Sym → variable index (1..5)
    Σ_poly::SparsePoly            # r² + a²χ²
    Δ_poly::SparsePoly            # r² - 2r + a²
    s2_poly::SparsePoly           # 1 - χ²
    r_poly::SparsePoly            # r (for clearing r-denominators in sGB)
    # Fingerprints for identifying denom factors in the tree
    Σ_hash::UInt
    Δ_hash::UInt
    s2_hash::UInt
end

"""
    SymToPolyCtx(var_syms, a_val)

Build conversion context.  `var_syms` = [r, χ, ω_re, ω_im, iu] as Symbolics.Num.
"""
function SymToPolyCtx(var_syms::Vector, a_val::Float64)
    var_idx = Dict{Any,Int}()
    for (i, v) in enumerate(var_syms)
        var_idx[Symbolics.unwrap(v)] = i
    end

    r  = var_poly(1)  # r
    χ  = var_poly(2)  # χ
    a2 = a_val^2

    Σ_poly  = r^2 + a2 * (χ^2)                    # r² + a²χ²
    Δ_poly  = r^2 + (-2.0) * r + SparsePoly(a2)   # r² - 2r + a²
    s2_poly = SparsePoly(1.0) - χ^2                # 1 - χ²
    r_poly  = r                                     # r (for sGB r-denominator clearing)

    # Hash the unwrapped Symbolics expressions for these factors
    # so we can identify them in Div denominators
    Σ_hash  = UInt(0)  # we'll match structurally instead
    Δ_hash  = UInt(0)
    s2_hash = UInt(0)

    SymToPolyCtx(var_idx, Σ_poly, Δ_poly, s2_poly, r_poly, Σ_hash, Δ_hash, s2_hash)
end

"""
Clear denominators in `rp` to reach `target` DenomSig by multiplying
numerator by the appropriate factor polynomials.
"""
function clear_to(rp::RatPoly, target::DenomSig, ctx::SymToPolyCtx)
    num = rp.num
    dp = target.p - rp.den.p
    dq = target.q - rp.den.q
    ds = target.s - rp.den.s
    dt = target.t - rp.den.t
    dp < 0 && error("Cannot clear Σ: target.p=$(target.p) < current=$(rp.den.p)")
    dq < 0 && error("Cannot clear Δ: target.q=$(target.q) < current=$(rp.den.q)")
    ds < 0 && error("Cannot clear s2: target.s=$(target.s) < current=$(rp.den.s)")
    dt < 0 && error("Cannot clear r: target.t=$(target.t) < current=$(rp.den.t)")
    dp > 0 && (num = num * (ctx.Σ_poly ^ dp))
    dq > 0 && (num = num * (ctx.Δ_poly ^ dq))
    ds > 0 && (num = num * (ctx.s2_poly ^ ds))
    dt > 0 && (num = num * (ctx.r_poly ^ dt))
    num
end

"""
Add a vector of RatPolys by finding LCD and clearing.
"""
function add_ratpolys(rps::AbstractVector{RatPoly}, ctx::SymToPolyCtx)
    isempty(rps) && return RatPoly(SparsePoly(), DenomSig())
    target = lcd([rp.den for rp in rps])
    result = SparsePoly()
    for rp in rps
        result = result + clear_to(rp, target, ctx)
    end
    RatPoly(result, target)
end

"""
    symexpr_to_poly(expr_num::Num, ctx::SymToPolyCtx) → RatPoly

Convert a Symbolics expression (after substituting a=numerical value)
to a RatPoly.  Single recursive pass — no expand, no simplify_fractions.
"""
function symexpr_to_poly(expr_num::Num, ctx::SymToPolyCtx)
    _convert(Symbolics.unwrap(expr_num), ctx)
end

function _convert(expr, ctx::SymToPolyCtx)::RatPoly
    # ── Number ────────────────────────────────────────────────────────────
    if expr isa Number
        return RatPoly(Float64(real(expr)))
    end

    # ── Symbol (variable) ────────────────────────────────────────────────
    if SymbolicUtils.issym(expr)
        idx = get(ctx.var_idx, expr, 0)
        if idx > 0
            return RatPoly(var_poly(idx))
        else
            # Unknown symbol — should not happen after substituting a
            error("symexpr_to_poly: unknown symbol '$(expr)'. Did you forget to substitute a parameter?")
        end
    end

    # ── Division ─────────────────────────────────────────────────────────
    if SymbolicUtils.isdiv(expr)
        args = SymbolicUtils.arguments(expr)
        num_rp = _convert(args[1], ctx)
        den_rp = _convert(args[2], ctx)
        return _divide(num_rp, den_rp, ctx)
    end

    # ── Addition ─────────────────────────────────────────────────────────
    if SymbolicUtils.isadd(expr)
        terms = RatPoly[]
        # Constant term
        if !iszero(expr.coeff)
            push!(terms, RatPoly(Float64(expr.coeff)))
        end
        for (term, mult) in expr.dict
            rp = _convert(term, ctx)
            if !isone(mult)
                rp = scale_ratpoly(Float64(mult), rp)
            end
            push!(terms, rp)
        end
        isempty(terms) && return RatPoly(0.0)
        length(terms) == 1 && return terms[1]
        return add_ratpolys(terms, ctx)
    end

    # ── Multiplication ───────────────────────────────────────────────────
    if SymbolicUtils.ismul(expr)
        result = RatPoly(Float64(expr.coeff))
        for (base, exp_val) in expr.dict
            factor = _convert(base, ctx)
            ev = _to_int(exp_val)
            if ev != 1
                factor = _pow_ratpoly(factor, ev, ctx)
            end
            result = mul_ratpoly(result, factor)
        end
        return result
    end

    # ── Power ────────────────────────────────────────────────────────────
    if SymbolicUtils.ispow(expr)
        args = SymbolicUtils.arguments(expr)
        base_rp = _convert(args[1], ctx)
        exp_val = args[2]
        return _pow_ratpoly(base_rp, exp_val, ctx)
    end

    # ── Fallback: try numeric conversion ────────────────────────────────
    # SymbolicUtils may wrap plain integers as symbolic constants
    try
        return RatPoly(Float64(expr))
    catch; end
    try
        v = Symbolics.value(Num(expr))
        v isa Number && return RatPoly(Float64(v))
    catch; end
    # Check if it's a SymbolicUtils constant integer
    if hasproperty(expr, :val) || hasfield(typeof(expr), :val)
        try return RatPoly(Float64(expr.val)) catch; end
    end

    error("symexpr_to_poly: unhandled node type $(typeof(expr)): $expr")
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Division: identify known denom factors or fall back to general poly division
# ═══════════════════════════════════════════════════════════════════════════════

function _divide(num_rp::RatPoly, den_rp::RatPoly, ctx::SymToPolyCtx)::RatPoly
    # If den_rp itself is rational (has tracked denom factors), then:
    #   num / (den_num / den_denom) = num * den_denom / den_num
    # The den_denom factors move to the numerator side.
    if !iszero(den_rp.den)
        # Multiply numerator by the denominator's known factors
        num_cleared = clear_to(num_rp, num_rp.den + den_rp.den, ctx)
        # Now divide by den_rp.num (which is a pure polynomial)
        return _divide(RatPoly(num_cleared, num_rp.den + den_rp.den),
                       RatPoly(den_rp.num, DenomSig()), ctx)
    end

    den_poly = den_rp.num

    # Try to identify the denominator as a known factor or power thereof
    result_id = _identify_denom(den_poly, ctx)
    if result_id !== nothing
        # Known factor — just track it (with possible scalar)
        return RatPoly(num_rp.num, num_rp.den + result_id)
    end

    # Try to factor as product of known factors (with scalar)
    result_f = _factor_known_denoms(den_poly, ctx)
    if result_f !== nothing
        sig_product, scalar = result_f
        # Divide numerator by the scalar factor
        scaled_num = (1.0 / scalar) * num_rp.num
        return RatPoly(scaled_num, num_rp.den + sig_product)
    end

    # Unknown denominator — dump for debugging
    sorted = sort(collect(den_poly.terms), by=kv->sum(kv[1]), rev=true)
    terms_str = join(["r^$(e[1])χ^$(e[2])ωr^$(e[3])ωi^$(e[4])iu^$(e[5])=$(round(c,sigdigits=4))"
                      for (e,c) in sorted[1:min(5,end)]], ", ")
    error("symexpr_to_poly: cannot identify denominator as c×Σ^p×Δ^q×(1-χ²)^s×r^t.\n  den has $(length(den_poly.terms)) terms: $terms_str...")
end

"""
Try to identify `poly` as a power of one of the known factors Σ, Δ, (1-χ²), r.
Returns DenomSig or nothing.
"""
function _identify_denom(poly::SparsePoly, ctx::SymToPolyCtx)
    for fac in [:Σ, :Δ, :s2, :r]
        base = fac == :Σ ? ctx.Σ_poly : fac == :Δ ? ctx.Δ_poly :
               fac == :s2 ? ctx.s2_poly : ctx.r_poly
        power = _match_power(poly, base)
        if power !== nothing
            return fac == :Σ  ? DenomSig(power, 0, 0, 0) :
                   fac == :Δ  ? DenomSig(0, power, 0, 0) :
                   fac == :s2 ? DenomSig(0, 0, power, 0) :
                                DenomSig(0, 0, 0, power)
        end
    end
    return nothing
end

"""
Check if `poly` equals `base^n` for some positive integer n.
Returns n or nothing.
"""
function _match_power(poly::SparsePoly, base::SparsePoly)
    # Quick check: if poly has same terms as base, it's base^1
    if _polys_equal(poly, base)
        return 1
    end
    # Try higher powers up to a reasonable limit
    power = base
    for n in 2:8
        power = power * base
        cleanup!(power)
        if _polys_equal(poly, power)
            return n
        end
    end
    return nothing
end

function _polys_equal(a::SparsePoly, b::SparsePoly; tol=1e-12)
    length(a.terms) != length(b.terms) && return false
    for (e, ca) in a.terms
        cb = get(b.terms, e, 0.0)
        abs(ca - cb) > tol * max(1.0, abs(ca), abs(cb)) && return false
    end
    return true
end

"""
Try to factor `poly` as `c × Σ^p × Δ^q × (1-χ²)^s × r^t` for scalar c and known factors.
Greedy: repeatedly divide by each factor, trying all orderings.
Returns (DenomSig, scalar_coeff) or nothing.
"""
function _factor_known_denoms(poly::SparsePoly, ctx::SymToPolyCtx)
    factors = [(:Σ, ctx.Σ_poly), (:Δ, ctx.Δ_poly), (:s2, ctx.s2_poly), (:r, ctx.r_poly)]

    # Tolerance relative to input polynomial scale (not remainder scale)
    poly_scale = isempty(poly.terms) ? 1.0 : maximum(abs, values(poly.terms))
    zero_tol = 1e-10 * poly_scale

    # Try all permutations of factor order (4! = 24 permutations)
    perms = [[i,j,k,l] for i in 1:4 for j in 1:4 for k in 1:4 for l in 1:4
             if length(Set([i,j,k,l])) == 4]

    for perm in perms
        remainder = poly
        sig_p, sig_q, sig_s, sig_t = 0, 0, 0, 0

        for idx in perm
            fac_sym, fac_poly = factors[idx]
            while true
                q, r = _poly_divmod(remainder, fac_poly)
                if q !== nothing
                    cleanup!(r; tol=zero_tol)
                    if iszero(r)
                        cleanup!(q)
                        remainder = q
                        if fac_sym == :Σ; sig_p += 1
                        elseif fac_sym == :Δ; sig_q += 1
                        elseif fac_sym == :s2; sig_s += 1
                        else sig_t += 1; end
                    else
                        break
                    end
                else
                    break
                end
            end
        end

        # Check if fully factored (scalar remainder)
        cleanup!(remainder; tol=zero_tol)
        if length(remainder.terms) <= 1
            if isempty(remainder.terms)
                return (DenomSig(sig_p, sig_q, sig_s, sig_t), 1.0)
            end
            e, c = first(remainder.terms)
            if all(==(0), e) && abs(c) > 1e-15
                return (DenomSig(sig_p, sig_q, sig_s, sig_t), c)
            end
        end
    end

    return nothing
end

"""
Attempt polynomial division: a = q * b + r.
Returns (q, r) or (nothing, a) if b doesn't divide a cleanly.

Uses multivariate polynomial long division with graded lex order.
"""
function _poly_divmod(a::SparsePoly, b::SparsePoly)
    isempty(b.terms) && error("division by zero polynomial")

    # Find leading term of b (highest total degree, then lex)
    lt_b_exp, lt_b_coeff = _leading_term(b)

    # Scale-aware tolerance: just above machine epsilon × input scale
    # Must not drop legitimate small coefficients (e.g. a=0.1 → a^14 ≈ 1e-14)
    scale_a = isempty(a.terms) ? 0.0 : maximum(abs, values(a.terms))
    scale_b = maximum(abs, values(b.terms))
    ε = 1e-15 * max(1.0, scale_a, scale_b)

    q = SparsePoly()
    r = SparsePoly()
    remainder = copy(a.terms)

    max_iter = length(a.terms) * 10  # safety bound
    iter = 0
    while !isempty(remainder) && iter < max_iter
        iter += 1
        # Find leading term of remainder
        lt_r_exp, lt_r_coeff = _leading_term_dict(remainder)

        # Can lt_b divide lt_r?
        diff = lt_r_exp .- lt_b_exp
        if all(>=(0), diff)
            # Yes — compute quotient monomial
            q_coeff = lt_r_coeff / lt_b_coeff
            # Subtract q_mono * b from remainder
            for (eb, cb) in b.terms
                e_sub = diff .+ eb
                remainder[e_sub] = get(remainder, e_sub, 0.0) - q_coeff * cb
                abs(remainder[e_sub]) < ε && delete!(remainder, e_sub)
            end
            q.terms[diff] = get(q.terms, diff, 0.0) + q_coeff
        else
            # Can't divide — move this term to remainder
            r.terms[lt_r_exp] = lt_r_coeff
            delete!(remainder, lt_r_exp)
        end
    end

    # Any remaining terms go to r
    for (e, c) in remainder
        r.terms[e] = get(r.terms, e, 0.0) + c
    end

    cleanup!(q)
    cleanup!(r)
    return (q, r)
end

function _leading_term(p::SparsePoly)
    _leading_term_dict(p.terms)
end

function _leading_term_dict(terms::Dict{ExpVec, Float64})
    best_exp = ZERO_EXP
    best_coeff = 0.0
    best_deg = -1
    for (e, c) in terms
        abs(c) < 1e-15 && continue
        d = sum(e)
        if d > best_deg || (d == best_deg && e > best_exp)
            best_exp = e
            best_coeff = c
            best_deg = d
        end
    end
    return best_exp, best_coeff
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Power of RatPoly
# ═══════════════════════════════════════════════════════════════════════════════

function _to_int(n)::Int
    n isa Int && return n
    n isa Integer && return Int(n)
    n isa Number && return Int(n)
    # SymbolicUtils wrapped integer — try Symbolics.value or direct conversion
    try return Int(Symbolics.value(Num(n))) catch; end
    try return Int(Float64(Symbolics.value(Num(n)))) catch; end
    # Last resort: check if it's a literal wrapped in symbolic
    if SymbolicUtils.issym(n) || SymbolicUtils.ispow(n) || SymbolicUtils.ismul(n)
        error("non-numeric exponent: $n ($(typeof(n)))")
    end
    # Try the raw value
    try return Int(n.val) catch; end
    error("cannot convert exponent to Int: $n ($(typeof(n)))")
end

function _pow_ratpoly(rp::RatPoly, n, ctx::SymToPolyCtx)::RatPoly
    n_int = _to_int(n)
    if n_int >= 0
        return RatPoly(rp.num ^ n_int, DenomSig(rp.den.p * n_int, rp.den.q * n_int,
                                                  rp.den.s * n_int, rp.den.t * n_int))
    else
        # Negative power: this goes into the denominator
        # num^(-k) / den^(-k) = den^k / num^k
        pos = -n_int
        # The numerator poly becomes a denominator — identify it
        den_poly = rp.num ^ pos
        cleanup!(den_poly)
        sig = _identify_denom(den_poly, ctx)
        if sig !== nothing
            # The denominator of the original is now in the numerator
            new_num = !iszero(rp.den) ?
                clear_to(RatPoly(SparsePoly(1.0), DenomSig()),
                         DenomSig(rp.den.p * pos, rp.den.q * pos,
                                  rp.den.s * pos, rp.den.t * pos),
                         ctx) :
                SparsePoly(1.0)
            return RatPoly(new_num, sig)
        end

        # Try factoring
        result_f = _factor_known_denoms(den_poly, ctx)
        if result_f !== nothing
            sig_f, scalar_f = result_f
            new_num2 = !iszero(rp.den) ?
                clear_to(RatPoly(SparsePoly(1.0 / scalar_f), DenomSig()),
                         DenomSig(rp.den.p * pos, rp.den.q * pos,
                                  rp.den.s * pos, rp.den.t * pos),
                         ctx) :
                SparsePoly(1.0 / scalar_f)
            return RatPoly(new_num2, sig_f)
        end

        error("symexpr_to_poly: negative power of unrecognized polynomial")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Top-level convenience: expression → cleared polynomial coefficients
# ═══════════════════════════════════════════════════════════════════════════════

"""
    reduce_ratpoly(rp::RatPoly, ctx) → RatPoly

Try to cancel common known factors between numerator and denominator.
If the numerator is divisible by Σ (or Δ, or s2), divide and reduce
the denominator power accordingly.
"""
function reduce_ratpoly(rp::RatPoly, ctx::SymToPolyCtx)
    dp, dq, ds, dt = rp.den.p, rp.den.q, rp.den.s, rp.den.t
    (dp == 0 && dq == 0 && ds == 0 && dt == 0) && return rp

    factors = [(ctx.Σ_poly, 1), (ctx.Δ_poly, 2), (ctx.s2_poly, 3), (ctx.r_poly, 4)]
    best_num = rp.num
    best_pqst = [dp, dq, ds, dt]
    best_total = dp + dq + ds + dt

    # Tolerance relative to numerator scale (not remainder scale)
    num_scale = isempty(rp.num.terms) ? 1.0 : maximum(abs, values(rp.num.terms))
    zero_tol = 1e-10 * num_scale

    # Try all permutations of factor ordering (division is order-dependent)
    perms = [[i,j,k,l] for i in 1:4 for j in 1:4 for k in 1:4 for l in 1:4
             if length(Set([i,j,k,l])) == 4]

    for perm in perms
        trial_num = rp.num
        trial_pqst = [dp, dq, ds, dt]

        for idx in perm
            fac_poly, which = factors[idx]
            while trial_pqst[which] > 0
                q, r = _poly_divmod(trial_num, fac_poly)
                cleanup!(r; tol=zero_tol)
                if q !== nothing && iszero(r)
                    cleanup!(q)
                    trial_num = q
                    trial_pqst[which] -= 1
                else
                    break
                end
            end
        end

        trial_total = sum(trial_pqst)
        if trial_total < best_total
            best_num = trial_num
            best_pqst = copy(trial_pqst)
            best_total = trial_total
        end
        best_total == 0 && break  # fully reduced
    end

    RatPoly(best_num, DenomSig(best_pqst[1], best_pqst[2], best_pqst[3], best_pqst[4]))
end

"""
    clear_denominators(rp::RatPoly, P, Q, S, ctx; T=0) → (SparsePoly, DenomSig)

Given a RatPoly with tracked denominator Σ^p Δ^q (1-χ²)^s r^t,
first reduce (cancel common factors), then multiply numerator
by enough factors to clear all remaining denominators.

Returns (cleared_polynomial, actual_clearing_powers).
T defaults to 0 (auto-clear whatever r-power is present).
"""
function clear_denominators(rp::RatPoly, P::Int, Q::Int, S::Int, ctx::SymToPolyCtx;
                            T::Int=0)
    # First reduce: cancel num/denom common factors
    rp_reduced = reduce_ratpoly(rp, ctx)

    # Use the larger of requested and remaining denominator powers
    actual = DenomSig(max(P, rp_reduced.den.p),
                      max(Q, rp_reduced.den.q),
                      max(S, rp_reduced.den.s),
                      max(T, rp_reduced.den.t))
    result = clear_to(rp_reduced, actual, ctx)
    cleanup!(result)
    return result, actual
end

"""
    symexpr_to_cleared_poly(expr_num, ctx, P, Q, S) → Dict{NTuple{5,Int}, Float64}

One-shot: Symbolics expression → cleared polynomial coefficient dict.
This replaces the entire simplify_fractions + expand + _walk_expanded_poly pipeline.
"""
function symexpr_to_cleared_poly(expr_num::Num, ctx::SymToPolyCtx,
                                  P::Int, Q::Int, S::Int)
    rp = symexpr_to_poly(expr_num, ctx)
    poly, actual = clear_denominators(rp, P, Q, S, ctx)
    return poly.terms, actual
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers for DenomSig
# ═══════════════════════════════════════════════════════════════════════════════

Base.iszero(d::DenomSig) = d.p == 0 && d.q == 0 && d.s == 0 && d.t == 0
