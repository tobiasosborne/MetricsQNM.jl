# sGB background data: Mathematica BoxData parser + evaluation.
# Parses supplementary materials from arxiv:2406.11986 to extract
# H₁-H₄(r,χ,a), ϑ(r,χ,a), Ω_H⁽¹⁾(a), κ⁽¹⁾(a).

export SGBBackground, sgb_background, sgb_H_derivs, sgb_H_params

# ═══════════════════════════════════════════════════════════════════════════════
#  Mathematica expression parser (recursive descent)
# ═══════════════════════════════════════════════════════════════════════════════

function _nb_skip_ws(s::String, i::Int)
    n = ncodeunits(s)
    @inbounds while i <= n
        c = s[i]
        if c == ' ' || c == '\t' || c == '\n' || c == '\r'
            i += 1
        elseif c == '\\' && i + 1 <= n && (s[i+1] == '\n' || s[i+1] == '\r')
            # Line continuation: backslash + newline
            i += 2
        else
            break
        end
    end
    i
end

"""
    _nb_parse(s, i) → (ast, new_i)

Parse one Mathematica expression from byte position `i`.
Returns nested Julia data:
  - String: atoms (identifiers, string contents, numbers)
  - Vector{Any}: lists `{...}`
  - Tuple{String, Vector{Any}}: function calls `f[...]`
"""
function _nb_parse(s::String, i::Int)
    i = _nb_skip_ws(s, i)
    n = ncodeunits(s)
    i > n && error("Unexpected end at $i")
    c = s[i]

    if c == '"'
        # String literal — scan to closing quote, skip \-escapes
        j = i + 1
        while j <= n
            s[j] == '\\' && (j += 2; continue)
            s[j] == '"' && break
            j += 1
        end
        # Strip embedded line continuations (\+newline+spaces)
        raw = s[i+1:j-1]
        raw = replace(raw, r"\\\n\s*" => "")
        return raw, j + 1

    elseif c == '{'
        # List
        elems = Any[]
        j = i + 1
        while true
            j = _nb_skip_ws(s, j)
            j > n && error("Unterminated list at $i")
            s[j] == '}' && return elems, j + 1
            s[j] == ',' && (j += 1; continue)
            j_before = j
            e, j = _nb_parse(s, j)
            j == j_before && (j += 1; continue)  # safety: skip stuck chars
            # Skip Rule (->)
            jj = _nb_skip_ws(s, j)
            if jj + 1 <= n && s[jj] == '-' && s[jj+1] == '>'
                _, j = _nb_parse(s, jj + 2)
                continue
            end
            e isa String && isempty(e) && continue  # skip empty atoms
            push!(elems, e)
        end

    elseif isdigit(c) || c == '.'
        j = i
        while j <= n
            if isdigit(s[j]) || s[j] == '.'
                j += 1
            elseif s[j] == '\\' && j + 1 <= n && (s[j+1] == '\n' || s[j+1] == '\r')
                j += 2  # skip line continuation within numbers
                while j <= n && (s[j] == '\n' || s[j] == '\r' || s[j] == ' '); j += 1; end
            else
                break
            end
        end
        # Mathematica scientific: *^N
        if j + 1 <= n && s[j] == '*' && s[j+1] == '^'
            j += 2
            (j <= n && s[j] == '-') && (j += 1)
            while j <= n && isdigit(s[j]); j += 1; end
        end
        # Build number string, stripping embedded whitespace/continuations
        num = replace(s[i:j-1], r"\\?\s+" => "")
        return num, j

    elseif c == '-'
        # Negative number or standalone minus
        if i + 1 <= n && (isdigit(s[i+1]) || s[i+1] == '.')
            j = i + 1
            while j <= n && (isdigit(s[j]) || s[j] == '.'); j += 1; end
            return s[i:j-1], j
        else
            return "-", i + 1
        end

    elseif isletter(c) || c == '$' || c == '\\'
        j = i
        if c == '\\' && i + 1 <= n && s[i+1] == '['
            close = findnext(']', s, i + 2)
            close === nothing && error("Unterminated \\[...] at $i")
            j = close + 1
        elseif c == '\\'
            # Bare backslash not followed by [ — skip it
            return "", i + 1
        else
            while j <= n && (isletter(s[j]) || isdigit(s[j]) || s[j] == '_' || s[j] == '$')
                j += 1
            end
        end
        name = s[i:j-1]
        jj = _nb_skip_ws(s, j)
        if jj <= n && s[jj] == '['
            # Function call
            args = Any[]
            j = jj + 1
            while true
                j = _nb_skip_ws(s, j)
                j > n && error("Unterminated call to $name at $i")
                s[j] == ']' && return (name, args), j + 1
                s[j] == ',' && (j += 1; continue)
                j_before = j
                a, j = _nb_parse(s, j)
                j == j_before && (j += 1; continue)  # safety: skip stuck chars
                jjj = _nb_skip_ws(s, j)
                if jjj + 1 <= n && s[jjj] == '-' && s[jjj+1] == '>'
                    _, j = _nb_parse(s, jjj + 2)
                    continue
                end
                a isa String && isempty(a) && continue  # skip empty atoms
                push!(args, a)
            end
        else
            return name, j
        end
    else
        return string(c), i + 1
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  BoxData AST → Julia expression string
# ═══════════════════════════════════════════════════════════════════════════════

"""Convert integer string to safe Julia literal (avoid Int64 overflow for large numbers)."""
function _nb_numstr(s::String)::String
    # Strip line continuations and whitespace that may have leaked through
    s = replace(s, r"\\?\s+" => "")
    isempty(s) && return "0"
    neg = startswith(s, '-')
    digits = neg ? s[2:end] : s
    '.' in digits && return s
    length(digits) <= 15 && return s
    return string(parse(Float64, s))
end

"""Convert parsed BoxData AST to a Julia expression string."""
function _box_to_str(ast)::String
    if ast isa String
        s = ast
        # Operators — handled in _rowbox_to_str, but as fallback:
        s == " " && return " * "
        s == "+" && return " + "
        s == "-" && return " - "
        s == "(" && return "("
        s == ")" && return ")"
        s in (";", "=") && return ""
        # Variables
        s == "\\[Chi]" && return "_χ_"
        s == "\\[Alpha]" && return "_α_"
        s == "M" && return "_M_"
        s == "r" && return "_r_"
        s == "a" && return "_a_"
        # LHS names (skip — we only convert RHS)
        s in ("\\[Phi]", "\\[CapitalPhi]", "\\[CapitalOmega]", "\\[Kappa]",
              "H1", "H2", "H3", "H4") && return ""
        # Number
        if !isempty(s) && (isdigit(s[1]) || (length(s) > 1 && s[1] == '-' && isdigit(s[2])))
            return _nb_numstr(s)
        end
        return s

    elseif ast isa Tuple
        name::String, args::Vector = ast
        name == "RowBox"        && return _rowbox_to_str(args[1]::Vector)
        name == "FractionBox"   && return "(($(  _box_to_str(args[1]))) / ($(_box_to_str(args[2]))))"
        name == "SuperscriptBox" && return "(($(_box_to_str(args[1])))^($(_box_to_str(args[2]))))"
        name in ("FormBox", "StyleBox", "BoxData", "TextData") && return _box_to_str(args[1])
        name == "SubscriptBox" && return ""
        !isempty(args) && return _box_to_str(args[1])
        return ""

    elseif ast isa Vector
        return join(_box_to_str.(ast))
    end
    return ""
end

"""Convert a RowBox element list to Julia expression string with implicit multiplication."""
function _rowbox_to_str(elems::Vector)::String
    tokens = Tuple{Symbol, String}[]
    for e in elems
        if e isa String
            e == " "  && (push!(tokens, (:mul, " * ")); continue)
            e == "+"  && (push!(tokens, (:add, " + ")); continue)
            e == "-"  && (push!(tokens, (:add, " - ")); continue)
            e == "("  && (push!(tokens, (:lp,  "("));   continue)
            e == ")"  && (push!(tokens, (:rp,  ")"));   continue)
            e in (";", "=") && continue
        end
        s = _box_to_str(e)
        isempty(s) || push!(tokens, (:val, s))
    end
    # Insert implicit multiplication: val*val, val*(, )*val, )*(
    parts = String[]
    for (i, (typ, s)) in enumerate(tokens)
        if i > 1
            pt = tokens[i-1][1]
            if (pt == :val && typ in (:val, :lp)) ||
               (pt == :rp  && typ in (:val, :lp))
                push!(parts, " * ")
            end
        end
        push!(parts, s)
    end
    join(parts)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Navigate AST to find expression after "="
# ═══════════════════════════════════════════════════════════════════════════════

function _find_rhs(ast)
    if ast isa Tuple
        name, args = ast
        if name == "RowBox" && !isempty(args) && args[1] isa Vector
            elems = args[1]::Vector
            for i in eachindex(elems)
                if elems[i] isa String && elems[i] == "=" && i < length(elems)
                    for j in (i+1):length(elems)
                        elems[j] isa String && elems[j] == ";" && continue
                        return elems[j]
                    end
                end
            end
            for elem in elems
                r = _find_rhs(elem)
                r !== nothing && return r
            end
        elseif name in ("Cell", "BoxData", "FormBox", "StyleBox")
            for arg in args
                r = _find_rhs(arg)
                r !== nothing && return r
            end
        end
    elseif ast isa Vector
        for elem in ast
            r = _find_rhs(elem)
            r !== nothing && return r
        end
    end
    nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Notebook section extraction
# ═══════════════════════════════════════════════════════════════════════════════

const _NB_FILE = joinpath(@__DIR__, "..", "reference", "2406.11986_source",
                          "Supplementary_materials.nb")

# Search patterns: variable name → string to search in notebook
const _NB_SECTIONS = [
    ("Phi",      "\\[Phi]"),
    ("H1",       "\"H1\""),
    ("H2",       "\"H2\""),
    ("H3",       "\"H3\""),
    ("H4",       "\"H4\""),
    ("OmegaH1",  "\\[CapitalOmega]"),
    ("kappa1",   "\\[Kappa]"),
]

"""Find the byte position of `Cell[BoxData[` before position `pos`."""
function _find_cell_start(text::String, pos::Int)
    target = "Cell[BoxData["
    for i in pos:-1:1
        if i + length(target) - 1 <= ncodeunits(text) &&
           text[i:i+length(target)-1] == target
            return i
        end
    end
    error("Cannot find Cell[BoxData[ before position $pos")
end

"""
Find the Input cell containing a given pattern AND an "=" sign.
Skips Section header cells (which don't contain "=").
"""
function _find_input_cell(text::String, pattern::String; from::Int=1)
    pos = findnext(pattern, text, from)
    pos === nothing && error("Pattern '$pattern' not found in notebook")
    start_byte = first(pos)

    cell_start = _find_cell_start(text, start_byte)

    # Find end of this cell (next "Cell[" or end of file)
    next_cell = findnext("Cell[", text, cell_start + 5)
    cell_end = next_cell === nothing ? length(text) : first(next_cell)

    # Check if "=" exists within THIS cell
    eq_pos = findnext("\"=\"", text, cell_start)
    if eq_pos !== nothing && first(eq_pos) < cell_end
        return cell_start
    end
    # No "=" in this cell → skip to next occurrence
    return _find_input_cell(text, pattern; from=last(pos)+1)
end

"""Load and parse all sGB background sections from the Mathematica notebook."""
function _load_nb_sections(; verbose::Bool=false)
    isfile(_NB_FILE) || error("Notebook not found: $_NB_FILE\n" *
        "Place the supplementary materials from arxiv:2406.11986 at this path.")
    verbose && (println("Reading notebook..."); flush(stdout))
    text = read(_NB_FILE, String)

    sections = Dict{String, Any}()
    for (name, pattern) in _NB_SECTIONS
        verbose && (print("  Parsing $name... "); flush(stdout))
        t0 = time()

        cell_start = _find_input_cell(text, pattern)
        ast, _ = _nb_parse(text, cell_start)
        rhs = _find_rhs(ast)
        rhs === nothing && error("Could not find '=' in section $name")
        sections[name] = rhs

        verbose && (@printf("%.2fs\n", time() - t0); flush(stdout))
    end
    sections
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Compile AST to Julia function
# ═══════════════════════════════════════════════════════════════════════════════

"""Compile a BoxData AST to a Julia function (_r_, _χ_, _a_, _M_, _α_) → Float64."""
function _ast_to_func(ast)
    expr_str = _box_to_str(ast)
    expr_str = strip(expr_str)
    isempty(expr_str) && error("Empty expression after BoxData conversion")
    body = Meta.parse(expr_str)
    @eval (_r_::Number, _χ_::Number, _a_::Number, _M_::Number, _α_::Number) -> Float64($body)
end

"""Compile a BoxData AST to a scalar function (_a_, _M_, _α_) → Float64."""
function _ast_to_scalar_func(ast)
    expr_str = _box_to_str(ast)
    expr_str = strip(expr_str)
    body = Meta.parse(expr_str)
    @eval (_a_::Number, _M_::Number, _α_::Number) -> Float64($body)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Cache + data container
# ═══════════════════════════════════════════════════════════════════════════════

struct _SGBData
    H_funcs::NTuple{4, Function}
    Phi_func::Function
    OmegaH1_func::Function
    kappa1_func::Function
end

const _SGB_CACHE = Ref{Union{Nothing, _SGBData}}(nothing)

function _load_sgb_data(; verbose::Bool=false)
    _SGB_CACHE[] !== nothing && return _SGB_CACHE[]

    verbose && (println("Loading sGB background data..."); flush(stdout))
    t0 = time()
    sections = _load_nb_sections(; verbose)

    verbose && (print("  Compiling functions... "); flush(stdout))
    tc = time()
    H_funcs = ntuple(i -> _ast_to_func(sections["H$(i)"]), 4)
    Phi_func = _ast_to_func(sections["Phi"])
    OmegaH1_func = _ast_to_scalar_func(sections["OmegaH1"])
    kappa1_func = _ast_to_scalar_func(sections["kappa1"])
    verbose && (@printf("%.2fs\n", time() - tc); flush(stdout))

    data = _SGBData(H_funcs, Phi_func, OmegaH1_func, kappa1_func)
    _SGB_CACHE[] = data
    verbose && @printf("  Total load: %.1fs\n", time() - t0)
    data
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

"""
    SGBBackground

sGB background correction data evaluated at dimensionless spin `a` with M=1.
"""
struct SGBBackground
    H::NTuple{4, Function}    # H[i](r, χ) → Float64
    vartheta::Function         # ϑ(r, χ) → Float64
    Omega_H1::Float64          # Ω_H⁽¹⁾ at this spin
    kappa1::Float64            # κ⁽¹⁾ at this spin
    a::Float64
end

"""
    sgb_background(a; verbose=false) → SGBBackground

Load sGB background corrections from the Mathematica supplementary materials
of arxiv:2406.11986, evaluated at spin `a` with M=1, α=1.

The α factor is absorbed into the coupling constant ζ = α²/M⁴.
H₁-H₄ and ϑ are returned as closures `(r, χ) → Float64`.
"""
function sgb_background(a::Float64; verbose::Bool=false)
    data = _load_sgb_data(; verbose)

    H = ntuple(4) do i
        f = data.H_funcs[i]
        (r, χ) -> Base.invokelatest(f, r, χ, a, 1.0, 1.0)
    end

    vartheta = let f = data.Phi_func
        (r, χ) -> Base.invokelatest(f, r, χ, a, 1.0, 1.0)
    end

    Omega_H1 = Base.invokelatest(data.OmegaH1_func, a, 1.0, 1.0)
    kappa1 = Base.invokelatest(data.kappa1_func, a, 1.0, 1.0)

    verbose && @printf("  sGB background at a=%.4f: Ω_H⁽¹⁾=%.6e, κ⁽¹⁾=%.6e\n",
                       a, Omega_H1, kappa1)

    SGBBackground(H, vartheta, Omega_H1, kappa1, a)
end

"""
    sgb_H_derivs(bg, i, r, χ; ε=1e-5)

Evaluate H_i and all its derivatives at (r, χ) via central finite differences.

Returns a NamedTuple (val, dr, dchi, drr, dchichi, drchi) matching the
ordering of the abstract H_i parameters in sgb_linearize.jl.
"""
function sgb_H_derivs(bg::SGBBackground, i::Int, r::Float64, χ::Float64;
                       ε::Float64=1e-5)
    f = bg.H[i]
    val = f(r, χ)
    # First derivatives (central difference)
    dr   = (f(r + ε, χ) - f(r - ε, χ)) / (2ε)
    dchi = (f(r, χ + ε) - f(r, χ - ε)) / (2ε)
    # Second derivatives
    drr     = (f(r + ε, χ) - 2val + f(r - ε, χ)) / ε^2
    dchichi = (f(r, χ + ε) - 2val + f(r, χ - ε)) / ε^2
    drchi   = (f(r + ε, χ + ε) - f(r + ε, χ - ε) - f(r - ε, χ + ε) + f(r - ε, χ - ε)) / (4ε^2)
    return (val=val, dr=dr, dchi=dchi, drr=drr, dchichi=dchichi, drchi=drchi)
end

"""
    sgb_H_params(bg, r, χ; ε=1e-5)

Evaluate all 24 H_i parameters at (r, χ) for the compiled sGB correction evaluator.

Returns a length-24 Float64 vector in the order:
    [H1, H1_r, H1_χ, H1_rr, H1_χχ, H1_rχ, H2, ..., H4_rχ]
"""
function sgb_H_params(bg::SGBBackground, r::Float64, χ::Float64;
                       ε::Float64=1e-5)
    params = Vector{Float64}(undef, 24)
    for i in 1:4
        d = sgb_H_derivs(bg, i, r, χ; ε)
        offset = (i - 1) * 6
        params[offset + 1] = d.val
        params[offset + 2] = d.dr
        params[offset + 3] = d.dchi
        params[offset + 4] = d.drr
        params[offset + 5] = d.dchichi
        params[offset + 6] = d.drchi
    end
    params
end

# ═══════════════════════════════════════════════════════════════════════════════
#  H_i as RatPoly (for exact symbolic sGB K-coefficient extraction)
# ═══════════════════════════════════════════════════════════════════════════════

export load_H_ratpolys, load_H_ratpolys_per_order

"""
    _differentiate_ratpoly_r(rp::RatPoly) → RatPoly

Differentiate a RatPoly with respect to r (variable index 1).
For denom with r^t factor: d/dr [f/r^t] has numerator = f'·r - t·f
and denominator r^{t+1}. Other denom factors (Σ, Δ, s2) don't depend
solely on r, so for H_i (which only have r-denominators), this is exact.

NOTE: This assumes the only denominator factor is r^t (true for H_i from
the paper, where Σ/Δ/(1-χ²) don't appear in H_i denominators).
"""
function _differentiate_ratpoly_r(rp::RatPoly)
    # f(r,χ) / r^t  →  (f'r - t·f) / r^{t+1}
    f = rp.num
    t = rp.den.t
    f_r = differentiate(f, 1)   # df/dr
    if t == 0
        return RatPoly(f_r, rp.den)
    end
    r_poly = var_poly(1)
    new_num = f_r * r_poly - Float64(t) * f
    cleanup!(new_num; tol=1e-15)
    RatPoly(new_num, DenomSig(rp.den.p, rp.den.q, rp.den.s, t + 1))
end

"""
    _differentiate_ratpoly_chi(rp::RatPoly) → RatPoly

Differentiate a RatPoly with respect to χ (variable index 2).
For H_i, the denominator is r^t (no χ dependence), so d/dχ is simple.
"""
function _differentiate_ratpoly_chi(rp::RatPoly)
    f_chi = differentiate(rp.num, 2)   # df/dχ
    cleanup!(f_chi; tol=1e-15)
    RatPoly(f_chi, rp.den)
end

"""
    load_H_ratpolys(a; verbose=false) → Vector{RatPoly}

Load H₁-H₄ from the Mathematica notebook and convert to 24 RatPoly objects
(4 functions × 6 derivative components: val, ∂r, ∂χ, ∂²r, ∂²χ, ∂r∂χ).

The RatPolys are in the SparsePoly basis (r, χ, ω_re, ω_im, iu) but only
use variables 1 (r) and 2 (χ). Denominators are r-powers (DenomSig.t).

Requires a SymToPolyCtx (for the variable mapping). Creates one internally
using a dummy a_val (the H_i are evaluated at the given `a`).
"""
function load_H_ratpolys(a::Float64; verbose::Bool=false)
    verbose && (println("Loading H_i as RatPoly..."); flush(stdout))
    t0 = time()

    sections = _load_nb_sections(; verbose=false)

    # Build a minimal SymToPolyCtx for the conversion
    # We need r_poly and the var_idx mapping — create with Symbolics variables
    # that match the notebook variables (_r_, _χ_)
    @variables _r_ _χ_ _dummy_ω_re _dummy_ω_im _dummy_iu _dummy_a
    var_list = Num[_r_, _χ_, _dummy_ω_re, _dummy_ω_im, _dummy_iu, _dummy_a]
    ctx = SymToPolyCtx(var_list, a)

    H_ratpolys = Vector{RatPoly}(undef, 24)

    for i in 1:4
        verbose && (print("  H$i: parsing... "); flush(stdout))
        ti = time()

        expr_str = _box_to_str(sections["H$(i)"])

        # Parse to Julia expression, substitute M=1, α=1, a=numerical
        body = Meta.parse(expr_str)
        # Create a module-level evaluation with the correct variables
        sym_expr = Base.invokelatest(eval, :(let _r_ = $(Num(_r_)), _χ_ = $(Num(_χ_)),
                                                 _a_ = $a, _M_ = 1, _α_ = 1
                                                 Num($body)
                                             end))

        verbose && (print("symexpr_to_poly... "); flush(stdout))
        rp_val = symexpr_to_poly(sym_expr, ctx)
        cleanup!(rp_val.num; tol=1e-15)

        verbose && (print("diff... "); flush(stdout))
        rp_dr   = _differentiate_ratpoly_r(rp_val)
        rp_dchi = _differentiate_ratpoly_chi(rp_val)
        rp_drr  = _differentiate_ratpoly_r(rp_dr)
        rp_dchichi = _differentiate_ratpoly_chi(rp_dchi)
        rp_drchi   = _differentiate_ratpoly_chi(rp_dr)

        # Clean up all derivatives
        for rp in [rp_dr, rp_dchi, rp_drr, rp_dchichi, rp_drchi]
            cleanup!(rp.num; tol=1e-15)
        end

        offset = (i - 1) * 6
        H_ratpolys[offset + 1] = rp_val
        H_ratpolys[offset + 2] = rp_dr
        H_ratpolys[offset + 3] = rp_dchi
        H_ratpolys[offset + 4] = rp_drr
        H_ratpolys[offset + 5] = rp_dchichi
        H_ratpolys[offset + 6] = rp_drchi

        verbose && (@printf("%.1fs (val: %d terms, den.t=%d)\n",
                    time() - ti, length(rp_val.num.terms), rp_val.den.t); flush(stdout))
    end

    verbose && (@printf("  Total: %.1fs\n", time() - t0); flush(stdout))
    return H_ratpolys
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Per-a-order H_i loading (avoids d_max inflation from LCD)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _split_ratpoly_by_var(rp::RatPoly, var_idx::Int) → Dict{Int, RatPoly}

Split a RatPoly into per-exponent groups for variable `var_idx`.
Each group has the variable exponent stripped (set to 0) and the r-denominator
reduced by extracting common r-factors from the numerator.

After symexpr_to_poly with `a` as variable 3, the LCD clearing puts all
a-orders into one numerator with shared denominator r^{max_t}. Splitting
recovers per-order r-denominators by finding the minimum r-exponent in
each a-group and subtracting it.
"""
function _split_ratpoly_by_var(rp::RatPoly, var_idx::Int)
    @assert rp.den.p == 0 && rp.den.q == 0 && rp.den.s == 0 "Split by var only valid for r-only denominators (got p=$(rp.den.p), q=$(rp.den.q), s=$(rp.den.s))"
    groups = Dict{Int, Dict{ExpVec, Float64}}()
    for (e, c) in rp.num.terms
        a_exp = e[var_idx]
        if !haskey(groups, a_exp)
            groups[a_exp] = Dict{ExpVec, Float64}()
        end
        new_e = ntuple(i -> i == var_idx ? 0 : e[i], N_VARS)
        groups[a_exp][new_e] = get(groups[a_exp], new_e, 0.0) + c
    end

    result = Dict{Int, RatPoly}()
    for (a_exp, terms) in groups
        isempty(terms) && continue
        # Find minimum r-exponent (variable 1) to reduce r-denominator
        min_r = minimum(e[1] for (e, _) in terms)
        if min_r > 0
            new_terms = Dict{ExpVec, Float64}()
            for (e, c) in terms
                new_terms[ntuple(i -> i == 1 ? e[i] - min_r : e[i], N_VARS)] = c
            end
            result[a_exp] = RatPoly(SparsePoly(new_terms),
                                     DenomSig(rp.den.p, rp.den.q, rp.den.s,
                                              max(0, rp.den.t - min_r)))
        else
            result[a_exp] = RatPoly(SparsePoly(terms), rp.den)
        end
    end
    return result
end

"""
    _eval_ratpoly_rchi(rp::RatPoly, r_val, chi_val) → Float64

Evaluate a RatPoly at (r, χ) assuming only variables 1 (r) and 2 (χ) are used,
and denominator is r^t only (p=q=s=0). For verification purposes.
"""
function _eval_ratpoly_rchi(rp::RatPoly, r_val::Float64, chi_val::Float64)
    @assert rp.den.p == 0 && rp.den.q == 0 && rp.den.s == 0 "eval_ratpoly_rchi only valid for r-only denominators"
    num_val = 0.0
    for (e, c) in rp.num.terms
        num_val += c * r_val^e[1] * chi_val^e[2]
    end
    den_val = r_val^rp.den.t
    return num_val / den_val
end

"""
    load_H_ratpolys_per_order(; verbose=false) → (Vector{Vector{RatPoly}}, Vector{Int})

Load H₁-H₄ from the Mathematica notebook as per-a-order RatPolys.

Returns `(H_per_order, a_powers)` where:
  - `H_per_order[k]`: 24-element Vector{RatPoly} for a-order k
  - `a_powers[k]`: the a-exponent for order k (e.g., 0, 2, 4, ..., 40)

Each inner vector: [H1, H1_r, H1_χ, H1_rr, H1_χχ, H1_rχ, H2, ..., H4_rχ]

The spin parameter `a` is NOT evaluated. Use `a^a_powers[k]` as weight when
summing orders downstream. This avoids the LCD d_max inflation (80→~15 per order)
that occurs when all a-orders are summed before polynomial extraction.
"""
function load_H_ratpolys_per_order(; verbose::Bool=false)
    verbose && (println("Loading H_i as per-a-order RatPoly..."); flush(stdout))
    t0 = time()

    sections = _load_nb_sections(; verbose=false)

    # Map _a_ to variable index 6 (consistent with N_VARS=6 layout).
    # Use a_val=1.0 for SymToPolyCtx so Σ/Δ polys have multi-term structure
    # and don't accidentally match H_i's monomial r-denominators.
    @variables _r_ _χ_ _dummy_ωre _dummy_ωim _dummy_iu _a_
    var_list = Num[_r_, _χ_, _dummy_ωre, _dummy_ωim, _dummy_iu, _a_]
    ctx = SymToPolyCtx(var_list, 1.0)

    # Parse each H_i and split by a-power
    all_rp_split = Vector{Dict{Int, RatPoly}}(undef, 4)
    all_a_orders = Set{Int}()

    for i in 1:4
        verbose && (print("  H$i: parsing... "); flush(stdout))
        ti = time()

        expr_str = _box_to_str(sections["H$(i)"])
        body = Meta.parse(expr_str)
        # Keep _a_ as Symbolics variable (NOT numerical)
        sym_expr = Base.invokelatest(eval, :(let _r_ = $(Num(_r_)), _χ_ = $(Num(_χ_)),
                                                 _a_ = $(Num(_a_)),
                                                 _M_ = 1, _α_ = 1
                                                 Num($body)
                                             end))

        verbose && (print("symexpr_to_poly... "); flush(stdout))
        rp_full = symexpr_to_poly(sym_expr, ctx)
        cleanup!(rp_full.num; tol=1e-15)

        verbose && (print("split... "); flush(stdout))
        split = _split_ratpoly_by_var(rp_full, 6)  # 6 = a-variable index
        all_rp_split[i] = split
        union!(all_a_orders, keys(split))

        verbose && (@printf("%.1fs (%d a-orders, LCD den.t=%d)\n",
                    time() - ti, length(split), rp_full.den.t); flush(stdout))
    end

    sorted_orders = sort(collect(all_a_orders))
    n_orders = length(sorted_orders)
    verbose && (println("  Total a-orders: $n_orders (a^$(sorted_orders[1])..a^$(sorted_orders[end]))"); flush(stdout))

    # Build per-order RatPolys with derivatives
    result = Vector{Vector{RatPoly}}(undef, n_orders)

    for (oi, a_pow) in enumerate(sorted_orders)
        H_order = Vector{RatPoly}(undef, 24)

        for i in 1:4
            rp_val = get(all_rp_split[i], a_pow, RatPoly(0.0))
            cleanup!(rp_val.num; tol=1e-15)

            rp_dr   = _differentiate_ratpoly_r(rp_val)
            rp_dchi = _differentiate_ratpoly_chi(rp_val)
            rp_drr  = _differentiate_ratpoly_r(rp_dr)
            rp_dchichi = _differentiate_ratpoly_chi(rp_dchi)
            rp_drchi   = _differentiate_ratpoly_chi(rp_dr)

            for rp in [rp_dr, rp_dchi, rp_drr, rp_dchichi, rp_drchi]
                cleanup!(rp.num; tol=1e-15)
            end

            offset = (i - 1) * 6
            H_order[offset + 1] = rp_val
            H_order[offset + 2] = rp_dr
            H_order[offset + 3] = rp_dchi
            H_order[offset + 4] = rp_drr
            H_order[offset + 5] = rp_dchichi
            H_order[offset + 6] = rp_drchi
        end

        if verbose
            max_t = maximum(H_order[j].den.t for j in 1:24)
            max_terms = maximum(length(H_order[j].num.terms) for j in 1:24)
            @printf("  a^%d: max den.t=%d, max terms=%d\n", a_pow, max_t, max_terms)
            flush(stdout)
        end

        result[oi] = H_order
    end

    verbose && (@printf("  Total: %.1fs\n", time() - t0); flush(stdout))
    return result, sorted_orders
end

"""
    verify_H_ratpolys_per_order(H_per_order, a_powers, a; r_test, chi_test, verbose)

Verify per-order splitting by comparing summed per-order evaluation against
the original `load_H_ratpolys(a)` at test points.
"""
function verify_H_ratpolys_per_order(H_per_order::Vector{Vector{RatPoly}},
                                      a_powers::Vector{Int},
                                      a::Float64;
                                      r_test::Float64=5.0,
                                      chi_test::Float64=0.5,
                                      verbose::Bool=false)
    H_ref = load_H_ratpolys(a; verbose=false)
    max_err = 0.0
    for j in 1:24
        ref_val = _eval_ratpoly_rchi(H_ref[j], r_test, chi_test)
        sum_val = 0.0
        for (oi, a_pow) in enumerate(a_powers)
            rp = H_per_order[oi][j]
            sum_val += a^a_pow * _eval_ratpoly_rchi(rp, r_test, chi_test)
        end
        err = abs(sum_val - ref_val)
        rel = ref_val == 0 ? err : err / abs(ref_val)
        max_err = max(max_err, rel)
        verbose && rel > 1e-10 && @printf("  H[%d] at (%.1f,%.1f): ref=%.6e sum=%.6e rel=%.2e\n",
                                           j, r_test, chi_test, ref_val, sum_val, rel)
    end
    verbose && @printf("  Max relative error: %.2e\n", max_err)
    return max_err
end
