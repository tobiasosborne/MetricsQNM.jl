# Symbolic linearization of sGB-modified Einstein equations.
# Extends linearize.jl to handle g = g_Kerr + ζ·δg background metric.
#
# Strategy: use ABSTRACT symbolic variables for H_i and their derivatives
# (24 parameters total), keeping expressions Kerr-sized. At evaluation time,
# plug in numerical H_i values from sgb_background.jl.
#
# Uses O(ζ) perturbation theory:
#   g⁻¹ = g⁻¹_Kerr - ζ · g⁻¹_Kerr · δg · g⁻¹_Kerr + O(ζ²)
#   Γ = Γ_Kerr + ζ · δΓ_bg + O(ζ²)
#   δR = δR_GR + ζ · correction + O(ζ²)

using Symbolics
using TensorGR: symbolic_metric, symbolic_christoffel, sym_deriv

export compute_sgb_correction_equations, compile_sgb_correction
export extract_sgb_coefficients, extract_sgb_coefficients_complex, separate_sgb_omega

# ═══════════════════════════════════════════════════════════════════════════════
#  Abstract H_i symbolic variables
# ═══════════════════════════════════════════════════════════════════════════════

# 4 functions × (value + 2 first derivs + 3 second derivs) = 24 parameters.
# These are ABSTRACT: they don't carry the actual polynomial content of H_i.
# At evaluation time, they get numerical values from sgb_background.

@variables H1 H1_r H1_chi H1_rr H1_chichi H1_rchi
@variables H2 H2_r H2_chi H2_rr H2_chichi H2_rchi
@variables H3 H3_r H3_chi H3_rr H3_chichi H3_rchi
@variables H4 H4_r H4_chi H4_rr H4_chichi H4_rchi

const _H_PARAMS = Num[H1, H1_r, H1_chi, H1_rr, H1_chichi, H1_rchi,
                       H2, H2_r, H2_chi, H2_rr, H2_chichi, H2_rchi,
                       H3, H3_r, H3_chi, H3_rr, H3_chichi, H3_rchi,
                       H4, H4_r, H4_chi, H4_rr, H4_chichi, H4_rchi]

"""
    _get_H_params()

Return 4 named tuples grouping the 24 H_i symbolic variables:
    H_params[i] = (val, dr, dchi, drr, dchichi, drchi)
"""
function _get_H_params()
    ntuple(4) do i
        offset = (i - 1) * 6
        (val     = _H_PARAMS[offset + 1],
         dr      = _H_PARAMS[offset + 2],
         dchi    = _H_PARAMS[offset + 3],
         drr     = _H_PARAMS[offset + 4],
         dchichi = _H_PARAMS[offset + 5],
         drchi   = _H_PARAMS[offset + 6])
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Metric correction using abstract H_i
# ═══════════════════════════════════════════════════════════════════════════════

"""
Build δg_{μν} using abstract H_i symbolic variables.
The H_i.val are the function values; derivatives are handled separately
when computing ∂δg in the Christoffel correction.
"""
function _metric_correction_abstract(Hp, Sig, Del, a_s, r, chi, M)
    H1, H2, H3, H4 = Hp[1].val, Hp[2].val, Hp[3].val, Hp[4].val

    δg = zeros(Num, 4, 4)
    δg[1,1] = -H1
    δg[1,4] = δg[4,1] = -2M^2 * a_s * r * (1 - chi^2) / Sig * H2
    δg[2,2] = Sig / Del * H3
    δg[3,3] = Sig / (1 - chi^2) * H3
    δg[4,4] = (r^2 + M^2 * a_s^2 + 2M^3 * a_s^2 * r * (1 - chi^2) / Sig) *
              (1 - chi^2) * H4
    return δg
end

"""
Compute ∂_coord δg_{μν} where coord is r or chi.
Uses the product rule: ∂(g_Kerr · H_i) = (∂g_Kerr)·H_i + g_Kerr·(∂H_i).
The ∂H_i are abstract symbolic variables (Hp[i].dr or Hp[i].dchi).
"""
function _metric_correction_deriv(Hp, Sig, Del, a_s, r, chi, M, coord, coords)
    # Determine which H derivative to use
    is_r = isequal(coord, coords[2])
    dH(hp) = is_r ? hp.dr : hp.dchi

    H1, H2, H3, H4 = Hp[1].val, Hp[2].val, Hp[3].val, Hp[4].val
    dH1, dH2, dH3, dH4 = dH(Hp[1]), dH(Hp[2]), dH(Hp[3]), dH(Hp[4])

    # Kerr metric pieces (for product rule)
    g_tphi_kerr = -2M^2 * a_s * r * (1 - chi^2) / Sig
    g_rr_kerr = Sig / Del
    g_chichi_kerr = Sig / (1 - chi^2)
    g_phiphi_kerr = (r^2 + M^2 * a_s^2 + 2M^3 * a_s^2 * r * (1 - chi^2) / Sig) * (1 - chi^2)

    dδg = zeros(Num, 4, 4)
    # ∂(δg_tt) = ∂(-H1) = -∂H1
    dδg[1,1] = -dH1
    # ∂(δg_tφ) = ∂(g_tφ_Kerr · H2) = (∂g_tφ)·H2 + g_tφ·∂H2
    dg_tphi = sym_deriv(g_tphi_kerr, coord)
    dδg[1,4] = dδg[4,1] = dg_tphi * H2 + g_tphi_kerr * dH2
    # ∂(δg_rr) = ∂(g_rr · H3) = (∂g_rr)·H3 + g_rr·∂H3
    dg_rr = sym_deriv(g_rr_kerr, coord)
    dδg[2,2] = dg_rr * H3 + g_rr_kerr * dH3
    # ∂(δg_χχ) = ∂(g_χχ · H3)
    dg_chichi = sym_deriv(g_chichi_kerr, coord)
    dδg[3,3] = dg_chichi * H3 + g_chichi_kerr * dH3
    # ∂(δg_φφ) = ∂(g_φφ · H4)
    dg_phiphi = sym_deriv(g_phiphi_kerr, coord)
    dδg[4,4] = dg_phiphi * H4 + g_phiphi_kerr * dH4

    return dδg
end

"""
Compute the O(ζ) correction to the inverse metric:
    δg^{μν} = -g^{μα}_Kerr · δg_{αβ} · g^{βν}_Kerr
"""
function _inverse_correction(ginv_kerr, δg)
    dim = size(ginv_kerr, 1)
    δginv = zeros(Num, dim, dim)
    for μ in 1:dim, ν in 1:dim
        s = Num(0)
        for α in 1:dim, β in 1:dim
            s -= ginv_kerr[μ, α] * δg[α, β] * ginv_kerr[β, ν]
        end
        δginv[μ, ν] = s
    end
    δginv
end

"""
Compute the O(ζ) correction to Christoffel symbols using abstract H derivatives.
Instead of calling sym_deriv on δg (which would need to differentiate through H_i),
we use pre-computed ∂_r δg and ∂_χ δg from _metric_correction_deriv.
"""
function _christoffel_correction_abstract(ginv_kerr, δginv, g_kerr_cov, δg,
                                           dδg_dr, dδg_dchi, coords)
    dim = 4
    # Precompute Kerr metric derivatives
    dg_dr = [sym_deriv(g_kerr_cov[i, j], coords[2]) for i in 1:dim, j in 1:dim]
    dg_dchi = [sym_deriv(g_kerr_cov[i, j], coords[3]) for i in 1:dim, j in 1:dim]

    # Derivative lookup: ∂_coord g or δg
    function dg_kerr(i, j, coord_idx)
        coord_idx == 1 && return Num(0)  # ∂_t g = 0
        coord_idx == 2 && return dg_dr[i, j]
        coord_idx == 3 && return dg_dchi[i, j]
        coord_idx == 4 && return Num(0)  # ∂_φ g = 0
    end
    function ddg(i, j, coord_idx)
        coord_idx == 1 && return Num(0)
        coord_idx == 2 && return dδg_dr[i, j]
        coord_idx == 3 && return dδg_dchi[i, j]
        coord_idx == 4 && return Num(0)
    end

    δΓ_bg = Array{Num}(undef, dim, dim, dim)
    for α in 1:dim, β in 1:dim, γ in 1:dim
        s = Num(0)
        for δ in 1:dim
            # From δg^{αδ} and Kerr metric derivatives
            s += δginv[α, δ] * (dg_kerr(γ, δ, β) + dg_kerr(β, δ, γ) - dg_kerr(β, γ, δ))
            # From g^{αδ}_Kerr and δg derivatives
            s += ginv_kerr[α, δ] * (ddg(γ, δ, β) + ddg(β, δ, γ) - ddg(β, γ, δ))
        end
        δΓ_bg[α, β, γ] = s / 2
    end
    δΓ_bg
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_sgb_correction_equations(m_mode; verbose=false)

Compute the O(ζ) correction to the 10 linearized field equations from the
sGB-modified background metric, using ABSTRACT H_i symbolic parameters.

The resulting equations contain 24 abstract H parameters (H1, H1_r, H1_chi, ...)
plus the usual (r, chi, a_s, omega_re, omega_im, iu_sym, h_k derivatives).

Returns (eqs_correction, coords, params, hfuncs, omega_vars, H_params).
"""
function compute_sgb_correction_equations(m_mode::Int; verbose::Bool=false)
    t0_total = time()

    Hp = _get_H_params()

    # ── Step 1: Build pure Kerr quantities ────────────────────────────────
    verbose && (println("Building Kerr background..."); flush(stdout))
    t0 = time()
    sm_kerr, coords, params = kerr_symbolic_metric()
    Γ_kerr = symbolic_christoffel(sm_kerr)
    ginv_kerr = sm_kerr.ginv
    verbose && (@printf("  Kerr: %.1fs\n", time() - t0); flush(stdout))

    @variables r chi a_s
    M = 1
    Sig = r^2 + a_s^2 * chi^2
    Del = r^2 - 2M * r + a_s^2

    # ── Step 2: Build metric correction and derivatives ───────────────────
    verbose && (println("Building metric correction δg..."); flush(stdout))
    t0 = time()
    δg = _metric_correction_abstract(Hp, Sig, Del, a_s, r, chi, M)
    dδg_dr = _metric_correction_deriv(Hp, Sig, Del, a_s, r, chi, M, coords[2], coords)
    dδg_dchi = _metric_correction_deriv(Hp, Sig, Del, a_s, r, chi, M, coords[3], coords)
    verbose && (@printf("  δg + derivs: %.1fs\n", time() - t0); flush(stdout))

    g_kerr_cov = zeros(Num, 4, 4)
    g_kerr_cov[1,1] = -(1 - 2M * r / Sig)
    g_kerr_cov[1,4] = g_kerr_cov[4,1] = -2M^2 * a_s * r * (1 - chi^2) / Sig
    g_kerr_cov[2,2] = Sig / Del
    g_kerr_cov[3,3] = Sig / (1 - chi^2)
    g_kerr_cov[4,4] = (r^2 + M^2 * a_s^2 + 2M^3 * a_s^2 * r * (1 - chi^2) / Sig) * (1 - chi^2)

    verbose && (println("Computing δg⁻¹..."); flush(stdout))
    t0 = time()
    δginv = _inverse_correction(ginv_kerr, δg)
    verbose && (@printf("  δg⁻¹: %.1fs\n", time() - t0); flush(stdout))

    verbose && (println("Computing δΓ_bg..."); flush(stdout))
    t0 = time()
    δΓ_bg = _christoffel_correction_abstract(ginv_kerr, δginv, g_kerr_cov, δg,
                                              dδg_dr, dδg_dchi, coords)
    verbose && (@printf("  δΓ_bg: %.1fs\n", time() - t0); flush(stdout))

    # ── Step 3: Perturbation ansatz ──────────────────────────────────────
    @variables omega_re omega_im iu_sym
    h, hfuncs = regge_wheeler_perturbation(coords, m_mode)

    # ── Step 4: GR linearized Christoffel ────────────────────────────────
    verbose && (println("Computing δΓ_pert (GR)..."); flush(stdout))
    t0 = time()
    δΓ_pert = linearized_christoffel(ginv_kerr, h, coords, omega_re, omega_im, iu_sym, m_mode)
    verbose && (@printf("  δΓ_pert: %.1fs\n", time() - t0); flush(stdout))

    # ── Step 5: O(ζ) correction to linearized Christoffel ────────────────
    verbose && (println("Computing δΓ_pert_corr (O(ζ))..."); flush(stdout))
    t0 = time()
    δΓ_pert_corr = linearized_christoffel(δginv, h, coords, omega_re, omega_im, iu_sym, m_mode)
    verbose && (@printf("  δΓ_pert_corr: %.1fs\n", time() - t0); flush(stdout))

    # ── Step 6: O(ζ) correction to linearized Ricci ──────────────────────
    verbose && (println("Computing δR correction (O(ζ))..."); flush(stdout))
    t0 = time()
    dim = 4
    t_coord, phi_coord = coords[1], coords[4]
    δR_corr = Array{Any}(undef, dim, dim)

    for μ in 1:dim, ν in 1:dim
        verbose && μ == ν && (print("  ($μ,$ν)... "); flush(stdout))
        s = Num(0)

        # Source B: linearized Ricci of δΓ_pert_corr with Kerr background Γ
        for α in 1:dim
            s += mode_deriv(δΓ_pert_corr[α, μ, ν], coords[α], t_coord, phi_coord,
                           omega_re, omega_im, iu_sym, m_mode)
            s -= mode_deriv(δΓ_pert_corr[α, α, ν], coords[μ], t_coord, phi_coord,
                           omega_re, omega_im, iu_sym, m_mode)
        end
        for α in 1:dim, β in 1:dim
            s += Γ_kerr[α, α, β] * δΓ_pert_corr[β, μ, ν]
            s += δΓ_pert_corr[α, α, β] * Γ_kerr[β, μ, ν]
            s -= Γ_kerr[α, μ, β] * δΓ_pert_corr[β, α, ν]
            s -= δΓ_pert_corr[α, μ, β] * Γ_kerr[β, α, ν]
        end

        # Source A: cross-terms between δΓ_bg and δΓ_pert
        for α in 1:dim, β in 1:dim
            s += δΓ_bg[α, α, β] * δΓ_pert[β, μ, ν]
            s += δΓ_pert[α, α, β] * δΓ_bg[β, μ, ν]
            s -= δΓ_bg[α, μ, β] * δΓ_pert[β, α, ν]
            s -= δΓ_pert[α, μ, β] * δΓ_bg[β, α, ν]
        end

        δR_corr[μ, ν] = s
        verbose && μ == ν && (@printf("done\n"); flush(stdout))
    end
    verbose && (@printf("  δR correction: %.1fs\n", time() - t0); flush(stdout))

    # ── Step 7: Raise index ──────────────────────────────────────────────
    verbose && (println("Computing δR^μ_ν correction..."); flush(stdout))
    t0 = time()

    δR_GR = linearized_ricci(Γ_kerr, δΓ_pert, coords, omega_re, omega_im, iu_sym, m_mode)

    δR_mixed_corr = Array{Any}(undef, dim, dim)
    for μ in 1:dim, ν in 1:dim
        s = Num(0)
        for α in 1:dim
            s += ginv_kerr[μ, α] * δR_corr[α, ν]
            s += δginv[μ, α] * δR_GR[α, ν]
        end
        δR_mixed_corr[μ, ν] = s
    end
    verbose && (@printf("  δR^μ_ν correction: %.1fs\n", time() - t0); flush(stdout))

    # ── Step 8: Extract 10 components ────────────────────────────────────
    idx = [(1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,3),(3,4),(4,4)]
    eqs_correction = [δR_mixed_corr[μ, ν] for (μ, ν) in idx]

    verbose && @printf("Total: %.1fs\n", time() - t0_total)
    return eqs_correction, coords, params, hfuncs, (omega_re, omega_im, iu_sym), Hp
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Compile to fast evaluator
# ═══════════════════════════════════════════════════════════════════════════════

struct CompiledSGBCorrection
    fns::Vector{Any}           # 10 compiled functions
    h_param_syms::Vector{Num}  # the 24 H parameter symbols in order
    h_deriv_names::Vector{String}  # names of the 40 h-derivative variables
    n_base::Int                # number of base vars (r, chi, a, ω_re, ω_im, iu)
    n_H::Int                   # number of H params (24)
    n_h::Int                   # number of h-derivative vars (40)
end

"""
    compile_sgb_correction(m_mode; verbose=false)

Compute and compile the 10 sGB correction equations into fast Julia callables.

Each callable takes a single vector argument:
    args = [r, χ, ω_re, ω_im, iu, a, H1, H1_r, ..., H4_rchi, p1, ..., p_nh]
           (6 base)  (24 H params)  (n_h h-derivative placeholders)

Returns CompiledSGBCorrection.
"""
function compile_sgb_correction(m_mode::Int; verbose::Bool=false)
    eqs, coords, params, aparams, ωvars, Hp = compute_sgb_correction_equations(m_mode; verbose)

    rv, cv = coords[2], coords[3]
    a_s = aparams[1]
    omega_re, omega_im, iu_sym = ωvars

    # Discover h-derivative variables
    all_vars_set = Set{Any}()
    for eq in eqs
        union!(all_vars_set, Symbolics.get_variables(eq))
    end

    # Known non-h variables
    base_syms = [rv, cv, omega_re, omega_im, iu_sym, a_s]
    h_params = copy(_H_PARAMS)
    known = Set{Any}(base_syms)
    union!(known, h_params)

    # h-derivative variables: replace with plain placeholders (like coefficients.jl)
    h_deriv_vars = sort!(collect(setdiff(all_vars_set, known)), by=x -> string(x))
    n_h = length(h_deriv_vars)
    placeholders = [Symbolics.variable(Symbol("p_$i")) for i in 1:n_h]
    sub_dict = Dict{Any, Any}(h_deriv_vars[i] => placeholders[i] for i in 1:n_h)

    # Argument vector: [r, χ, ω_re, ω_im, iu, a, H1..H4_rchi, p_1..p_nh]
    all_args = vcat(Num[rv, cv, omega_re, omega_im, iu_sym, a_s], h_params, placeholders)

    verbose && println("Compiling $(length(eqs)) equations ($(length(all_args)) vars, $n_h h-derivs)...")
    verbose && flush(stdout)

    fns = Any[]
    for (k, eq) in enumerate(eqs)
        verbose && (print("  eq $k... "); flush(stdout))
        t0 = time()
        eq_sub = Symbolics.substitute(eq, sub_dict)
        f_expr = build_function(eq_sub, all_args; expression=Val{true})
        f = eval(f_expr)
        push!(fns, f)
        verbose && (@printf("%.1fs\n", time() - t0); flush(stdout))
    end

    h_deriv_names = [string(v) for v in h_deriv_vars]
    CompiledSGBCorrection(fns, h_params, h_deriv_names,
                           length(base_syms), length(h_params), n_h)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Coefficient extraction for compiled correction
# ═══════════════════════════════════════════════════════════════════════════════

"""
    extract_sgb_coefficients(csc, r, χ, ω_re, ω_im, a, h_params; iu_val=0.0)

Extract the 10 × n_h correction coefficient matrix at a single point.
Same probing approach as galerkin.jl: set p_d = 1, all others = 0.

`h_params` is a length-24 vector from `sgb_H_params(bg, r, χ)`.
"""
function extract_sgb_coefficients(csc::CompiledSGBCorrection,
                                   r, χ, ω_re, ω_im, a, h_params::Vector{Float64};
                                   iu_val::Float64=0.0)
    n_h = csc.n_h
    n_total = csc.n_base + csc.n_H + n_h
    args = zeros(Float64, n_total)
    args[1] = r; args[2] = χ; args[3] = ω_re; args[4] = ω_im
    args[5] = iu_val; args[6] = a
    args[7:30] .= h_params  # 24 H parameters

    C = zeros(Float64, 10, n_h)
    for d in 1:n_h
        args[31:end] .= 0.0
        args[30 + d] = 1.0
        for k in 1:10
            C[k, d] = csc.fns[k](args)
        end
    end
    return C
end

"""
    extract_sgb_coefficients_complex(csc, r, χ, ω, a, h_params)

Complex coefficient extraction using the iu-polynomial trick.
"""
function extract_sgb_coefficients_complex(csc::CompiledSGBCorrection,
                                           r, χ, ω::ComplexF64, a,
                                           h_params::Vector{Float64})
    f0  = extract_sgb_coefficients(csc, r, χ, real(ω), imag(ω), a, h_params; iu_val=0.0)
    f1  = extract_sgb_coefficients(csc, r, χ, real(ω), imag(ω), a, h_params; iu_val=1.0)
    fm1 = extract_sgb_coefficients(csc, r, χ, real(ω), imag(ω), a, h_params; iu_val=-1.0)
    a0 = f0
    a1 = (f1 - fm1) / 2
    a2 = (f1 + fm1) / 2 - f0
    return (a0 - a2) + im * a1
end

"""
    separate_sgb_omega(csc, r, χ, a, h_params) → (C0, C1, C2)

Extract ω-polynomial correction coefficients: C(ω) = C0 + C1·ω + C2·ω².
"""
function separate_sgb_omega(csc::CompiledSGBCorrection,
                             r, χ, a, h_params::Vector{Float64})
    C_at_0  = extract_sgb_coefficients_complex(csc, r, χ, ComplexF64(0), a, h_params)
    C_at_1  = extract_sgb_coefficients_complex(csc, r, χ, ComplexF64(1), a, h_params)
    C_at_m1 = extract_sgb_coefficients_complex(csc, r, χ, ComplexF64(-1), a, h_params)
    C0 = C_at_0
    C1 = (C_at_1 - C_at_m1) / 2
    C2 = (C_at_1 + C_at_m1) / 2 - C0
    return C0, C1, C2
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Collocation D̃⁽¹⁾ assembly
# ═══════════════════════════════════════════════════════════════════════════════

"""
    build_sgb_correction_system(a, N, m, bg; verbose=false)

Build the D̃⁽¹⁾ correction matrices for sGB gravity at spin `a`.

Uses the same collocation approach as the GR D̃ builder:
1. Compile the sGB correction equations (cached after first call)
2. At each collocation point, evaluate H_i and derivatives from sgb_background
3. Extract PDE coefficients using the iu-polynomial trick
4. Assemble D̃₀⁽¹⁾, D̃₁⁽¹⁾, D̃₂⁽¹⁾ via spectral interpolation

Returns a METRICSSystem containing the correction matrices.
"""
function build_sgb_correction_system(a::Float64, N::Int, m::Int,
                                      bg::SGBBackground;
                                      verbose::Bool=false)
    # Step 1: Get compiled correction evaluator
    verbose && (println("Compiling sGB correction..."); flush(stdout))
    t0 = time()
    csc = compile_sgb_correction(m; verbose)
    verbose && (@printf("  Compiled: %.1fs\n", time() - t0); flush(stdout))

    # Step 2: Build the GR system for spectral infrastructure
    verbose && (println("Building GR system for reference..."); flush(stdout))
    t0 = time()
    sys_gr = build_system_bespoke(a, N, m; verbose=false)
    verbose && (@printf("  GR system: %.1fs\n", time() - t0); flush(stdout))

    # Step 3: Use the GR compiled field equations to get h-derivative ordering
    # The GR evaluator in coefficients.jl uses the same h-derivative variables
    verbose && (println("Building correction D̃ via collocation..."); flush(stdout))
    t0 = time()

    # Get collocation infrastructure from the GR system
    cfe = compile_field_equations(m)
    basis = spectral_basis(N, m)
    bs = (N + 1)^2

    # Collocation points
    r_plus = r_plus_val = 1.0 + sqrt(1.0 - a^2)
    z_nodes = basis.radial.nodes   # Chebyshev nodes in z ∈ [-1, 1]
    χ_nodes = basis.angular.nodes  # Legendre nodes

    n_r = length(z_nodes)
    n_χ = length(χ_nodes)
    n_eq = 10
    n_fields = 6
    n_v = n_fields * bs

    # Initialize D̃⁽¹⁾ matrices (10·n_r·n_χ × 6·bs)
    n_rows = n_eq * n_r * n_χ
    D0_corr = zeros(ComplexF64, n_rows, n_v)
    D1_corr = zeros(ComplexF64, n_rows, n_v)
    D2_corr = zeros(ComplexF64, n_rows, n_v)

    # For each collocation point, evaluate the correction equations
    # using the iu-polynomial trick (same as galerkin.jl)
    for (j_r, z) in enumerate(z_nodes)
        r_val = r_of_z(z, r_plus_val)
        for (j_χ, χ_val) in enumerate(χ_nodes)
            # H_i params at this point
            h_params = sgb_H_params(bg, r_val, χ_val)

            # Extract coefficients of each h-derivative term
            # using the compiled correction evaluator
            row_base = ((j_r - 1) * n_χ + (j_χ - 1)) * n_eq

            # For each h-derivative variable, evaluate correction at unit perturbation
            # This requires knowing which h-derivative maps to which spectral coefficient
            # TODO: wire up the full collocation assembly
            # For now, this is a placeholder structure
        end
    end

    verbose && (@printf("  Collocation: %.1fs\n", time() - t0); flush(stdout))

    # Return correction system
    METRICSSystem(D0_corr, D1_corr, D2_corr, N, m, a)
end
