#!/usr/bin/env julia
# Test hypothesis: shared d_max fixes the perturbation theory.
# Both D̃⁽⁰⁾ and D̃⁽¹⁾ must use the SAME d_max in _r_to_z so they live
# in the same (1-z)^d_max polynomial space.

using MetricsQNM
using LinearAlgebra
using Printf
using Symbolics

function test_shared_dmax(a::Float64, N::Int, m::Int; parity::Symbol=:polar)
    println("=" ^ 70)
    @printf("Shared d_max test: a=%.3f, N=%d\n", a, N)
    println("=" ^ 70); flush(stdout)

    rp = MetricsQNM.r_plus(a)
    ω_leaver = leaver_qnm(a; l=2, m=m, n=0)

    # ── Extract GR K coefficients ──
    println("[1] GR K extraction..."); flush(stdout)
    t0 = time()
    K0_gr_r, K1_gr_r, K2_gr_r = extract_G_bespoke(a; verbose=true)
    d_max_gr = 0
    for K in (K0_gr_r, K1_gr_r, K2_gr_r), k in 1:10
        for ((γ, δ, σ, α, β, j), _) in K.equations[k]
            d_max_gr = max(d_max_gr, δ)
        end
    end
    @printf("  GR d_max = %d (%.1fs)\n", d_max_gr, time() - t0); flush(stdout)

    # ── Extract sGB K coefficients ──
    println("\n[2] sGB K extraction (Phase 1-3)..."); flush(stdout)
    t0 = time()
    H_ratpolys = load_H_ratpolys(a; verbose=true)
    coeffs, h_map, n_eqs = MetricsQNM.extract_sgb_coefficients_symbolic(a; verbose=true)
    @variables r chi omega_re omega_im iu_sym
    var_list = Num[r, chi, omega_re, omega_im, iu_sym]
    ctx = MetricsQNM.SymToPolyCtx(var_list, a)
    K0_sgb_r, K1_sgb_r, K2_sgb_r = MetricsQNM.combine_sgb_K(coeffs, H_ratpolys, h_map, n_eqs, ctx;
                                                                verbose=true)
    d_max_sgb = 0
    for K in (K0_sgb_r, K1_sgb_r, K2_sgb_r), k in 1:10
        for ((γ, δ, σ, α, β, j), _) in K.equations[k]
            d_max_sgb = max(d_max_sgb, δ)
        end
    end
    @printf("  sGB d_max = %d (%.1fs)\n", d_max_sgb, time() - t0); flush(stdout)

    # ── Shared d_max ──
    d_max = max(d_max_gr, d_max_sgb)
    @printf("\n  SHARED d_max = %d\n", d_max); flush(stdout)

    # ── Transform both with shared d_max ──
    println("\n[3] r → z with shared d_max..."); flush(stdout)
    K0_gr_z = MetricsQNM._r_to_z(K0_gr_r, rp, d_max)
    K1_gr_z = MetricsQNM._r_to_z(K1_gr_r, rp, d_max)
    K2_gr_z = MetricsQNM._r_to_z(K2_gr_r, rp, d_max)

    K0_sgb_z = MetricsQNM._r_to_z(K0_sgb_r, rp, d_max)
    K1_sgb_z = MetricsQNM._r_to_z(K1_sgb_r, rp, d_max)
    K2_sgb_z = MetricsQNM._r_to_z(K2_sgb_r, rp, d_max)

    # Compute max sigma
    s_max = 0
    for K in (K0_gr_z, K1_gr_z, K2_gr_z, K0_sgb_z, K1_sgb_z, K2_sgb_z), k in 1:10
        for ((γ, δ, σ, α, β, j), _) in K.equations[k]
            s_max = max(s_max, σ)
        end
    end
    @printf("  max_sigma = %d\n", s_max); flush(stdout)

    # ── Assemble both systems ──
    println("\n[4] Assembly..."); flush(stdout)
    basis = spectral_basis(N, m; max_delta=d_max, max_sigma=max(s_max, 25))

    # GR system
    D0_gr = assemble_system(K0_gr_z, basis, a).D0
    D1_gr = assemble_system(K1_gr_z, basis, a).D0
    D2_gr = assemble_system(K2_gr_z, basis, a).D0
    sys_gr = METRICSSystem(D0_gr, D1_gr, D2_gr, N, m, a)
    sys_gr, gr_norm = normalize_system!(sys_gr)
    @printf("  GR: ‖D₀‖=%.2e  ‖D₁‖=%.2e  ‖D₂‖=%.2e\n",
            norm(sys_gr.D0), norm(sys_gr.D1), norm(sys_gr.D2)); flush(stdout)

    # sGB correction system
    D0_corr = assemble_system(K0_sgb_z, basis, a).D0
    D1_corr = assemble_system(K1_sgb_z, basis, a).D0
    D2_corr = assemble_system(K2_sgb_z, basis, a).D0
    sys_corr = METRICSSystem(D0_corr, D1_corr, D2_corr, N, m, a)
    normalize_system!(sys_corr, gr_norm)  # Apply GR normalization
    @printf("  sGB: ‖D₀⁽¹⁾‖=%.2e  ‖D₁⁽¹⁾‖=%.2e  ‖D₂⁽¹⁾‖=%.2e\n",
            norm(sys_corr.D0), norm(sys_corr.D1), norm(sys_corr.D2)); flush(stdout)

    # ── QEP solve ──
    println("\n[5] QEP solve..."); flush(stdout)
    result = solve_qep_with_vectors(sys_gr; ω₀=ω_leaver, refine=1, normalize=false)
    evals, evecs = result.eigenvalues, result.eigenvectors

    physical = [(i, e) for (i, e) in enumerate(evals)
                if isfinite(e) && imag(e) < 0 && abs(e) < 20]
    isempty(physical) && error("No physical eigenvalues!")
    best_i, ω0 = physical[argmin([abs(e - ω_leaver) for (_, e) in physical])]
    v0_raw = evecs[:, best_i]

    bs = (N + 1)^2
    am = abs(m)
    pinned_k = parity == :polar ? 1 : 5
    pinned_idx = (pinned_k - 1) * bs + nl_index(0, am, N, m)
    v0 = v0_raw / v0_raw[pinned_idx]

    residual = norm(MetricsQNM.Dtilde(sys_gr, ComplexF64(ω0)) * v0)
    J, _, _ = compute_jacobian(sys_gr, ComplexF64(ω0), v0; parity=parity)

    @printf("  ω⁽⁰⁾ = %.12f %+.12fi\n", real(ω0), imag(ω0))
    @printf("  |Δω| = %.2e,  ‖D̃·v₀‖ = %.2e,  cond(J) = %.1e\n",
            abs(ω0 - ω_leaver), residual, cond(J)); flush(stdout)

    # ── Perturbation solve ──
    println("\n[6] Perturbation solve..."); flush(stdout)
    ω1 = solve_sgb_perturbation(sys_corr, ComplexF64(ω0), v0, J)
    @printf("  ω⁽¹⁾ = %.8f %+.8fi\n", real(ω1), imag(ω1))
    @printf("  Paper: ω⁽¹⁾ = -0.364190 -0.042360i (all sources)\n"); flush(stdout)

    println("\n" * "=" ^ 70)
    println("DONE"); flush(stdout)
    return (ω0=ω0, ω1=ω1, sys_gr=sys_gr, sys_corr=sys_corr)
end

if abspath(PROGRAM_FILE) == @__FILE__
    a = parse(Float64, get(ARGS, 1, "0.3"))
    N = parse(Int, get(ARGS, 2, "4"))
    println("Julia threads: $(Threads.nthreads())"); flush(stdout)
    test_shared_dmax(a, N, 2)
end
