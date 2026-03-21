#!/usr/bin/env julia
# N-convergence test for sGB ω⁽¹⁾ (Source 1 only).
# Computes K coefficients ONCE (Phase 1-3, ~30 min), then sweeps N for assembly + solve.

using MetricsQNM
using LinearAlgebra
using Printf
using Symbolics

const TABLE_I = [
    (0.005,  0.05984,  0.00719, -0.22664, -0.07525),
    (0.1,    0.07282,  0.01353, -0.26228, -0.07146),
    (0.3,    0.09087,  0.00782, -0.36419, -0.04236),
]

function sweep_sgb(a::Float64, N_range, m::Int; parity::Symbol=:polar)
    println("=" ^ 70)
    @printf("sGB N-convergence sweep: a=%.3f, m=%d, parity=%s\n", a, m, parity)
    @printf("N values: %s\n", N_range)
    println("=" ^ 70); flush(stdout)

    rp = MetricsQNM.r_plus(a)
    ω_leaver = leaver_qnm(a; l=2, m=m, n=0)
    @printf("Leaver: ω = %.12f %+.12fi\n\n", real(ω_leaver), imag(ω_leaver)); flush(stdout)

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 1-3: N-independent — compute K coefficients ONCE
    # ══════════════════════════════════════════════════════════════════════
    println("[Phase 1] Loading H_i as RatPoly..."); flush(stdout)
    t0 = time()
    H_ratpolys = load_H_ratpolys(a; verbose=true)
    @printf("  Phase 1: %.1fs\n\n", time() - t0); flush(stdout)

    println("[Phase 2] Extracting c_{k,d,p} via double probing..."); flush(stdout)
    t0 = time()
    coeffs, h_map, n_eqs = MetricsQNM.extract_sgb_coefficients_symbolic(a; verbose=true)
    @printf("  Phase 2: %.1fs\n\n", time() - t0); flush(stdout)

    # Build context for Phase 3
    @variables r chi omega_re omega_im iu_sym
    var_list = Num[r, chi, omega_re, omega_im, iu_sym]
    ctx = MetricsQNM.SymToPolyCtx(var_list, a)

    println("[Phase 3] Multiply c_{k,d,p} × H_p → K coefficients..."); flush(stdout)
    t0 = time()
    K0_r, K1_r, K2_r = MetricsQNM.combine_sgb_K(coeffs, H_ratpolys, h_map, n_eqs, ctx;
                                                    verbose=true)
    @printf("  Phase 3: %.1fs\n\n", time() - t0); flush(stdout)

    # Phase 4: r → z (N-independent)
    d_max = 0
    for K in (K0_r, K1_r, K2_r), k in 1:n_eqs
        for ((γ, δ, σ, α, β, j), _) in K.equations[k]
            d_max = max(d_max, δ)
        end
    end
    @printf("[Phase 4] r → z transform (d_max=%d)...\n", d_max); flush(stdout)
    t0 = time()
    K0_z = MetricsQNM._r_to_z(K0_r, rp, d_max)
    K1_z = MetricsQNM._r_to_z(K1_r, rp, d_max)
    K2_z = MetricsQNM._r_to_z(K2_r, rp, d_max)
    @printf("  Phase 4: %.1fs\n\n", time() - t0); flush(stdout)

    # Compute max sigma from z-space coefficients
    s_max = 0
    for K in (K0_z, K1_z, K2_z), k in 1:n_eqs
        for ((γ, δ, σ, α, β, j), _) in K.equations[k]
            s_max = max(s_max, σ)
        end
    end
    @printf("  max_delta=%d, max_sigma=%d\n\n", d_max, s_max); flush(stdout)

    # ══════════════════════════════════════════════════════════════════════
    #  Sweep N: GR solve + assembly + perturbation (N-dependent)
    # ══════════════════════════════════════════════════════════════════════
    results = []
    for N in N_range
        println("─" ^ 50)
        @printf("N = %d\n", N); flush(stdout)
        t0_N = time()

        # GR system + QEP solve
        sys_gr, gr_norm_factors = build_system_bespoke(a, N, m; verbose=false)
        result = solve_qep_with_vectors(sys_gr; ω₀=ω_leaver, refine=1)
        evals, evecs = result.eigenvalues, result.eigenvectors

        physical = [(i, e) for (i, e) in enumerate(evals)
                    if isfinite(e) && imag(e) < 0 && abs(e) < 20]
        isempty(physical) && (println("  No physical eigenvalues!"); continue)
        best_i, ω0 = physical[argmin([abs(e - ω_leaver) for (_, e) in physical])]
        v0_raw = evecs[:, best_i]

        # Normalize eigenvector
        bs = (N + 1)^2
        am = abs(m)
        pinned_k = parity == :polar ? 1 : 5
        pinned_idx = (pinned_k - 1) * bs + nl_index(0, am, N, m)
        v0 = v0_raw / v0_raw[pinned_idx]

        residual = norm(MetricsQNM.Dtilde(sys_gr, ComplexF64(ω0)) * v0)
        J, _, _ = compute_jacobian(sys_gr, ComplexF64(ω0), v0; parity=parity)

        @printf("  ω⁽⁰⁾ = %.12f %+.12fi\n", real(ω0), imag(ω0))
        @printf("  |Δω⁰| = %.2e,  ‖D̃·v₀‖ = %.2e,  cond(J) = %.1e\n",
                abs(ω0 - ω_leaver), residual, cond(J)); flush(stdout)

        # Assemble D̃⁽¹⁾ for this N
        basis = spectral_basis(N, m; max_delta=d_max, max_sigma=max(s_max, 25))
        D0 = assemble_system(K0_z, basis, a).D0
        D1 = assemble_system(K1_z, basis, a).D0
        D2 = assemble_system(K2_z, basis, a).D0
        sys_corr = METRICSSystem(D0, D1, D2, N, m, a)
        normalize_system!(sys_corr, gr_norm_factors)  # Apply GR normalization

        @printf("  ‖D₀⁽¹⁾‖=%.2e  ‖D₁⁽¹⁾‖=%.2e  ‖D₂⁽¹⁾‖=%.2e\n",
                norm(sys_corr.D0), norm(sys_corr.D1), norm(sys_corr.D2)); flush(stdout)

        # Perturbation solve
        ω1 = solve_sgb_perturbation(sys_corr, ComplexF64(ω0), v0, J)
        @printf("  ω⁽¹⁾ = %.8f %+.8fi\n", real(ω1), imag(ω1))
        @printf("  N=%d total: %.1fs\n", N, time() - t0_N); flush(stdout)

        push!(results, (N=N, ω0=ω0, ω1=ω1, residual=residual, cond_J=cond(J)))
    end

    # Summary table
    println("\n" * "=" ^ 70)
    println("CONVERGENCE SUMMARY (Source 1 only)")
    println("=" ^ 70)
    @printf("%-4s  %-28s  %-28s  %-10s\n", "N", "ω⁽⁰⁾", "ω⁽¹⁾", "‖D̃·v₀‖")
    println("-" ^ 70)
    for r in results
        @printf("%-4d  %+.10f%+.10fi  %+.10f%+.10fi  %.2e\n",
                r.N, real(r.ω0), imag(r.ω0), real(r.ω1), imag(r.ω1), r.residual)
    end

    # Paper comparison
    row = findfirst(t -> abs(t[1] - a) < 0.01, TABLE_I)
    if row !== nothing
        _, _, _, polar_re, polar_im = TABLE_I[row]
        @printf("\nPaper (all sources): ω⁽¹⁾ = %+.6f %+.6fi\n", polar_re, polar_im)
    end
    println("=" ^ 70); flush(stdout)

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    a = parse(Float64, get(ARGS, 1, "0.3"))
    N_range = [4, 6, 8]
    println("Julia threads: $(Threads.nthreads())"); flush(stdout)
    sweep_sgb(a, N_range, 2; parity=:polar)
end
