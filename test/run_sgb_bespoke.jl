#!/usr/bin/env julia
# Quick runner: sGB D̃⁽¹⁾ via per-a-order exact SparsePoly pipeline + perturbation solve.
# Uses per-a-order H_i loading to avoid d_max inflation (80→~15 per order).

using MetricsQNM
using LinearAlgebra
using Printf

const TABLE_I = [
    # a      axial_re     axial_im     polar_re     polar_im
    (0.005,  0.05984,     0.00719,    -0.22664,    -0.07525),
    (0.1,    0.07282,     0.01353,    -0.26228,    -0.07146),
    (0.3,    0.09087,     0.00782,    -0.36419,    -0.04236),
]

function run_bespoke(a::Float64, N::Int, m::Int; parity::Symbol=:polar,
                     epsilon::Float64=1e-14)
    println("=" ^ 70)
    @printf("sGB per-a-order pipeline: a=%.3f, N=%d, m=%d, parity=%s\n", a, N, m, parity)
    println("=" ^ 70); flush(stdout)

    # ── Step 1: sGB D̃⁽¹⁾ (build first to determine shared d_max) ──
    println("\n[1/3] Building sGB D̃⁽¹⁾ (per-a-order exact SparsePoly)..."); flush(stdout)
    t0 = time()
    sys_corr, d_max_sgb = build_sgb_system_bespoke(a, N, m; epsilon, verbose=true)
    @printf("  D̃⁽¹⁾ built: %.1fs  d_max=%d  size=%s\n",
            time() - t0, d_max_sgb, size(sys_corr.D0)); flush(stdout)

    @printf("  ‖D₀⁽¹⁾‖ = %.4e\n", norm(sys_corr.D0))
    @printf("  ‖D₁⁽¹⁾‖ = %.4e\n", norm(sys_corr.D1))
    @printf("  ‖D₂⁽¹⁾‖ = %.4e\n", norm(sys_corr.D2)); flush(stdout)

    # ── Step 2: GR solution (with shared d_max for perturbation compatibility) ──
    println("\n[2/3] GR system + QEP solve (d_max=$d_max_sgb)..."); flush(stdout)
    t0 = time()
    sys_gr, gr_norm_factors = build_system_bespoke(a, N, m;
                                                    d_max_override=d_max_sgb,
                                                    verbose=false)
    @printf("  GR system built: %.1fs  size=%s\n", time() - t0, size(sys_gr.D0)); flush(stdout)

    ω_leaver = leaver_qnm(a; l=2, m=m, n=0)
    result = solve_qep_with_vectors(sys_gr; ω₀=ω_leaver, refine=1)
    evals, evecs = result.eigenvalues, result.eigenvectors

    physical = [(i, e) for (i, e) in enumerate(evals)
                if isfinite(e) && imag(e) < 0 && abs(e) < 20]
    isempty(physical) && error("No physical eigenvalues!")
    best_i, ω0 = physical[argmin([abs(e - ω_leaver) for (_, e) in physical])]
    v0_raw = evecs[:, best_i]

    @printf("  ω⁽⁰⁾ = %.10f %+.10fi\n", real(ω0), imag(ω0))
    @printf("  |Δω| from Leaver: %.2e\n", abs(ω0 - ω_leaver)); flush(stdout)

    # Normalize
    bs = (N + 1)^2
    am = abs(m)
    pinned_k = parity == :polar ? 1 : 5
    pinned_idx = (pinned_k - 1) * bs + nl_index(0, am, N, m)
    v0 = v0_raw / v0_raw[pinned_idx]

    residual = norm(MetricsQNM.Dtilde(sys_gr, ComplexF64(ω0)) * v0)
    @printf("  ‖D̃(ω₀)·v₀‖ = %.2e\n", residual)

    J, _, _ = compute_jacobian(sys_gr, ComplexF64(ω0), v0; parity=parity)
    @printf("  J: %s, cond ≈ %.1e\n", size(J), cond(J))
    @printf("  Step 2 total: %.1fs\n", time() - t0); flush(stdout)

    # Apply GR normalization to sGB correction
    normalize_system!(sys_corr, gr_norm_factors)

    # ── Step 3: Perturbation solve ──
    println("\n[3/3] Perturbation solve..."); flush(stdout)
    ω1 = solve_sgb_perturbation(sys_corr, ComplexF64(ω0), v0, J)
    @printf("  ω⁽¹⁾ = %.6f %+.6fi\n", real(ω1), imag(ω1)); flush(stdout)

    # Compare to paper
    row = findfirst(t -> abs(t[1] - a) < 0.01, TABLE_I)
    if row !== nothing
        _, axial_re, axial_im, polar_re, polar_im = TABLE_I[row]
        ω1_paper = parity == :polar ? complex(polar_re, polar_im) : complex(axial_re, axial_im)
        @printf("  Paper:  ω⁽¹⁾ = %.6f %+.6fi\n", real(ω1_paper), imag(ω1_paper))
        @printf("  |Δω⁽¹⁾| = %.4e\n", abs(ω1 - ω1_paper))
        @printf("  Rel err: %.2f%%\n", 100 * abs(ω1 - ω1_paper) / abs(ω1_paper))
    end
    flush(stdout)

    return (ω0=ω0, ω1=ω1, sys_corr=sys_corr, J=J, v0=v0, d_max=d_max_sgb)
end

# ── Main ──
if abspath(PROGRAM_FILE) == @__FILE__
    a = parse(Float64, get(ARGS, 1, "0.3"))
    N = parse(Int, get(ARGS, 2, "8"))

    println("Starting sGB per-a-order pipeline..."); flush(stdout)
    println("Julia threads: $(Threads.nthreads())"); flush(stdout)
    println(); flush(stdout)

    result = run_bespoke(a, N, 2; parity=:polar)

    println("\n" * "=" ^ 70)
    println("COMPLETE")
    println("=" ^ 70); flush(stdout)
end
