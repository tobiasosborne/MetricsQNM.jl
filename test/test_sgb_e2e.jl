#!/usr/bin/env julia
# End-to-end test: sGB eigenvalue perturbation theory
# Compare ω⁽¹⁾ to Tables I–III of arxiv:2406.11986

using MetricsQNM
using LinearAlgebra
using Printf

# ═══════════════════════════════════════════════════════════════════════════════
#  Ground truth from Table I (nlm = 022, polar and axial)
# ═══════════════════════════════════════════════════════════════════════════════

const TABLE_I = [
    # a      axial_re     axial_im     polar_re     polar_im
    (0.005,  0.05984,     0.00719,    -0.22664,    -0.07525),
    (0.1,    0.07282,     0.01353,    -0.26228,    -0.07146),
    (0.3,    0.09087,     0.00782,    -0.36419,    -0.04236),
]

function test_single(a::Float64, N::Int, m::Int; parity::Symbol=:polar, verbose::Bool=true)
    println("=" ^ 70)
    @printf("sGB perturbation: a=%.3f, N=%d, m=%d, parity=%s\n", a, N, m, parity)
    println("=" ^ 70); flush(stdout)

    # Step 1: GR system + QEP solve (production method)
    println("\n[1/4] Building GR system (N=$N)..."); flush(stdout)
    t0 = time()
    sys_gr, gr_norm_factors = build_system_bespoke(a, N, m; verbose=false)
    @printf("  Built: %.1fs  size=%s\n", time() - t0, size(sys_gr.D0)); flush(stdout)

    ω_leaver = leaver_qnm(a; l=2, m=m, n=0)
    @printf("  Leaver reference: ω = %.10f %+.10fi\n", real(ω_leaver), imag(ω_leaver))
    flush(stdout)

    println("\n[2/4] QEP eigensolver + eigenvector extraction..."); flush(stdout)
    t0 = time()
    result = solve_qep_with_vectors(sys_gr; ω₀=ω_leaver, refine=1)
    evals = result.eigenvalues
    evecs = result.eigenvectors

    # Find eigenvalue closest to Leaver reference
    physical = [(i, e) for (i, e) in enumerate(evals)
                if isfinite(e) && imag(e) < 0 && abs(e) < 20]
    if isempty(physical)
        error("No physical eigenvalues found!")
    end
    best_i, ω0 = physical[argmin([abs(e - ω_leaver) for (_, e) in physical])]
    v0_raw = evecs[:, best_i]

    @printf("  ω⁽⁰⁾_QEP = %.10f %+.10fi\n", real(ω0), imag(ω0))
    @printf("  |Δω| from Leaver: %.2e\n", abs(ω0 - ω_leaver))

    # Parity check
    detected_parity = classify_parity(v0_raw, N)
    @printf("  Detected parity: %s (requested: %s)\n", detected_parity, parity)

    # Normalize eigenvector: pin one coefficient to 1
    bs = (N + 1)^2
    am = abs(m)
    pinned_k = parity == :polar ? 1 : 5
    pinned_idx = (pinned_k - 1) * bs + nl_index(0, am, N, m)
    v0 = v0_raw / v0_raw[pinned_idx]

    # Verify D̃(ω₀)·v₀ ≈ 0
    residual = norm(MetricsQNM.Dtilde(sys_gr, ComplexF64(ω0)) * v0)
    @printf("  ‖D̃(ω₀)·v₀‖ = %.2e\n", residual)

    # Compute Jacobian
    J, free_idx, _ = compute_jacobian(sys_gr, ComplexF64(ω0), v0; parity=parity)
    @printf("  J size: %s, cond ≈ %.1e\n", size(J), cond(J))
    @printf("  Done: %.1fs\n", time() - t0); flush(stdout)

    # Step 3: Build D̃⁽¹⁾ correction system via exact SparsePoly extraction
    println("\n[3/4] Building sGB correction D̃⁽¹⁾ (exact SparsePoly)..."); flush(stdout)
    t0 = time()
    sys_corr = build_sgb_system_bespoke(a, N, m; verbose=true)
    normalize_system!(sys_corr, gr_norm_factors)  # Apply GR normalization to D̃⁽¹⁾
    @printf("  Correction system: %.1fs  size=%s\n", time() - t0, size(sys_corr.D0))
    flush(stdout)

    # Diagnostics
    @printf("  ‖D₀_corr‖ = %.4e\n", norm(sys_corr.D0))
    @printf("  ‖D₁_corr‖ = %.4e\n", norm(sys_corr.D1))
    @printf("  ‖D₂_corr‖ = %.4e\n", norm(sys_corr.D2))
    flush(stdout)

    # Step 4: Perturbation solve
    println("\n[4/4] Perturbation solve:"); flush(stdout)
    ω1 = solve_sgb_perturbation(sys_corr, ComplexF64(ω0), v0, J)
    @printf("  ω⁽¹⁾ = %.6f %+.6fi\n", real(ω1), imag(ω1)); flush(stdout)

    # Compare to paper
    row = findfirst(t -> abs(t[1] - a) < 0.01, TABLE_I)
    if row !== nothing
        _, axial_re, axial_im, polar_re, polar_im = TABLE_I[row]
        if parity == :polar
            ω1_paper = complex(polar_re, polar_im)
        else
            ω1_paper = complex(axial_re, axial_im)
        end
        @printf("  Paper:  ω⁽¹⁾ = %.6f %+.6fi\n", real(ω1_paper), imag(ω1_paper))
        @printf("  |Δω⁽¹⁾| = %.4e\n", abs(ω1 - ω1_paper))
        @printf("  Relative error: %.2f%%\n", 100 * abs(ω1 - ω1_paper) / abs(ω1_paper))
    end
    flush(stdout)

    return (a=a, parity=parity, ω0=ω0, ω1=ω1, v0=v0, J=J, sys_corr=sys_corr)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    N = parse(Int, get(ARGS, 1, "8"))
    a = parse(Float64, get(ARGS, 2, "0.3"))

    result = test_single(a, N, 2; parity=:polar, verbose=true)

    println("\n" * "=" ^ 70)
    println("DONE")
    println("=" ^ 70)
end
