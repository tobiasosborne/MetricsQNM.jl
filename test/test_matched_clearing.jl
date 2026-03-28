#!/usr/bin/env julia
"""
Test: Matched per-equation clearing for D̃⁰ and D̃¹.
Both use the same per-equation (1+z)^{d_max_k} gauge.
"""

using Printf
using LinearAlgebra

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MetricsQNM
using MetricsQNM: build_matched_sgb_system, DenomSig

function main()
    a = 0.3
    m = 2
    println("=" ^ 70)
    println("TEST: Matched per-equation clearing")
    @printf("  a=%.2f, m=%d\n", a, m)
    println("=" ^ 70)
    flush(stdout)

    # ─── Build matched system ──────────────────────────────────────────
    t_total = time()
    sys_gr, sys_corr, nf, matched =
        build_matched_sgb_system(a, 18, m; max_a_order=2, verbose=true)
    N = sys_gr.N
    @printf("\nTotal build time: %.1fs\n", time() - t_total)
    @printf("D̃⁰ size: %d × %d, D̃¹ size: %d × %d\n",
            size(sys_gr.D0)..., size(sys_corr.D0)...)
    flush(stdout)

    # ─── Leaver reference ──────────────────────────────────────────────
    println("\n>>> Leaver reference")
    ω_leaver = leaver_qnm(a; s=-2, l=2, m=2, n=0)
    @printf("  ω_Leaver = %.14f %+.14fi\n", real(ω_leaver), imag(ω_leaver))
    flush(stdout)

    # ─── Newton on D̃⁰ ────────────────────────────────────────────────
    println("\n>>> Newton σ_min on D̃⁰")
    flush(stdout)
    t0 = time()
    ω0 = solve_qep_newton(sys_gr, ω_leaver; max_iter=30)
    res = qep_residual(sys_gr, ω0)
    @printf("  ω₀ = %.14f %+.14fi\n", real(ω0), imag(ω0))
    @printf("  Residual: %.2e\n", res)
    @printf("  |ω₀ - ω_Leaver| = %.2e\n", abs(ω0 - ω_leaver))
    @printf("  Newton: %.3fs\n", time() - t0)
    flush(stdout)

    if res > 1e-10
        println("  *** Newton did NOT converge! ***")
        println("  Trying with more iterations...")
        ω0 = solve_qep_newton(sys_gr, ω_leaver; max_iter=100)
        res = qep_residual(sys_gr, ω0)
        @printf("  ω₀ = %.14f %+.14fi\n", real(ω0), imag(ω0))
        @printf("  Residual: %.2e\n", res)
    end

    # ─── Eigenvector v₀ ───────────────────────────────────────────────
    println("\n>>> Eigenvector v₀")
    P_at_ω0 = sys_gr.D0 .+ ω0 .* sys_gr.D1 .+ ω0^2 .* sys_gr.D2
    F = svd(P_at_ω0)
    v0 = conj(F.Vt[end, :])
    @printf("  σ_min = %.2e, σ_max = %.2e\n", F.S[end], F.S[1])
    @printf("  ‖v₀‖ = %.6f\n", norm(v0))
    flush(stdout)

    # ─── Jacobian ─────────────────────────────────────────────────────
    println("\n>>> Jacobian")
    J, free_idx, pinned_idx = compute_jacobian(sys_gr, ω0, v0; parity=:polar)
    @printf("  J: %d × %d, cond = %.2e\n", size(J)..., cond(J))
    flush(stdout)

    # ─── Eigenvalue perturbation ──────────────────────────────────────
    println("\n>>> Eigenvalue perturbation")
    flush(stdout)

    source = (sys_corr.D0 + ω0 * sys_corr.D1 + ω0^2 * sys_corr.D2) * v0
    gr_check = (sys_gr.D0 + ω0 * sys_gr.D1 + ω0^2 * sys_gr.D2) * v0

    @printf("  ‖D̃¹(ω₀)·v₀‖ = %.6e (source)\n", norm(source))
    @printf("  ‖D̃⁰(ω₀)·v₀‖ = %.6e (should ≈ 0)\n", norm(gr_check))

    ω1 = solve_sgb_perturbation(sys_corr, ω0, v0, J)
    @printf("  ω⁽¹⁾ = %.10f %+.10fi\n", real(ω1), imag(ω1))
    @printf("  |ω⁽¹⁾| = %.6e\n", abs(ω1))

    x1 = -(pinv(J) * source)
    pinv_res = norm(J * x1 + source) / norm(source)
    @printf("  pinv residual = %.4e\n", pinv_res)

    println("\n  Reference check:")
    if abs(ω1) > 100
        println("    *** |ω₁| >> 1, unphysical ***")
    elseif abs(ω1) < 1e-6
        println("    *** |ω₁| ≈ 0, suspicious ***")
    else
        @printf("    |ω₁| = %.6f — plausible!\n", abs(ω1))
    end
    flush(stdout)

    println("\n" * "=" ^ 70)
    println("DONE")
    println("=" ^ 70)
end

main()
