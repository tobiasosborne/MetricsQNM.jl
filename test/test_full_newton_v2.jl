#!/usr/bin/env julia
"""
Test: Paper's ACTUAL Newton-Raphson — iterate on (v, ω) jointly.

Key difference from test_full_newton.jl:
- σ_min Newton iterates on ω only → spurious eigenvalues steal convergence
- Full Newton iterates on (v, ω) jointly → converges to mode nearest (v₀, ω₀)
"""

using Printf
using LinearAlgebra

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MetricsQNM
using MetricsQNM: build_matched_sgb_system, DenomSig,
                  solve_qep_full_newton, Dtilde

function main()
    a = 0.3
    m = 2
    ζ = 0.01

    println("=" ^ 70)
    println("TEST: Full Newton-Raphson on (v, ω) — paper's algorithm")
    @printf("  a=%.2f, m=%d, ζ=%.4f\n", a, m, ζ)
    println("=" ^ 70)
    flush(stdout)

    ω_leaver = leaver_qnm(a; s=-2, l=2, m=2, n=0)
    @printf("\nLeaver ω₀ = %.14f %+.14fi\n\n", real(ω_leaver), imag(ω_leaver))
    flush(stdout)

    # Start with a moderate N
    for N in [12, 14, 16, 18]
        @printf(">>> N=%d\n", N)
        flush(stdout)

        # Build matched system
        t0 = time()
        local sys_gr, sys_corr
        try
            sys_gr, sys_corr, _, _ =
                build_matched_sgb_system(a, N, m; max_a_order=2, verbose=false)
        catch e
            @printf("  BUILD FAILED: %s\n\n", string(e)[1:min(80,end)])
            flush(stdout)
            continue
        end
        build_time = time() - t0
        @printf("  Built in %.1fs. Size: %d × %d\n", build_time, size(sys_gr.D0)...)
        flush(stdout)

        # Combined system
        sys_full = METRICSSystem(
            sys_gr.D0 .+ ζ .* sys_corr.D0,
            sys_gr.D1 .+ ζ .* sys_corr.D1,
            sys_gr.D2 .+ ζ .* sys_corr.D2,
            N, m, a)

        # Step A: Get initial (ω₀, v₀) from GR system at ζ=0
        # Use σ_min Newton on the GR-only system (GR clearing, no inflation)
        # But we have the matched system... use Leaver as ω and construct v from SVD
        @printf("  Getting initial v₀ from D̃(ω_Leaver) SVD...\n")
        flush(stdout)

        P_init = Dtilde(sys_full, ComplexF64(ω_leaver))
        F_init = svd(P_init)
        v_init = conj(F_init.Vt[end, :])
        σ_init = F_init.S[end] / F_init.S[1]
        @printf("  σ_min/σ_max at ω_Leaver = %.2e\n", σ_init)
        flush(stdout)

        # Step B: Full Newton on combined system
        @printf("  Running full Newton-Raphson on (v, ω)...\n")
        flush(stdout)
        t0 = time()
        result = solve_qep_full_newton(sys_full, ω_leaver, v_init;
                                        parity=:polar, max_iter=30,
                                        tol=1e-12, verbose=true)
        newton_time = time() - t0

        ω_result = result.ω
        converged = result.converged
        Δω = ω_result - ω_leaver

        # Check residual
        res = norm(Dtilde(sys_full, ω_result) * result.v) / norm(result.v)
        physical = abs(imag(ω_result)) < 1.0 && real(ω_result) > 0

        if converged && physical
            @printf("  CONVERGED in %d iters, %.1fs\n", result.iters, newton_time)
            @printf("  ω = %.14f %+.14fi\n", real(ω_result), imag(ω_result))
            @printf("  |D̃·v|/|v| = %.2e\n", res)
            ω1_est = Δω / ζ
            @printf("  ω₁ ≈ %.10f %+.10fi  (|ω₁|=%.4f)\n",
                    real(ω1_est), imag(ω1_est), abs(ω1_est))
        elseif converged
            @printf("  CONVERGED but SPURIOUS: ω = %.4f %+.4fi\n",
                    real(ω_result), imag(ω_result))
        else
            @printf("  NOT CONVERGED after %d iters (%.1fs)\n", result.iters, newton_time)
            @printf("  ω = %.6f %+.6fi, |D̃·v|/|v| = %.2e\n",
                    real(ω_result), imag(ω_result), res)
        end
        println()
        flush(stdout)
    end

    println("=" ^ 70)
    println("DONE")
    println("=" ^ 70)
end

main()
