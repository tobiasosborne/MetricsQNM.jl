#!/usr/bin/env julia
"""
Test: Paper's actual algorithm.

1. Matched per-equation clearing for D̃⁰ and D̃¹
2. Newton on the FULL combined system D̃⁰ + ζ·D̃¹
3. Scan N values, pick optimal via backward modulus difference

This IS what the paper does (arxiv:2406.11986, Section IV).
"""

using Printf
using LinearAlgebra

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MetricsQNM
using MetricsQNM: build_matched_sgb_system, DenomSig

function solve_full_newton(sys_gr, sys_corr, ζ, ω_init; max_iter=30)
    # Combined system: D̃(ω) = (D̃⁰ + ζD̃¹)
    D0 = sys_gr.D0 .+ ζ .* sys_corr.D0
    D1 = sys_gr.D1 .+ ζ .* sys_corr.D1
    D2 = sys_gr.D2 .+ ζ .* sys_corr.D2
    return solve_qep_newton(D0, D1, D2, ComplexF64(ω_init); max_iter)
end

function full_residual(sys_gr, sys_corr, ζ, ω)
    D0 = sys_gr.D0 .+ ζ .* sys_corr.D0
    D1 = sys_gr.D1 .+ ζ .* sys_corr.D1
    D2 = sys_gr.D2 .+ ζ .* sys_corr.D2
    P = D0 .+ ω .* D1 .+ ω^2 .* D2
    svals = svdvals(P)
    return svals[end] / svals[1]
end

function main()
    a = 0.3
    m = 2
    ζ = 0.01  # small coupling for perturbative regime

    println("=" ^ 70)
    println("TEST: Paper's algorithm — Newton on full D̃⁰ + ζ·D̃¹")
    @printf("  a=%.2f, m=%d, ζ=%.4f\n", a, m, ζ)
    println("=" ^ 70)
    flush(stdout)

    # Leaver reference (GR)
    ω_leaver = leaver_qnm(a; s=-2, l=2, m=2, n=0)
    @printf("\nLeaver ω₀ = %.14f %+.14fi\n", real(ω_leaver), imag(ω_leaver))
    flush(stdout)

    # Scan N values (paper does N=1 to 25, we do a representative range)
    N_values = [8, 10, 12, 14, 16, 18, 20]

    println("\n>>> Scanning N values with matched clearing + full Newton\n")
    @printf("  %-4s | %-28s | %-10s | %-10s | %-8s\n",
            "N", "ω(ζ)", "residual", "|Δω|", "time")
    println("  " * "-"^80)
    flush(stdout)

    results = []

    for N in N_values
        @printf("  N=%2d | ", N)
        flush(stdout)

        t0 = time()

        # Build matched system
        local sys_gr, sys_corr
        try
            sys_gr, sys_corr, _, _ =
                build_matched_sgb_system(a, N, m; max_a_order=2, verbose=false)
        catch e
            @printf("BUILD FAILED: %s\n", string(e)[1:min(60,end)])
            flush(stdout)
            continue
        end

        build_time = time() - t0

        # Newton on full system from Leaver initial guess
        t0 = time()
        ω_full = solve_full_newton(sys_gr, sys_corr, ζ, ω_leaver; max_iter=50)
        newton_time = time() - t0
        total_time = build_time + newton_time

        res = full_residual(sys_gr, sys_corr, ζ, ω_full)
        Δω = ω_full - ω_leaver

        converged = res < 1e-10
        physical = abs(imag(ω_full)) < 1.0 && real(ω_full) > 0

        if converged && physical
            @printf("%.12f %+.12fi | %.2e | %.2e | %.1fs\n",
                    real(ω_full), imag(ω_full), res, abs(Δω), total_time)
            push!(results, (N=N, ω=ω_full, res=res, Δω=Δω))
        elseif converged && !physical
            @printf("SPURIOUS: %.4f %+.4fi | %.2e | %.2e | %.1fs\n",
                    real(ω_full), imag(ω_full), res, abs(Δω), total_time)
        else
            @printf("NOT CONVERGED | %.2e | %.2e | %.1fs\n",
                    res, abs(Δω), total_time)
        end
        flush(stdout)
    end

    # ─── Backward modulus difference (paper's convergence criterion) ───
    if length(results) >= 2
        println("\n>>> Backward modulus difference B(N):\n")
        for i in 2:length(results)
            N_prev = results[i-1].N
            N_curr = results[i].N
            B = abs(results[i].ω - results[i-1].ω)
            @printf("  B(N=%d) = |ω(%d) - ω(%d)| = %.2e\n", N_curr, N_curr, N_prev, B)
        end

        # Extract ω₁ from the converged result
        best = results[end]
        ω1_numerical = (best.ω - ω_leaver) / ζ
        println("\n>>> Extracted sGB correction:")
        @printf("  ω(ζ) = %.14f %+.14fi\n", real(best.ω), imag(best.ω))
        @printf("  ω₁ ≈ (ω(ζ) - ω₀)/ζ = %.10f %+.10fi\n", real(ω1_numerical), imag(ω1_numerical))
        @printf("  |ω₁| = %.6f\n", abs(ω1_numerical))

        println("\n  Reference check:")
        if abs(ω1_numerical) > 100
            println("    *** |ω₁| >> 1, unphysical ***")
        elseif abs(ω1_numerical) < 1e-6
            println("    *** |ω₁| ≈ 0, suspicious ***")
        else
            @printf("    |ω₁| = %.4f — plausible!\n", abs(ω1_numerical))
        end
    else
        println("\n  Not enough converged results for analysis.")
    end

    flush(stdout)
    println("\n" * "=" ^ 70)
    println("DONE")
    println("=" ^ 70)
end

main()
