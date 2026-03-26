# N-convergence study for sGB ω₁ (Source 1 only)
# Calls high-level pipeline functions per-N. No internal code duplication.

using MetricsQNM
using MetricsQNM: build_system_bespoke_sgb, build_sgb_Dtilde1,
                  solve_qep_with_vectors, solve_qep_newton,
                  compute_jacobian, solve_sgb_perturbation,
                  normalize_system!, sgb_clearing_targets
using LinearAlgebra
using Printf

function run_convergence(;
        a_spin::Float64 = 0.3,
        m_mode::Int = 2,
        max_a_order::Int = 2,
        N_values::Vector{Int} = [12, 16, 20, 24, 28])

    println("=" ^ 80)
    println("sGB ω₁ N-convergence study")
    println("a=$a_spin, m=$m_mode, max_a_order=$max_a_order, N=$N_values")
    println("=" ^ 80)
    flush(stdout)

    # Compute matching clearing targets (D̃⁽⁰⁾ and D̃⁽¹⁾ must use same clearing)
    println("Computing sGB clearing targets (one-time, ~85s)...")
    flush(stdout)
    t0 = time()
    P, Q, S = sgb_clearing_targets(; verbose=true)
    @printf("Clearing: P=%d Q=%d S=%d  (%.1fs)\n", P, Q, S, time() - t0)
    flush(stdout)

    # Leaver reference
    t0 = time()
    ω_L = leaver_qnm(a_spin; s=-2, l=2, m=m_mode, n=0)
    @printf("Leaver ω₀ = %.15f %+.15fi  (%.2fs)\n\n", real(ω_L), imag(ω_L), time() - t0)
    flush(stdout)

    results = NamedTuple[]

    for N in N_values
        println("=" ^ 80)
        @printf("N = %d\n", N)
        println("-" ^ 80)
        flush(stdout)

        # --- GR system (per-a-order, skip verification after first) ---
        do_verify = (N == N_values[1])
        t_gr = time()
        sys_gr, nf_gr, disc = build_system_bespoke_sgb(a_spin, N, m_mode;
                                                         P, Q, S,
                                                         verify=do_verify, verbose=true)
        t_gr = time() - t_gr
        @printf("  GR total: %.2fs", t_gr)
        do_verify && @printf(", disc=%.2e", disc)
        println()
        flush(stdout)

        # --- Eigenvalue: full QEP for first N, Newton from previous for rest ---
        t_qep = time()
        if isempty(results)
            # First N: full QEP (no good starting point for Newton)
            qep = solve_qep_with_vectors(sys_gr; ω₀=ω_L)
            evals = qep.eigenvalues
            best_idx = 0; best_dist = Inf
            for i in eachindex(evals)
                e = evals[i]
                (!isfinite(e) || imag(e) ≥ 0) && continue
                d = abs(e - ω_L)
                d < best_dist && (best_dist = d; best_idx = i)
            end
            if best_idx == 0
                @printf("  WARNING: no physical eigenvalue at N=%d, skipping\n", N)
                flush(stdout); continue
            end
            ω0 = evals[best_idx]
            v0 = qep.eigenvectors[:, best_idx]
            method = "QEP"
        else
            # Subsequent N: Newton from previous ω₀ (smooth in N)
            ω_prev = results[end].ω0
            ω0 = solve_qep_newton(sys_gr, ω_prev)
            P_ω = sys_gr.D0 + ω0 * sys_gr.D1 + ω0^2 * sys_gr.D2
            F = svd(P_ω)
            v0 = conj(F.V[:, end])
            method = "Newton"
        end
        t_qep = time() - t_qep
        Δω0 = abs(ω0 - ω_L)
        @printf("  %s: ω₀ = %.15f %+.15fi  |Δ_L|=%.2e  (%.2fs)\n",
                method, real(ω0), imag(ω0), Δω0, t_qep)
        flush(stdout)

        # --- Jacobian ---
        t_jac = time()
        J, _, _ = compute_jacobian(sys_gr, ω0, v0)
        cJ = cond(J)
        t_jac = time() - t_jac
        @printf("  Jacobian: cond=%.2e  (%.2fs)\n", cJ, t_jac)
        flush(stdout)

        # --- sGB D̃⁽¹⁾ (uses GR norm factors for row consistency) ---
        t_sgb = time()
        sys_corr = build_sgb_Dtilde1(a_spin, N, m_mode;
                                       norm_factors=nf_gr, P, Q, S,
                                       max_a_order=max_a_order, verbose=true)
        t_sgb = time() - t_sgb
        @printf("  sGB total: %.2fs\n", t_sgb)
        flush(stdout)

        # --- Perturbation ---
        t_pert = time()
        ω1 = solve_sgb_perturbation(sys_corr, ω0, v0, J)
        t_pert = time() - t_pert
        @printf("  ω₁ = %.8e %+.8ei  |ω₁|=%.8e  (%.2fs)\n",
                real(ω1), imag(ω1), abs(ω1), t_pert)
        flush(stdout)

        t_total = t_gr + t_qep + t_jac + t_sgb + t_pert
        @printf("  TOTAL for N=%d: %.1fs\n", N, t_total)
        flush(stdout)

        push!(results, (N=N, ω0=ω0, ω1=ω1, Δω0=Δω0,
                        t_gr=t_gr, t_sgb=t_sgb, t_total=t_total, cond_J=cJ))

        # Running table
        println()
        @printf("  %3s | %24s | %24s | %12s | %10s\n",
                "N", "Re(ω₁)", "Im(ω₁)", "|ω₁|", "|Δω₀_L|")
        println("  " * "-" ^ 82)
        for r in results
            @printf("  %3d | %+24.16e | %+24.16e | %.6e | %.2e\n",
                    r.N, real(r.ω1), imag(r.ω1), abs(r.ω1), r.Δω0)
        end
        flush(stdout)
    end

    # Final summary
    println("\n" * "=" ^ 110)
    println("FINAL CONVERGENCE TABLE  (a=$a_spin, m=$m_mode, max_a_order=$max_a_order)")
    println("=" ^ 110)
    @printf("%-3s | %-24s | %-24s | %-12s | %-10s | %-10s | %-6s\n",
            "N", "Re(ω₁)", "Im(ω₁)", "|ω₁|", "|Δω₀_L|", "cond(J)", "t_tot")
    println("-" ^ 110)
    for r in results
        @printf("%3d | %+24.16e | %+24.16e | %.6e | %.2e | %.2e | %5.1fs\n",
                r.N, real(r.ω1), imag(r.ω1), abs(r.ω1), r.Δω0, r.cond_J, r.t_total)
    end
    flush(stdout)

    if length(results) >= 2
        println("\nSUCCESSIVE DIFFERENCES:")
        for i in 2:length(results)
            Δ = abs(results[i].ω1 - results[i-1].ω1)
            @printf("  N=%d→%d: |Δω₁| = %.4e", results[i-1].N, results[i].N, Δ)
            if i >= 3
                Δ_prev = abs(results[i-1].ω1 - results[i-2].ω1)
                ratio = Δ / max(Δ_prev, 1e-30)
                @printf("  ratio=%.3f", ratio)
            end
            println()
        end
        flush(stdout)
    end

    return results
end

results = run_convergence()
