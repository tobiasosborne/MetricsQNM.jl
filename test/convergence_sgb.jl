#!/usr/bin/env julia
# N-convergence study for sGB ω₁ (Source 1 only).
#
# Runs Phases 1-3 ONCE (expensive), then sweeps N in Phases 4-5 + GR + perturbation.
# Post-cleanup-fix: per-order verification should now be ~1e-14.

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using MetricsQNM
using LinearAlgebra
using Printf
using Symbolics

import MetricsQNM: load_H_ratpolys_per_order, verify_H_ratpolys_per_order,
    extract_sgb_coefficients_symbolic, combine_sgb_K,
    SymToPolyCtx, PDECoefficients, METRICSSystem,
    _r_to_z, assemble_system, spectral_basis,
    r_plus, nl_index, normalize_system!

function run_convergence(; a=0.3, m=2, epsilon=1e-2, parity=:polar,
                           N_values=[10, 12, 15, 18, 20, 22, 25])

    println("sGB N-convergence study")
    println("  a=$a, m=$m, epsilon=$epsilon, parity=$parity")
    println("  N values: $N_values")
    println("  Julia threads: $(Threads.nthreads())")
    println("=" ^ 70); flush(stdout)

    # ── Phase 1: Load H_i per-a-order ──
    println("\n[Phase 1] Loading H_i per-a-order..."); flush(stdout)
    t0 = time()
    H_per_order, a_powers = load_H_ratpolys_per_order(; verbose=true)
    @printf("  Phase 1: %.1fs\n", time() - t0); flush(stdout)

    err = verify_H_ratpolys_per_order(H_per_order, a_powers, a; verbose=true)
    @printf("  Verification: max rel err = %.2e %s\n", err,
            err < 1e-10 ? "✓" : "✗ WARNING"); flush(stdout)

    # Truncate
    n_orders_total = length(a_powers)
    n_a_max = n_orders_total
    for (oi, a_pow) in enumerate(a_powers)
        if a_pow > 0 && abs(a)^a_pow < epsilon
            n_a_max = oi - 1
            break
        end
    end
    @printf("  Retaining %d/%d a-orders\n", n_a_max, n_orders_total); flush(stdout)

    # ── Phase 2: Extract c_{k,d,p} ──
    println("\n[Phase 2] Extracting c_{k,d,p}..."); flush(stdout)
    t0 = time()
    coeffs, h_map, n_eqs = extract_sgb_coefficients_symbolic(a; verbose=true)
    @printf("  Phase 2: %.1fs\n", time() - t0); flush(stdout)

    # ── Phase 3: combine per a-order ──
    println("\n[Phase 3] Combining c_{k,d,p} × H_p..."); flush(stdout)
    @variables r_v chi_v omega_re_v omega_im_v iu_v
    ctx = SymToPolyCtx(Num[r_v, chi_v, omega_re_v, omega_im_v, iu_v], a)

    t0 = time()
    K_r_per_order = Vector{NTuple{3, PDECoefficients}}(undef, n_a_max)
    for ao = 1:n_a_max
        t_ao = time()
        K0, K1, K2 = combine_sgb_K(coeffs, H_per_order[ao], h_map, n_eqs, ctx; verbose=false)
        K_r_per_order[ao] = (K0, K1, K2)

        d_ao, s_ao = 0, 0
        for K in (K0, K1, K2), k in 1:n_eqs
            for ((γ, δ, σ, α, β, j), _) in K.equations[k]
                d_ao = max(d_ao, δ); s_ao = max(s_ao, σ)
            end
        end
        n_terms = sum(length(K.equations[k]) for K in (K0, K1, K2) for k in 1:n_eqs)
        @printf("  a^%d: d_max=%d, s_max=%d, %d terms, %.1fs\n",
                a_powers[ao], d_ao, s_ao, n_terms, time() - t_ao); flush(stdout)
    end
    @printf("  Phase 3 total: %.1fs\n", time() - t0); flush(stdout)

    # Shared d_max, s_max
    d_max_shared, s_max_shared = 0, 0
    for ao = 1:n_a_max, K in K_r_per_order[ao], k in 1:n_eqs
        for ((γ, δ, σ, α, β, j), _) in K.equations[k]
            d_max_shared = max(d_max_shared, δ)
            s_max_shared = max(s_max_shared, σ)
        end
    end
    @printf("  Shared d_max=%d, s_max=%d\n", d_max_shared, s_max_shared); flush(stdout)

    rp = r_plus(a)
    ω_leaver = leaver_qnm(a; l=2, m=m, n=0)
    @printf("\n  Leaver ω⁰ = %.14f %+.14fi\n", real(ω_leaver), imag(ω_leaver))
    ω1_paper = complex(-0.36419, -0.04236)

    # ── Sweep N ──
    println("\n" * "=" ^ 80)
    println("N-CONVERGENCE SWEEP (d_max=$d_max_shared, epsilon=$epsilon, $(n_a_max) a-orders)")
    println("=" ^ 80)
    @printf("%-4s  %22s  %10s  %22s  %10s  %8s\n",
            "N", "ω⁰ (QEP)", "|Δω⁰|", "ω₁ (Source 1)", "|ω₁|", "cond(J)")
    println("-" ^ 80); flush(stdout)

    results = NamedTuple[]

    for N in N_values
        t_N = time()

        # Phase 4-5: r→z + assembly
        basis = spectral_basis(N, m; max_delta=d_max_shared,
                               max_sigma=max(s_max_shared, 25))
        bs = (N + 1)^2

        D0_t = zeros(ComplexF64, 10 * bs, 6 * bs)
        D1_t = zeros(ComplexF64, 10 * bs, 6 * bs)
        D2_t = zeros(ComplexF64, 10 * bs, 6 * bs)

        for ao = 1:n_a_max
            K0_r, K1_r, K2_r = K_r_per_order[ao]
            K0_z = _r_to_z(K0_r, rp, d_max_shared)
            K1_z = _r_to_z(K1_r, rp, d_max_shared)
            K2_z = _r_to_z(K2_r, rp, d_max_shared)

            D0_ao = assemble_system(K0_z, basis, a).D0
            D1_ao = assemble_system(K1_z, basis, a).D0
            D2_ao = assemble_system(K2_z, basis, a).D0

            w = a_powers[ao] == 0 ? 1.0 : a^a_powers[ao]
            D0_t .+= w .* D0_ao
            D1_t .+= w .* D1_ao
            D2_t .+= w .* D2_ao
        end

        sys_corr = METRICSSystem(D0_t, D1_t, D2_t, N, m, a)

        # GR
        sys_gr, gr_nf = build_system_bespoke(a, N, m;
                                              d_max_override=d_max_shared, verbose=false)
        result = solve_qep_with_vectors(sys_gr; ω₀=ω_leaver, refine=1)
        evals, evecs = result.eigenvalues, result.eigenvectors

        physical = [(i, e) for (i, e) in enumerate(evals)
                    if isfinite(e) && imag(e) < 0 && abs(e) < 20]
        if isempty(physical)
            @printf("%-4d  NO PHYSICAL EIGENVALUE\n", N); flush(stdout)
            continue
        end

        best_i, ω0 = physical[argmin([abs(e - ω_leaver) for (_, e) in physical])]
        v0_raw = evecs[:, best_i]

        pinned_k = parity == :polar ? 1 : 5
        pinned_idx = (pinned_k - 1) * bs + nl_index(0, abs(m), N, m)
        v0 = v0_raw / v0_raw[pinned_idx]

        J, _, _ = compute_jacobian(sys_gr, ComplexF64(ω0), v0; parity=parity)
        cJ = cond(J)

        normalize_system!(sys_corr, gr_nf)
        ω1 = solve_sgb_perturbation(sys_corr, ComplexF64(ω0), v0, J)
        Δω0 = abs(ω0 - ω_leaver)

        @printf("%-4d  %+.10f%+.10fi  %.2e  %+.6f%+.6fi  %.4e  %.1e  (%.1fs)\n",
                N, real(ω0), imag(ω0), Δω0,
                real(ω1), imag(ω1), abs(ω1), cJ, time() - t_N)
        flush(stdout)

        push!(results, (N=N, ω0=ω0, ω1=ω1, Δω0=Δω0, condJ=cJ))
    end

    # ── Summary ──
    println("\n" * "=" ^ 80)
    println("SUMMARY")
    println("=" ^ 80)
    println("Paper (all 4 sources, polar, a=0.3): ω₁ = $(ω1_paper)")
    println("Our result (Source 1 only):\n")
    @printf("%-4s  %12s  %12s  %12s  %10s\n", "N", "Re(ω₁)", "Im(ω₁)", "|Δω⁰|", "cond(J)")
    println("-" ^ 56)
    for r in results
        @printf("%-4d  %+12.6f  %+12.6f  %12.2e  %10.1e\n",
                r.N, real(r.ω1), imag(r.ω1), r.Δω0, r.condJ)
    end
    println("-" ^ 56)

    if length(results) >= 2
        println("\nConsecutive Δω₁ (convergence):")
        for i in 2:length(results)
            Δ = abs(results[i].ω1 - results[i-1].ω1)
            @printf("  N=%d→%d: |Δω₁| = %.4e\n", results[i-1].N, results[i].N, Δ)
        end
    end

    println("\nDone.")
    return results
end

run_convergence()
