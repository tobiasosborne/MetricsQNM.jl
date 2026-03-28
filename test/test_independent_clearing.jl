#!/usr/bin/env julia
"""
Test: D̃⁰ and D̃¹ with INDEPENDENT clearing at same N.

Strategy: D̃⁰ uses GR clearing (P=3,Q=1,S=1, d_max≈9).
          D̃¹ uses per-equation clearing (d_max≈16).
          Both built at the same N (large enough for D̃¹).

Solver strategies tested:
  A. Full QEP on D̃⁰ at large N → ω₀, v₀ (validation)
  B. Newton σ_min on D̃⁰ from Leaver initial guess → ω₀, v₀ (production)
  C. Eigenvalue perturbation: ω₁ = -J⁻¹ · D̃¹(ω₀)·v₀ (the sGB correction)
"""

using Printf
using LinearAlgebra

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MetricsQNM
using MetricsQNM: extract_sgb_correction_symbolic_a, DenomSig,
                  build_sgb_Dtilde1, build_system_bespoke_sgb,
                  normalize_system!, spectral_basis, assemble_system,
                  _r_to_z, extract_G_bespoke_symbolic_a, extract_G_bespoke,
                  PDECoefficients

function main()
    a = 0.3
    m = 2
    println("=" ^ 70)
    println("TEST: Independent clearing for D̃⁰ and D̃¹")
    @printf("  a=%.2f, m=%d\n", a, m)
    println("=" ^ 70)
    flush(stdout)

    # ─────────────────────────────────────────────────────────────────
    # Step 1: Determine d_max for D̃¹ (sGB correction)
    # ─────────────────────────────────────────────────────────────────
    println("\n>>> Step 1: Extract sGB correction coefficients")
    flush(stdout)

    t0 = time()
    c_kdp_per_a, a_powers_sgb, max_denom =
        extract_sgb_correction_symbolic_a(; P=3, Q=1, S=1, verbose=true)

    # Find d_max across all a-orders for sGB correction
    d_max_sgb = 0
    for a_exp in a_powers_sgb
        for (_, sp) in c_kdp_per_a[a_exp]
            for (e, _) in sp.terms
                d_max_sgb = max(d_max_sgb, e[1])
            end
        end
    end
    @printf("  sGB d_max (from per-eq clearing): %d\n", d_max_sgb)
    @printf("  Step 1 took %.1fs\n", time() - t0)
    flush(stdout)

    # ─────────────────────────────────────────────────────────────────
    # Step 2: Determine required N
    # ─────────────────────────────────────────────────────────────────
    # GR d_max is ~9. sGB d_max is ~16. Use N >= d_max_sgb + 1
    N = d_max_sgb + 1
    @printf("\n>>> Step 2: Using N=%d (d_max_sgb=%d)\n", N, d_max_sgb)
    flush(stdout)

    # ─────────────────────────────────────────────────────────────────
    # Step 3: Build D̃⁰ (GR) at large N with GR clearing (P=3,Q=1,S=1)
    # ─────────────────────────────────────────────────────────────────
    println("\n>>> Step 3: Build D̃⁰ (GR) at N=$N with P=3,Q=1,S=1")
    flush(stdout)

    t0 = time()
    # Use build_system_bespoke_sgb which does per-a-order extraction
    # d_max_override ensures basis is big enough for D̃¹
    sys_gr, nf, disc = build_system_bespoke_sgb(a, N, m;
        P=3, Q=1, S=1, d_max_override=d_max_sgb,
        verify=true, verbose=true)
    @printf("  D̃⁰ built. Verification discrepancy: %.2e\n", disc)
    @printf("  Step 3 took %.1fs\n", time() - t0)
    flush(stdout)

    # ─────────────────────────────────────────────────────────────────
    # Step 4A: Leaver reference value
    # ─────────────────────────────────────────────────────────────────
    println("\n>>> Step 4A: Leaver reference")
    flush(stdout)

    ω_leaver = leaver_qnm(a; s=-2, l=2, m=2, n=0)
    @printf("  ω_Leaver = %.14f %+.14fi\n", real(ω_leaver), imag(ω_leaver))
    flush(stdout)

    # ─────────────────────────────────────────────────────────────────
    # Step 4B: Newton σ_min on D̃⁰ from Leaver → ω₀
    # ─────────────────────────────────────────────────────────────────
    println("\n>>> Step 4B: Newton σ_min refinement from Leaver")
    flush(stdout)

    t0 = time()
    ω_newton = solve_qep_newton(sys_gr, ω_leaver; max_iter=30)
    res_newton = qep_residual(sys_gr, ω_newton)
    @printf("  ω_Newton = %.14f %+.14fi\n", real(ω_newton), imag(ω_newton))
    @printf("  Residual: %.2e\n", res_newton)
    @printf("  |ω_Newton - ω_Leaver| = %.2e\n", abs(ω_newton - ω_leaver))
    @printf("  Step 4B took %.3fs\n", time() - t0)
    flush(stdout)

    # ─────────────────────────────────────────────────────────────────
    # Step 4C: Full QEP on D̃⁰ (if feasible) for validation
    # ─────────────────────────────────────────────────────────────────
    bs = (N + 1)^2
    qep_size = 6 * bs  # number of columns = 2n companion size
    println("\n>>> Step 4C: Full QEP (companion size = 2×$qep_size = $(2*qep_size))")
    flush(stdout)

    if 2 * qep_size > 8000
        println("  SKIPPING: companion too large ($(2*qep_size)). Newton is production solver.")
    else
        t0 = time()
        eigs = solve_qep_svd(sys_gr; ω₀=ω_leaver, refine=1, normalize=false)
        physical = filter(e -> isfinite(e) && imag(e) < 0, eigs)
        if !isempty(physical)
            diffs = abs.(physical .- ω_leaver)
            best = physical[argmin(diffs)]
            @printf("  ω_QEP   = %.14f %+.14fi\n", real(best), imag(best))
            @printf("  |ω_QEP - ω_Leaver| = %.2e\n", abs(best - ω_leaver))
            @printf("  |ω_QEP - ω_Newton| = %.2e\n", abs(best - ω_newton))
        else
            println("  No physical eigenvalues found!")
        end
        @printf("  Step 4C took %.1fs\n", time() - t0)
    end
    flush(stdout)

    # ─────────────────────────────────────────────────────────────────
    # Step 5: Get eigenvector v₀ from Newton's final SVD
    # ─────────────────────────────────────────────────────────────────
    println("\n>>> Step 5: Extract eigenvector v₀")
    flush(stdout)

    ω0 = ω_newton  # use Newton result
    P_at_ω0 = sys_gr.D0 .+ ω0 .* sys_gr.D1 .+ ω0^2 .* sys_gr.D2
    F_svd = svd(P_at_ω0)
    v0 = F_svd.Vt[end, :] |> conj  # last right singular vector
    @printf("  σ_min(D̃⁰(ω₀)) = %.2e (should be ≈0)\n", F_svd.S[end])
    @printf("  σ_max(D̃⁰(ω₀)) = %.2e\n", F_svd.S[1])
    @printf("  ‖v₀‖ = %.6f\n", norm(v0))
    flush(stdout)

    # ─────────────────────────────────────────────────────────────────
    # Step 6: Compute Jacobian J from D̃⁰
    # ─────────────────────────────────────────────────────────────────
    println("\n>>> Step 6: Compute Jacobian J")
    flush(stdout)

    J, free_idx, pinned_idx = compute_jacobian(sys_gr, ω0, v0; parity=:polar)
    @printf("  J size: %d × %d\n", size(J)...)
    @printf("  cond(J): %.2e\n", cond(J))
    flush(stdout)

    # ─────────────────────────────────────────────────────────────────
    # Step 7: Build D̃¹ (sGB correction) at same N with per-eq clearing
    # ─────────────────────────────────────────────────────────────────
    println("\n>>> Step 7: Build D̃¹ (sGB correction) at N=$N")
    flush(stdout)

    t0 = time()
    sys_corr = build_sgb_Dtilde1(a, N, m;
        norm_factors=nf, P=3, Q=1, S=1, max_a_order=2, verbose=true)
    @printf("  D̃¹ built.\n")
    @printf("  ‖D̃¹₀‖ = %.2e, ‖D̃¹₁‖ = %.2e, ‖D̃¹₂‖ = %.2e\n",
            norm(sys_corr.D0), norm(sys_corr.D1), norm(sys_corr.D2))
    @printf("  Step 7 took %.1fs\n", time() - t0)
    flush(stdout)

    # Check dimension compatibility
    @printf("  D̃⁰ size: %d × %d\n", size(sys_gr.D0)...)
    @printf("  D̃¹ size: %d × %d\n", size(sys_corr.D0)...)
    if size(sys_gr.D0) != size(sys_corr.D0)
        println("  *** DIMENSION MISMATCH! ***")
        return
    end
    flush(stdout)

    # ─────────────────────────────────────────────────────────────────
    # Step 8: Eigenvalue perturbation → ω₁
    # ─────────────────────────────────────────────────────────────────
    println("\n>>> Step 8: Eigenvalue perturbation")
    flush(stdout)

    ω1 = solve_sgb_perturbation(sys_corr, ω0, v0, J)

    @printf("  ω⁽¹⁾ = %.10f %+.10fi\n", real(ω1), imag(ω1))
    @printf("  |ω⁽¹⁾| = %.6e\n", abs(ω1))
    flush(stdout)

    # Reference: paper Table II, a=0.3, (l=2,m=2,n=0)
    # ω₁ should be O(0.1) for the physical mode
    println("\n  Reference check:")
    println("    |ω₁| ~ O(0.1) expected from paper Table II")
    if abs(ω1) > 100
        println("    *** WARNING: |ω₁| >> 1, likely unphysical ***")
    elseif abs(ω1) < 1e-3
        println("    *** WARNING: |ω₁| ~ 0, suspicious ***")
    else
        @printf("    |ω₁| = %.4f — plausible range!\n", abs(ω1))
    end
    flush(stdout)

    # ─────────────────────────────────────────────────────────────────
    # Step 9: Source vector diagnostics
    # ─────────────────────────────────────────────────────────────────
    println("\n>>> Step 9: Source vector diagnostics")
    flush(stdout)

    source = (sys_corr.D0 + ω0 * sys_corr.D1 + ω0^2 * sys_corr.D2) * v0
    @printf("  ‖source‖ = %.6e\n", norm(source))
    @printf("  ‖D̃⁰(ω₀)·v₀‖ = %.6e (should be ≈0)\n",
            norm((sys_gr.D0 + ω0 * sys_gr.D1 + ω0^2 * sys_gr.D2) * v0))

    x1 = -(pinv(J) * source)
    @printf("  ‖x₁‖ = %.6e\n", norm(x1))
    @printf("  pinv residual ‖J·x₁ + source‖/‖source‖ = %.2e\n",
            norm(J * x1 + source) / norm(source))
    flush(stdout)

    println("\n" * "=" ^ 70)
    println("DONE")
    println("=" ^ 70)
end

main()
