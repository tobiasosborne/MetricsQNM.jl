#!/usr/bin/env julia
#
# Test the full pipeline: symbolic G extraction → SVD compression QEP → QNM frequencies
#
using MetricsQNM
using MetricsQNM: Dtilde, _svd_compress_solve, _solve_square_qep
using LinearAlgebra
using Printf

println("=" ^ 70)
println("SVD COMPRESSION QEP — FULL PIPELINE TEST")
println("=" ^ 70)
flush(stdout)

# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 0: Verify solver on synthetic problem with known eigenvalues
# ═══════════════════════════════════════════════════════════════════════════════
println("\n[0] Synthetic rectangular QEP validation...")
flush(stdout)

let
    using Random
    rng = MersenneTwister(42)
    m_syn, n_syn = 60, 36  # 10:6 ratio like METRICS

    # Build square QEP with known eigenvalues
    A_sq = randn(rng, ComplexF64, n_syn, n_syn)
    B_sq = randn(rng, ComplexF64, n_syn, n_syn)
    C_sq = randn(rng, ComplexF64, n_syn, n_syn)

    # Embed into rectangular system (extra rows = consistent constraints)
    R = randn(rng, ComplexF64, m_syn - n_syn, n_syn) * 0.01
    U_embed = vcat(Matrix{ComplexF64}(I, n_syn, n_syn), R)
    A_rect = U_embed * A_sq
    B_rect = U_embed * B_sq
    C_rect = U_embed * C_sq

    # Ground truth: eigenvalues of square QEP
    eigs_true = _solve_square_qep(A_sq, B_sq, C_sq)

    # SVD compression solver
    eigs_svd = MetricsQNM.solve_qep_svd(A_rect, B_rect, C_rect; ω₀=0.0, refine=1)

    # Check: every true eigenvalue should be recovered
    n_found = 0
    max_err = 0.0
    for e_true in eigs_true
        !isfinite(e_true) && continue
        dists = [abs(e - e_true) for e in eigs_svd if isfinite(e)]
        isempty(dists) && continue
        err = minimum(dists)
        max_err = max(max_err, err)
        err < 1e-6 && (n_found += 1)
    end
    n_finite_true = count(isfinite, eigs_true)
    println("  Synthetic: $(n_found)/$(n_finite_true) eigenvalues recovered, max error = $(round(max_err, sigdigits=3))")
    @assert n_found == n_finite_true "SVD compression failed on synthetic problem!"
    println("  PASS")
    flush(stdout)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1: Build system via symbolic pipeline
# ═══════════════════════════════════════════════════════════════════════════════
a = 0.5; m = 2

println("\n[1] Building D̃ system (a=$a, m=$m)...")
flush(stdout)

# First, extract G coefficients (this is the slow symbolic step)
println("  Extracting G coefficients...")
flush(stdout)
t0 = time()
K0_r, K1_r, K2_r = MetricsQNM.extract_G_exact(a; P=3, Q=1, S=1, verbose=true)
t_extract = time() - t0
println("  G extraction: $(round(t_extract, digits=1))s")
flush(stdout)

# Count terms
for (name, K) in [("K₀", K0_r), ("K₁", K1_r), ("K₂", K2_r)]
    n = sum(length(d) for d in K.equations)
    println("  $name: $n terms")
end
flush(stdout)

# Determine d_max for r→z transform
global d_max = 0
for K in (K0_r, K1_r, K2_r), k in 1:10
    for ((γ, δ, σ, α, β, j), _) in K.equations[k]
        global d_max = max(d_max, δ)
    end
end
println("  Max r-degree: $d_max")
rp = MetricsQNM.r_plus(a)

# Transform r → z
K0_z = MetricsQNM._r_to_z(K0_r, rp, d_max)
K1_z = MetricsQNM._r_to_z(K1_r, rp, d_max)
K2_z = MetricsQNM._r_to_z(K2_r, rp, d_max)

# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2: SVD convergence study
# ═══════════════════════════════════════════════════════════════════════════════
ω_leav = leaver_qnm(a; l=2, m=m, n=0)
println("\n[2] SVD convergence study (Leaver ω = $ω_leav)")
flush(stdout)

@printf("  %-4s  %-12s  %-12s  %-12s  %-12s  %-6s\n",
        "N", "size", "σ_min", "σ_max", "ratio", "#small")
println("  " * "-" ^ 62)

for N in [4, 6, 8, 10, 12]
    basis = spectral_basis(N, m)
    D0 = assemble_system(K0_z, basis, a).D0
    D1 = assemble_system(K1_z, basis, a).D0
    D2 = assemble_system(K2_z, basis, a).D0
    sys = METRICSSystem(D0, D1, D2, N, m, a)

    D = Dtilde(sys, ω_leav)
    sv = svdvals(D)
    n_small = count(x -> x < 1e-6 * sv[1], sv)
    @printf("  %-4d  %-12s  %-12.3e  %-12.3e  %-12.3e  %-6d\n",
            N, "$(size(D,1))×$(size(D,2))", sv[end], sv[1], sv[end]/sv[1], n_small)
    flush(stdout)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 3: QEP eigenvalue search at multiple N
# ═══════════════════════════════════════════════════════════════════════════════
println("\n[3] QEP eigenvalue search:")
flush(stdout)

@printf("  %-4s  %-6s  %-28s  %-12s  %-12s\n",
        "N", "cols", "ω_best", "|Δω|", "residual")
println("  " * "-" ^ 66)

for N in [6, 8, 10, 12]
    basis = spectral_basis(N, m)
    D0 = assemble_system(K0_z, basis, a).D0
    D1 = assemble_system(K1_z, basis, a).D0
    D2 = assemble_system(K2_z, basis, a).D0
    sys = METRICSSystem(D0, D1, D2, N, m, a)

    # SVD compression QEP — project near Leaver guess
    t0 = time()
    eigs = solve_qep_svd(sys; ω₀=ω_leav, refine=1)
    t_qep = time() - t0

    # Find closest physical eigenvalue
    physical = filter(e -> isfinite(e) && imag(e) < 0 && abs(e) < 20, eigs)
    if !isempty(physical)
        dists = [abs(e - ω_leav) for e in physical]
        best_idx = argmin(dists)
        ω_best = physical[best_idx]
        Δω = dists[best_idx]
        res = qep_residual(sys, ω_best)

        @printf("  %-4d  %-6d  %+.10f%+.10fi  %-12.4e  %-12.4e\n",
                N, size(D0,2), real(ω_best), imag(ω_best), Δω, res)
    else
        @printf("  %-4d  %-6d  %-28s  %-12s  %-12s\n",
                N, size(D0,2), "NO PHYSICAL EIGENVALUE", "-", "-")
    end
    flush(stdout)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 4: Newton refinement of best QEP eigenvalue
# ═══════════════════════════════════════════════════════════════════════════════
println("\n[4] Newton refinement at N=12:")
flush(stdout)

N = 12
basis = spectral_basis(N, m)
D0 = assemble_system(K0_z, basis, a).D0
D1 = assemble_system(K1_z, basis, a).D0
D2 = assemble_system(K2_z, basis, a).D0
sys = METRICSSystem(D0, D1, D2, N, m, a)

eigs = solve_qep_svd(sys; ω₀=ω_leav, refine=1)
physical = filter(e -> isfinite(e) && imag(e) < 0 && abs(e) < 20, eigs)

if !isempty(physical)
    # Refine top 5 closest
    dists = [abs(e - ω_leav) for e in physical]
    perm = sortperm(dists)

    for i in 1:min(5, length(perm))
        ω_init = physical[perm[i]]
        ω_ref = solve_qep_newton(sys, ω_init; max_iter=30)
        res_init = qep_residual(sys, ω_init)
        res_ref = qep_residual(sys, ω_ref)
        Δ_init = abs(ω_init - ω_leav)
        Δ_ref = abs(ω_ref - ω_leav)
        @printf("  #%d: QEP %+.8f%+.8fi |Δ|=%.3e res=%.3e → Newton %+.8f%+.8fi |Δ|=%.3e res=%.3e\n",
                i, real(ω_init), imag(ω_init), Δ_init, res_init,
                real(ω_ref), imag(ω_ref), Δ_ref, res_ref)
        flush(stdout)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 5: Table I for a single spin (full pipeline demo)
# ═══════════════════════════════════════════════════════════════════════════════
println("\n[5] Table I demo (a=0.5):")
flush(stdout)
results = reproduce_table1(; N=12, m=2, spins=[0.5])

println("\n" * "=" ^ 70)
println("DONE")
