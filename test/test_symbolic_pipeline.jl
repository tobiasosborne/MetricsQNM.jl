#!/usr/bin/env julia
using MetricsQNM
using MetricsQNM: Dtilde
using LinearAlgebra

println("=" ^ 70)
println("SYMBOLIC G EXTRACTION — TREE-WALKING APPROACH")
println("=" ^ 70)
flush(stdout)

a = 0.5; m = 2

# Build system
println("\n[1] Full pipeline (a=$a, N=8, P=3, Q=1, S=1)...")
flush(stdout)
t0 = time()
sys = build_system_symbolic(a, 8, m; P=3, Q=1, S=1, verbose=true)
println("  TOTAL: $(round(time()-t0, digits=1))s")
flush(stdout)

println("\n  ||D₀||=$(round(norm(sys.D0),sigdigits=4))")
println("  ||D₁||=$(round(norm(sys.D1),sigdigits=4))")
println("  ||D₂||=$(round(norm(sys.D2),sigdigits=4))")

ω_leav = leaver_qnm(a; l=2, m=m, n=0)
println("  Leaver: ω = $ω_leav")
flush(stdout)

# SVD at multiple N (reuse extracted K by building systems at different N)
println("\n[2] SVD convergence study:")
flush(stdout)

# Re-extract for reuse (fast — already compiled)
K0_r, K1_r, K2_r = MetricsQNM.extract_G_exact(a; P=3, Q=1, S=1, verbose=false)

# Determine d_max
function find_d_max(Ks...)
    dm = 0
    for K in Ks, k in 1:10
        for ((γ, δ, σ, α, β, j), _) in K.equations[k]
            dm = max(dm, δ)
        end
    end
    return dm
end
d_max = find_d_max(K0_r, K1_r, K2_r)
rp = MetricsQNM.r_plus(a)
K0_z = MetricsQNM._r_to_z(K0_r, rp, d_max)
K1_z = MetricsQNM._r_to_z(K1_r, rp, d_max)
K2_z = MetricsQNM._r_to_z(K2_r, rp, d_max)

for N in [6, 8, 10, 12, 15]
    basis = spectral_basis(N, m)
    D0 = assemble_system(K0_z, basis, a).D0
    D1 = assemble_system(K1_z, basis, a).D0
    D2 = assemble_system(K2_z, basis, a).D0
    s = METRICSSystem(D0, D1, D2, N, m, a)

    D = Dtilde(s, ω_leav)
    sv = svdvals(D)
    n_small = count(x -> x < 1e-6*sv[1], sv)
    println("  N=$N: size=$(size(D)) σ_min=$(round(sv[end],sigdigits=3)) σ_max=$(round(sv[1],sigdigits=3)) ratio=$(round(sv[end]/sv[1],sigdigits=3)) #small=$n_small")
    flush(stdout)
end

# QEP at N=12
println("\n[3] QEP eigenvalue (N=12)...")
flush(stdout)
N_qep = 12
basis = spectral_basis(N_qep, m)
D0 = assemble_system(K0_z, basis, a).D0
D1 = assemble_system(K1_z, basis, a).D0
D2 = assemble_system(K2_z, basis, a).D0
sys_q = METRICSSystem(D0, D1, D2, N_qep, m, a)

ncol = size(D0, 2)
U = svd(D0).U[:, 1:ncol]
R0 = U' * D0; R1 = U' * D1; R2 = U' * D2
R2i = pinv(R2)
n = ncol
M = [zeros(ComplexF64, n, n) Matrix{ComplexF64}(I, n, n);
     -R2i * R0              -R2i * R1]
evals = eigen(M).values
physical = filter(λ -> isfinite(λ) && abs(λ) < 10 && imag(λ) < 0, evals)

if !isempty(physical)
    dists = [abs(λ - ω_leav) for λ in physical]
    perm = sortperm(dists)
    println("  Top 5 closest to Leaver:")
    for i in 1:min(5, length(perm))
        λ = physical[perm[i]]
        println("    ω=$(round(real(λ),sigdigits=10))+$(round(imag(λ),sigdigits=10))i  |Δω|=$(round(dists[perm[i]],sigdigits=4))")
    end
end

println("\n" * "=" ^ 70)
println("DONE")
