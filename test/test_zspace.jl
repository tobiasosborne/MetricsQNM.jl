#!/usr/bin/env julia
using MetricsQNM
using MetricsQNM: compile_field_equations, compute_field_equations,
                  Dtilde, Dtilde_deriv, extract_chebyshev_coefficients,
                  assemble_system_chebyshev
using Symbolics
using LinearAlgebra

println("=" ^ 70)
println("Z-SPACE + A_k: QEP CONVERGENCE STUDY")
println("=" ^ 70)
flush(stdout)

cfe = compile_field_equations(2; verbose=false)
h_term_strings = let
    eqs, coords, params, hfuncs, freq_vars = compute_field_equations(2)
    all_vars = Set{Any}()
    for eq in eqs, v in Symbolics.get_variables(eq)
        any(occursin("h$(j)", string(v)) for j in 1:6) && push!(all_vars, v)
    end
    string.(sort(collect(all_vars), by=string))
end
println("  compiled")
flush(stdout)

a = 0.5; m = 2
ω_leav = leaver_qnm(a; l=2, m=m, n=0)
println("  Leaver: ω = $ω_leav\n")
flush(stdout)

println("[1] Extracting (P4Q2 Ak d_z=28)...")
flush(stdout)
K0, K1, K2 = extract_chebyshev_coefficients(
    cfe, a, m; h_term_strings, P=4, Q=2, S=1, d_z=28, d_chi=10, use_Ak=true)
println("  done\n")
flush(stdout)

# QEP solver function
function solve_qep(D0, D1, D2, ω_target)
    nrow, ncol = size(D0)
    # Project to square via thin SVD of D₀
    U_thin = svd(D0).U[:, 1:ncol]
    R0 = U_thin' * D0
    R1 = U_thin' * D1
    R2 = U_thin' * D2
    n = ncol

    # Companion: [0 I; -R₀ -R₁] z = ω [I 0; 0 R₂] z
    # If R₂ is well-conditioned, use standard form
    R2i = pinv(R2)
    M = [zeros(ComplexF64, n, n)  Matrix{ComplexF64}(I, n, n);
         -R2i * R0                -R2i * R1]

    evals = eigen(M).values

    # Filter physical eigenvalues
    physical = filter(λ -> isfinite(λ) && abs(λ) < 10 && imag(λ) < 0, evals)
    isempty(physical) && return nothing

    # Find closest to target
    dists = [abs(λ - ω_target) for λ in physical]
    idx = argmin(dists)
    return physical[idx]
end

# Test at multiple N
println("[2] QEP convergence:")
println("  N     ω_QEP_Re        ω_QEP_Im         |Δω|")
flush(stdout)

for N in [6, 8, 10, 12]
    t0 = time()
    basis = spectral_basis(N, m)
    D0 = assemble_system_chebyshev(K0, basis, a).D0
    D1 = assemble_system_chebyshev(K1, basis, a).D0
    D2 = assemble_system_chebyshev(K2, basis, a).D0

    ω_qep = solve_qep(D0, D1, D2, ω_leav)
    t = time() - t0

    if ω_qep !== nothing
        Δω = abs(ω_qep - ω_leav)
        println("  $N   $(round(real(ω_qep),sigdigits=10))  $(round(imag(ω_qep),sigdigits=10))  $(round(Δω,sigdigits=4))  $(round(t,digits=1))s")
    else
        println("  $N   --- no physical eigenvalue found ---")
    end
    flush(stdout)
end

# ── SVD-Newton refinement from QEP initial guess ────────────────────────
println("\n[3] SVD-Newton refinement from QEP guess (N=10)...")
flush(stdout)

N = 10
basis = spectral_basis(N, m)
D0_n = assemble_system_chebyshev(K0, basis, a).D0
D1_n = assemble_system_chebyshev(K1, basis, a).D0
D2_n = assemble_system_chebyshev(K2, basis, a).D0
sys = METRICSSystem(D0_n, D1_n, D2_n, N, m, a)

ω_qep = solve_qep(D0_n, D1_n, D2_n, ω_leav)
println("  QEP initial: ω = $ω_qep")
flush(stdout)

if ω_qep !== nothing
    # SVD-Newton refinement
    ω = ComplexF64(ω_qep)
    for iter in 1:20
        D = Dtilde(sys, ω)
        # Row normalize
        for i in axes(D, 1)
            s = norm(@view D[i, :]); s > 0 && (D[i,:] ./= s)
        end
        F = svd(D)
        σ_min = F.S[end]
        u = F.U[:, end]; v = F.V[:, end]
        dD = Dtilde_deriv(sys, ω)
        for i in axes(dD, 1)
            s = norm(@view Dtilde(sys, ω)[i, :]); s > 0 && (dD[i,:] ./= s)
        end
        Δω = -σ_min / (u' * dD * v)
        ω += Δω
        println("    iter=$iter: σ_min=$(round(σ_min,sigdigits=4)) |Δω_step|=$(round(abs(Δω),sigdigits=3)) ω=$(round(real(ω),sigdigits=10))+$(round(imag(ω),sigdigits=10))i")
        flush(stdout)
        abs(Δω) < 1e-10 && break
    end
    println("\n  Final: ω = $ω")
    println("  |Δω_Leaver| = $(abs(ω - ω_leav))")
end
flush(stdout)

println("\n" * "=" ^ 70)
println("DONE")
