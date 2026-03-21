# Reproduce all physics results from arxiv:2312.08435
# Generates data + plots for Figs 1, 2, 4, 5, 6, 12 and Table I.
#
# Usage: julia --project=. test/reproduce_paper.jl

using MetricsQNM
using LinearAlgebra
using Printf
using CairoMakie

const OUTDIR = joinpath(@__DIR__, "..", "figures")
mkpath(OUTDIR)

const TABLE1_SPINS = [0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# ═══════════════════════════════════════════════════════════════════════════════
#  Convergence study via QEP (Figs 1, 4)
# ═══════════════════════════════════════════════════════════════════════════════

function convergence_study(a::Float64; N_range=2:12, m=2)
    ω_L = leaver_qnm(a; s=-2, l=2, m=m, n=0)
    @printf("Convergence a=%.2f, ω_L=%.10f%+.10fi\n", a, real(ω_L), imag(ω_L))
    flush(stdout)

    systems = sweep_N(a, m, N_range; verbose=true)

    Ns = Int[]
    ωs = ComplexF64[]
    Rs = Float64[]  # QEP residual σ_min/σ_max at Leaver

    for N in sort(collect(N_range))
        sys = systems[N]
        # QEP with refined projection → eigenvalue (machine precision)
        eigs = solve_qep_svd(sys; ω₀=ω_L, refine=1, normalize=false)
        phys = filter(e -> isfinite(e) && imag(e) < 0 && abs(e) < 20, eigs)
        isempty(phys) && continue
        best = phys[argmin(abs.(phys .- ω_L))]
        # Spectral approximation quality: residual at Leaver frequency
        R = qep_residual(sys, ω_L)
        push!(Ns, N)
        push!(ωs, best)
        push!(Rs, R)
        @printf("  N=%2d: |Δω|=%.2e  R(ω_L)=%.2e\n", N, abs(best - ω_L), R)
        flush(stdout)
    end

    E = abs.(ωs .- ω_L)
    B = abs.(diff(ωs))
    return (Ns=Ns, ω_L=ω_L, ωs=ωs, E=E, B=B, Rs=Rs, a=a)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Axial/Polar separation via Newton-Raphson (Figs 2, 6, 12)
# ═══════════════════════════════════════════════════════════════════════════════

function axial_polar_study(a::Float64; N_range=4:12, m=2)
    ω_L = leaver_qnm(a; s=-2, l=2, m=m, n=0)
    ω_guess = round(real(ω_L), sigdigits=2) + im * round(imag(ω_L), sigdigits=2)
    @printf("Axial/Polar a=%.2f, guess=%.2f%+.2fi\n", a, real(ω_guess), imag(ω_guess))
    flush(stdout)

    systems = sweep_N(a, m, N_range; verbose=false)

    Ns = Int[]
    ω_A = ComplexF64[]
    ω_P = ComplexF64[]
    v_A_list = Vector{ComplexF64}[]
    v_P_list = Vector{ComplexF64}[]
    R_A = Float64[]
    R_P = Float64[]

    for N in sort(collect(N_range))
        sys = systems[N]
        res_P = solve_qnm(sys, ω_guess; parity=:polar, tol=1e-7, maxiter=20)
        res_A = solve_qnm(sys, ω_guess; parity=:axial, tol=1e-7, maxiter=20)

        push!(Ns, N)
        push!(ω_P, res_P.ω)
        push!(ω_A, res_A.ω)
        push!(v_P_list, copy(res_P.v))
        push!(v_A_list, copy(res_A.v))
        push!(R_P, res_P.residual_ratio)
        push!(R_A, res_A.residual_ratio)

        @printf("  N=%2d: ω_P=%.8f%+.8fi (R=%.1e)  ω_A=%.8f%+.8fi (R=%.1e)\n",
                N, real(res_P.ω), imag(res_P.ω), res_P.residual_ratio,
                real(res_A.ω), imag(res_A.ω), res_A.residual_ratio)
        flush(stdout)
    end

    E_A = abs.(ω_A .- ω_L)
    E_P = abs.(ω_P .- ω_L)
    ω_avg = (ω_A .+ ω_P) ./ 2
    E_avg = abs.(ω_avg .- ω_L)

    return (Ns=Ns, ω_L=ω_L, ω_A=ω_A, ω_P=ω_P, ω_avg=ω_avg,
            E_A=E_A, E_P=E_P, E_avg=E_avg,
            v_A=v_A_list, v_P=v_P_list, R_A=R_A, R_P=R_P, a=a)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

function plot_fig1(data_qep, data_ap; filename="fig1_convergence_a01.pdf")
    fig = Figure(size=(900, 400))

    # Left: QEP residual at Leaver (spectral approximation quality)
    ax1 = Axis(fig[1,1], xlabel="N", ylabel="log₁₀(σ_min/σ_max)",
               title="QEP residual at ω_Leaver (a=$(data_qep.a))")
    ok = data_qep.Rs .> 0
    scatter!(ax1, data_qep.Ns[ok], log10.(data_qep.Rs[ok]),
             marker=:diamond, color=:green, markersize=10, label="QEP residual")

    # Overlay NR absolute error where available
    ok_A = data_ap.E_A .> 0
    ok_P = data_ap.E_P .> 0
    any(ok_A) && scatter!(ax1, data_ap.Ns[ok_A], log10.(data_ap.E_A[ok_A]),
                          marker=:circle, color=:blue, label="NR axial |Δω|")
    any(ok_P) && scatter!(ax1, data_ap.Ns[ok_P], log10.(data_ap.E_P[ok_P]),
                          marker=:rect, color=:red, label="NR polar |Δω|")
    axislegend(ax1, position=:rt)

    # Right: Backward modulus difference
    ax2 = Axis(fig[1,2], xlabel="N", ylabel="log₁₀(B)",
               title="Backward modulus difference (a=$(data_qep.a))")
    Ns_B = data_qep.Ns[2:end]
    ok_B = data_qep.B .> 0
    any(ok_B) && scatter!(ax2, Ns_B[ok_B], log10.(data_qep.B[ok_B]),
                          marker=:diamond, color=:green, label="QEP")
    axislegend(ax2, position=:rt)

    save(joinpath(OUTDIR, filename), fig)
    println("  Saved $filename"); flush(stdout)
end

function plot_fig2(data_ap; filename="fig2_isospectrality_a01.pdf")
    fig = Figure(size=(500, 400))
    ax = Axis(fig[1,1], xlabel="N", ylabel="log₁₀(ratio)",
              title="Isospectrality (a=$(data_ap.a))")

    iso_diff = abs.(data_ap.ω_A .- data_ap.ω_P)
    ok_A = data_ap.E_A .> 0
    ok_P = data_ap.E_P .> 0
    ratio_A = iso_diff ./ max.(data_ap.E_A, 1e-30)
    ratio_P = iso_diff ./ max.(data_ap.E_P, 1e-30)

    scatter!(ax, data_ap.Ns[ok_A], log10.(ratio_A[ok_A]),
             marker=:circle, color=:blue, label="|ω_A-ω_P|/E_A")
    scatter!(ax, data_ap.Ns[ok_P], log10.(ratio_P[ok_P]),
             marker=:rect, color=:red, label="|ω_A-ω_P|/E_P")
    hlines!(ax, [0.0], color=:black, linestyle=:dash)
    axislegend(ax, position=:rt)

    save(joinpath(OUTDIR, filename), fig)
    println("  Saved $filename"); flush(stdout)
end

function plot_fig4(data_list; filename="fig4_convergence_multi_a.pdf")
    fig = Figure(size=(900, 400))
    colors = [:blue, :red, :green]
    markers = [:circle, :rect, :diamond]

    ax1 = Axis(fig[1,1], xlabel="N", ylabel="log₁₀(σ_min/σ_max)",
               title="QEP residual at ω_Leaver")
    ax2 = Axis(fig[1,2], xlabel="N", ylabel="log₁₀(B)",
               title="Backward modulus difference")

    for (i, data) in enumerate(data_list)
        ok = data.Rs .> 0
        scatter!(ax1, data.Ns[ok], log10.(data.Rs[ok]),
                 marker=markers[i], color=colors[i], label="a=$(data.a)")
        Ns_B = data.Ns[2:end]
        ok_B = data.B .> 0
        any(ok_B) && scatter!(ax2, Ns_B[ok_B], log10.(data.B[ok_B]),
                              marker=markers[i], color=colors[i], label="a=$(data.a)")
    end
    axislegend(ax1, position=:rt)
    axislegend(ax2, position=:rt)

    save(joinpath(OUTDIR, filename), fig)
    println("  Saved $filename"); flush(stdout)
end

function plot_fig5(; filename="fig5_qnm_complex_plane.pdf")
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1,1], xlabel="ωRe", ylabel="ωIm", title="022-mode QNM frequencies")

    # Dense Leaver curve
    a_dense = 0.005:0.005:0.95
    ω_dense = [leaver_qnm(a; s=-2, l=2, m=2, n=0) for a in a_dense]
    lines!(ax, real.(ω_dense), imag.(ω_dense), color=:gray, linewidth=2, label="Leaver")

    # QEP at Table I spins
    ωRe = Float64[]; ωIm = Float64[]
    for a in TABLE1_SPINS
        sys, _ = build_system_bespoke(a, 8, 2)
        ω_L = leaver_qnm(a; s=-2, l=2, m=2, n=0)
        eigs = solve_qep_svd(sys; ω₀=ω_L, refine=1, normalize=false)
        phys = filter(e -> isfinite(e) && imag(e) < 0, eigs)
        !isempty(phys) || continue
        best = phys[argmin(abs.(phys .- ω_L))]
        push!(ωRe, real(best)); push!(ωIm, imag(best))
    end
    scatter!(ax, ωRe, ωIm, marker=:star5, color=:black, markersize=10, label="METRICS (N=8)")
    axislegend(ax, position=:lb)

    save(joinpath(OUTDIR, filename), fig)
    println("  Saved $filename"); flush(stdout)
end

function plot_fig6(; filename="fig6_accuracy_vs_spin.pdf")
    fig = Figure(size=(900, 800))
    a_v = Float64[]; E_v = Float64[]; ΔRe_v = Float64[]; ΔIm_v = Float64[]; R_v = Float64[]

    for a in TABLE1_SPINS
        sys, _ = build_system_bespoke(a, 8, 2)
        ω_L = leaver_qnm(a; s=-2, l=2, m=2, n=0)
        eigs = solve_qep_svd(sys; ω₀=ω_L, refine=1, normalize=false)
        phys = filter(e -> isfinite(e) && imag(e) < 0, eigs)
        !isempty(phys) || continue
        best = phys[argmin(abs.(phys .- ω_L))]
        push!(a_v, a)
        push!(E_v, abs(best - ω_L))
        push!(ΔRe_v, abs(1 - real(best)/real(ω_L)))
        push!(ΔIm_v, abs(1 - imag(best)/imag(ω_L)))
        push!(R_v, qep_residual(sys, best))
    end

    ax1 = Axis(fig[1,1], xlabel="a", ylabel="log₁₀(E)", title="Absolute error")
    scatter!(ax1, a_v, log10.(max.(E_v, 1e-16)), marker=:diamond, color=:black)

    ax2 = Axis(fig[1,2], xlabel="a", ylabel="log₁₀(R)", title="QEP residual σ_min/σ_max")
    scatter!(ax2, a_v, log10.(max.(R_v, 1e-16)), marker=:diamond, color=:black)

    ax3 = Axis(fig[2,1], xlabel="a", ylabel="log₁₀(ΔRe)", title="Relative error (Re)")
    scatter!(ax3, a_v, log10.(max.(ΔRe_v, 1e-16)), marker=:diamond, color=:black)

    ax4 = Axis(fig[2,2], xlabel="a", ylabel="log₁₀(ΔIm)", title="Relative error (Im)")
    scatter!(ax4, a_v, log10.(max.(ΔIm_v, 1e-16)), marker=:diamond, color=:black)

    save(joinpath(OUTDIR, filename), fig)
    println("  Saved $filename"); flush(stdout)
end

function plot_fig12(; N=8, filename="fig12_parity_dominance.pdf")
    fig = Figure(size=(500, 400))
    ax = Axis(fig[1,1], xlabel="a", ylabel="log₁₀(PD)", title="Parity dominance")

    a_v = Float64[]; PD_A = Float64[]; PD_P = Float64[]
    bs = (N + 1)^2

    for a in TABLE1_SPINS
        sys, _ = build_system_bespoke(a, N, 2)
        ω_L = leaver_qnm(a; s=-2, l=2, m=2, n=0)
        ω_guess = round(real(ω_L), sigdigits=2) + im * round(imag(ω_L), sigdigits=2)

        res_P = solve_qnm(sys, ω_guess; parity=:polar, tol=1e-7, maxiter=20)
        res_A = solve_qnm(sys, ω_guess; parity=:axial, tol=1e-7, maxiter=20)

        function parity_dom(v)
            polar_n = sum(abs2, v[1:4*bs])
            axial_n = sum(abs2, v[4*bs+1:6*bs])
            return sqrt(polar_n / max(axial_n, 1e-30))
        end

        push!(a_v, a)
        push!(PD_A, parity_dom(res_A.v))
        push!(PD_P, parity_dom(res_P.v))
        @printf("  a=%.3f: PD_A=%.4f  PD_P=%.4f\n", a, PD_A[end], PD_P[end])
        flush(stdout)
    end

    scatter!(ax, a_v, log10.(PD_A), marker=:circle, color=:blue, label="Axial-led")
    scatter!(ax, a_v, log10.(PD_P), marker=:rect, color=:red, label="Polar-led")
    hlines!(ax, [0.0], color=:black, linestyle=:dash)
    axislegend(ax, position=:rt)

    save(joinpath(OUTDIR, filename), fig)
    println("  Saved $filename"); flush(stdout)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println("=" ^ 70)
    println("Reproducing physics results from arxiv:2312.08435")
    println("Output: $OUTDIR")
    println("=" ^ 70); flush(stdout)

    # ── Fig 5: Complex plane (fast, no N-sweep) ────────────────────────
    println("\n--- Fig 5: QNM in complex plane ---"); flush(stdout)
    plot_fig5()

    # ── Fig 6: Accuracy vs spin (fast, N=8 only) ─────────────────────
    println("\n--- Fig 6: Accuracy metrics vs spin ---"); flush(stdout)
    plot_fig6()

    # ── Fig 1 + 2: a=0.1 convergence + isospectrality ────────────────
    println("\n--- Fig 1 & 2: a=0.1 convergence + isospectrality ---"); flush(stdout)
    qep_01 = convergence_study(0.1; N_range=2:14)
    ap_01 = axial_polar_study(0.1; N_range=4:12)
    plot_fig1(qep_01, ap_01)
    plot_fig2(ap_01)

    # ── Fig 4: Multi-spin convergence ────────────────────────────────
    println("\n--- Fig 4: Convergence a=0.3, 0.6, 0.9 ---"); flush(stdout)
    qep_03 = convergence_study(0.3; N_range=2:14)
    qep_06 = convergence_study(0.6; N_range=2:14)
    qep_09 = convergence_study(0.9; N_range=2:14)
    plot_fig4([qep_03, qep_06, qep_09])

    # ── Fig 12: Parity dominance ────────────────────────────────────────
    println("\n--- Fig 12: Parity dominance ---"); flush(stdout)
    plot_fig12()

    println("\n" * "=" ^ 70)
    println("All figures saved to $OUTDIR")
    println("=" ^ 70)
end

main()
