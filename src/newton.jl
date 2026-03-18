# Newton-Raphson QNM solver with Moore-Penrose pseudoinverse.
# Eqs. 36–45 of arxiv:2312.08435.

"""
    solve_qnm(sys, ω_guess; parity=:polar, tol=1e-7, maxiter=20)

Find the QNM frequency ω by solving D̃(ω)v = 0 via Newton-Raphson
with Moore-Penrose pseudoinverse updates.

Returns a NamedTuple: (ω, v, iterations, residual, residual_ratio).
"""
function solve_qnm(sys::METRICSSystem, ω_guess::Number;
                    parity::Symbol=:polar, tol::Float64=1e-7, maxiter::Int=20)
    N = sys.N
    m = sys.m
    bs = (N + 1)^2   # block size

    # Number of unknowns per field: 6 fields × bs coefficients
    n_v = 6 * bs
    # Index of the pinned coefficient
    pinned_k = parity == :polar ? 1 : 5
    pinned_idx = (pinned_k - 1) * bs + nl_index(0, abs(m), N, m)

    # Initialize: v with one pinned coefficient = 1, rest = 0
    v = zeros(ComplexF64, n_v)
    v[pinned_idx] = 1.0
    ω = ComplexF64(ω_guess)

    # Free variable indices (all v except pinned, plus ω at the end)
    free_idx = setdiff(1:n_v, pinned_idx)
    x = vcat(v[free_idx], ω)

    residual_initial = NaN

    for iter in 1:maxiter
        # Reconstruct full v from x
        ω = x[end]
        v[free_idx] .= x[1:end-1]
        v[pinned_idx] = 1.0

        # Residual
        D = Dtilde(sys, ω)
        f = D * v

        residual_norm = norm(f)
        if iter == 1
            residual_initial = residual_norm
        end

        if residual_norm < tol
            return (ω = ω,
                    v = v,
                    iterations = iter,
                    residual = residual_norm,
                    residual_ratio = residual_norm / residual_initial)
        end

        # Jacobian: [∂f/∂v_free | ∂f/∂ω]
        J_v = D[:, free_idx]
        J_ω = Dtilde_deriv(sys, ω) * v
        J = hcat(J_v, J_ω)

        # Moore-Penrose update
        x .-= pinv(J) * f
    end

    ω = x[end]
    v[free_idx] .= x[1:end-1]
    v[pinned_idx] = 1.0
    res = norm(Dtilde(sys, ω) * v)

    @warn "Newton-Raphson did not converge" iterations=maxiter residual=res
    return (ω = ω, v = v, iterations = maxiter,
            residual = res, residual_ratio = res / residual_initial)
end

"""
    reproduce_table1(; N=12, m=2, spins=nothing, verbose=true)

Reproduce Table I of arxiv:2312.08435 using SVD compression QEP solver.
For each spin value:
1. Build D̃₀, D̃₁, D̃₂ via symbolic G extraction pipeline
2. Solve rectangular QEP via SVD compression + companion QZ
3. Find eigenvalue closest to Leaver reference
4. Optionally refine with Newton on σ_min
"""
function reproduce_table1(; N::Int=12, m::Int=2,
                            spins::Union{Nothing,Vector{Float64}}=nothing,
                            verbose::Bool=true)
    if spins === nothing
        spins = [0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    end
    results = NamedTuple[]

    for a in spins
        ω_leaver = leaver_qnm(a; l=2, m=m, n=0)
        verbose && println("a=$a: Leaver ω = $ω_leaver")

        # Build system via symbolic pipeline
        t0 = time()
        sys = build_system_symbolic(a, N, m; P=3, Q=1, S=1, verbose=false)
        t_build = time() - t0
        verbose && println("  Build: $(round(t_build, digits=1))s  size=$(size(sys.D0))")

        # Solve via SVD compression QEP
        t0 = time()
        eigs = solve_qep_svd(sys; ω₀=ω_leaver, refine=1)
        t_solve = time() - t0

        # Find closest physical eigenvalue (Im < 0, finite)
        physical = filter(e -> isfinite(e) && imag(e) < 0 && abs(e) < 20, eigs)
        if !isempty(physical)
            dists = [abs(e - ω_leaver) for e in physical]
            best_idx = argmin(dists)
            ω_qep = physical[best_idx]
            Δω = abs(ω_qep - ω_leaver)

            # Newton refinement
            ω_refined = solve_qep_newton(sys, ω_qep; max_iter=30)
            Δω_refined = abs(ω_refined - ω_leaver)

            verbose && println("  QEP:    ω = $(round(real(ω_qep),sigdigits=10)) + $(round(imag(ω_qep),sigdigits=10))i  |Δω|=$(round(Δω,sigdigits=4))")
            verbose && println("  Newton: ω = $(round(real(ω_refined),sigdigits=10)) + $(round(imag(ω_refined),sigdigits=10))i  |Δω|=$(round(Δω_refined,sigdigits=4))")
            verbose && println("  Solve: $(round(t_solve, digits=2))s")

            push!(results, (a=a, ω_leaver=ω_leaver, ω_qep=ω_qep, ω_refined=ω_refined,
                           Δω_qep=Δω, Δω_refined=Δω_refined, t_build=t_build, t_solve=t_solve))
        else
            verbose && println("  No physical eigenvalues found!")
            push!(results, (a=a, ω_leaver=ω_leaver, ω_qep=NaN+NaN*im, ω_refined=NaN+NaN*im,
                           Δω_qep=NaN, Δω_refined=NaN, t_build=t_build, t_solve=t_solve))
        end
    end

    if verbose
        println("\n" * "=" ^ 70)
        println("SUMMARY — Table I reproduction (N=$N)")
        println("=" ^ 70)
        @printf("  %-6s  %-28s  %-12s  %-12s\n", "a", "ω_METRICS", "|Δω| QEP", "|Δω| Newton")
        println("  " * "-" ^ 64)
        for r in results
            @printf("  %-6.3f  %+.8f %+.8fi  %.4e  %.4e\n",
                    r.a, real(r.ω_refined), imag(r.ω_refined), r.Δω_qep, r.Δω_refined)
        end
    end

    return results
end
