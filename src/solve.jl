# QNM solver: full Newton-Raphson on the overdetermined system D̃(ω)v = 0
# with D̃ rebuilt at each step (supports ω-dependent A_k factorization).

export solve_qnm_direct

"""
    solve_qnm_direct(build_D, ω0, N, m; parity=:polar, tol=1e-7, maxiter=20, δω=1e-6)

Find the QNM frequency by Newton-Raphson on D̃(ω)v = 0.

`build_D(ω)` returns the row-normalized D̃(ω) matrix.

At each step:
1. Build D̃(ω_n)
2. Compute residual f = D̃(ω_n) v_n  (with fixed normalization v[pinned] = 1)
3. Compute Jacobian J = [D̃ columns for free v | (D̃(ω+δ)-D̃(ω))/δ × v]
4. Update (v_free, ω) via Moore-Penrose: Δx = -pinv(J) f
"""
function solve_qnm_direct(build_D, ω0::Number, N::Int, m::Int;
                           parity::Symbol=:polar, tol::Float64=1e-7,
                           maxiter::Int=20, δω::Float64=1e-6)
    am = abs(m)
    bs = (N + 1)^2
    n_v = 6 * bs

    # Pinned coefficient index
    pinned_k = parity == :polar ? 1 : 5
    pinned_idx = (pinned_k - 1) * bs + nl_index(0, am, N, m)

    # Initialize
    v = zeros(ComplexF64, n_v)
    v[pinned_idx] = 1.0
    ω = ComplexF64(ω0)

    free_idx = setdiff(1:n_v, pinned_idx)
    x = vcat(v[free_idx], ω)

    residual_initial = NaN

    for iter in 1:maxiter
        ω = x[end]
        v[free_idx] .= x[1:end-1]
        v[pinned_idx] = 1.0

        # Build D̃ at current ω
        D = build_D(ω)

        # Residual
        f = D * v
        res = norm(f)
        iter == 1 && (residual_initial = res)

        if res < tol
            @info "Converged" iter residual=res ω
            return (ω=ω, v=v, iterations=iter, residual=res,
                    residual_ratio=res / residual_initial)
        end

        # Jacobian: [∂f/∂v_free | ∂f/∂ω]
        J_v = D[:, free_idx]

        # ∂f/∂ω via finite difference
        D_h = build_D(ω + δω)
        J_ω = (D_h * v - f) / δω

        J = hcat(J_v, J_ω)

        # Moore-Penrose update
        Δx = pinv(J) * f
        x .-= Δx

        if iter ≤ 5 || iter % 5 == 0
            @info "Newton step" iter residual=res Δω=abs(Δx[end]) ω=x[end]
        end
    end

    ω = x[end]
    v[free_idx] .= x[1:end-1]
    v[pinned_idx] = 1.0
    res = norm(build_D(ω) * v)

    @warn "No convergence" maxiter residual=res
    return (ω=ω, v=v, iterations=maxiter, residual=res,
            residual_ratio=res / residual_initial)
end
