# QNM solver via Newton-Raphson on the factored D̃(ω) system.
# D̃(ω) = D̃₀ + D̃₁ω + D̃₂ω² is row-normalized at each step for conditioning.

export solve_qnm_direct

"""
    solve_qnm_direct(sys, ω0; parity=:polar, tol=1e-7, maxiter=20)

Find the QNM frequency using a METRICSSystem with constant D̃₀, D̃₁, D̃₂.

At each step:
1. Assemble D̃(ω) = D̃₀ + D̃₁ω + D̃₂ω²
2. Row-normalize for conditioning
3. Apply the standard overdetermined Newton-Raphson with pinv
"""
function solve_qnm_direct(sys::METRICSSystem, ω0::Number;
                           parity::Symbol=:polar, tol::Float64=1e-7, maxiter::Int=20)
    N, m = sys.N, sys.m
    am = abs(m)
    bs = (N + 1)^2
    n_v = 6 * bs

    pinned_k = parity == :polar ? 1 : 5
    pinned_idx = (pinned_k - 1) * bs + nl_index(0, am, N, m)

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

        # Assemble and row-normalize
        D = Dtilde(sys, ω)
        scales = [norm(@view D[i, :]) for i in axes(D, 1)]
        for i in axes(D, 1)
            scales[i] > 0 && (D[i, :] ./= scales[i])
        end

        f = D * v
        res = norm(f)
        iter == 1 && (residual_initial = res)

        if res < tol
            return (ω=ω, v=v, iterations=iter, residual=res,
                    residual_ratio=res / residual_initial)
        end

        # Jacobian with normalized D̃ derivative
        dD = Dtilde_deriv(sys, ω)
        for i in axes(dD, 1)
            scales[i] > 0 && (dD[i, :] ./= scales[i])
        end

        J_v = D[:, free_idx]
        J_ω = dD * v
        J = hcat(J_v, J_ω)

        Δx = pinv(J) * f
        x .-= Δx

        if iter ≤ 5 || iter % 5 == 0
            @info "Newton" iter residual=res Δω=abs(Δx[end]) ω=x[end]
        end
    end

    ω = x[end]; v[free_idx] .= x[1:end-1]; v[pinned_idx] = 1.0
    res = norm(Dtilde(sys, ω) * v)
    @warn "No convergence" maxiter residual=res
    return (ω=ω, v=v, iterations=maxiter, residual=res,
            residual_ratio=res / residual_initial)
end
