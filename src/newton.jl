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
    reproduce_table1(; N=30, m=2, tol=1e-7)

Reproduce Table I of arxiv:2312.08435 for all 11 spin values.
"""
function reproduce_table1(; N::Int=30, m::Int=2, tol::Float64=1e-7)
    spins = [0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    results = NamedTuple[]

    for a in spins
        ω_leaver = leaver_qnm(a; l=2, m=m, n=0)
        ω_guess = round(ω_leaver, sigdigits=2)

        # TODO: assemble system and solve once assembly pipeline is complete
        # sys = assemble_system(a, N, m)
        # result_polar = solve_qnm(sys, ω_guess; parity=:polar, tol)
        # result_axial = solve_qnm(sys, ω_guess; parity=:axial, tol)

        push!(results, (a=a, ω_leaver=ω_leaver))
    end
    return results
end
