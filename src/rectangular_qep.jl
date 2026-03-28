# Rectangular QEP solver via SVD compression.
# Adapted from af-tests/examples13/rectangular_qep.jl.
#
# Solves (D₀ + ωD₁ + ω²D₂)v = 0 where D₀,D₁,D₂ ∈ ℂ^{m×n}, m ≥ n.
# Algorithm: SVD compress to n×n, then companion linearization + QZ.

export solve_qep_svd, solve_qep_newton, qep_residual, validate_qep
export solve_qep_with_vectors, classify_parity

"""
    solve_qep_svd(sys; ω₀=0.0, refine=1) → eigenvalues::Vector{ComplexF64}

Find ALL eigenvalues of the rectangular QEP D̃(ω)v = 0 via SVD compression.

1. SVD of P(ω₀) = D₀ + ω₀D₁ + ω₀²D₂  →  leading n left singular vectors Uₙ
2. Project to square: Ã = Uₙ'D₀, etc.
3. Companion linearization → 2n×2n generalized eigenvalue problem → QZ
4. Optionally refine by re-projecting at a found eigenvalue.
"""
function solve_qep_svd(sys::METRICSSystem; ω₀::Number=0.0, refine::Int=1, normalize::Bool=true)
    normalize && normalize_system!(sys)
    solve_qep_svd(sys.D0, sys.D1, sys.D2; ω₀, refine)
end

function solve_qep_svd(D0::AbstractMatrix, D1::AbstractMatrix, D2::AbstractMatrix;
                        ω₀::Number=0.0, refine::Int=1)
    m, n = size(D0)
    @assert size(D1) == size(D2) == (m, n)

    eigs = _svd_compress_solve(D0, D1, D2, ComplexF64(ω₀))

    for _ in 1:refine
        finite_eigs = filter(isfinite, eigs)
        isempty(finite_eigs) && break
        # Re-project at the eigenvalue with smallest imaginary part (most physical)
        physical = filter(e -> imag(e) < 0, finite_eigs)
        if !isempty(physical)
            ω_ref = physical[argmin(abs.(imag.(physical)))]
        else
            ω_ref = finite_eigs[argmin(abs.(finite_eigs))]
        end
        eigs = _svd_compress_solve(D0, D1, D2, ComplexF64(ω_ref))
    end

    return eigs
end

function _svd_compress_solve(D0, D1, D2, ω₀::ComplexF64)
    m, n = size(D0)
    P0 = D0 .+ ω₀ .* D1 .+ ω₀^2 .* D2
    F = svd(P0)

    # Project onto leading n left singular vectors
    if m > n
        Un = F.U[:, 1:n]
    else
        Un = F.U  # square case
    end

    A = Un' * D0
    B = Un' * D1
    C = Un' * D2

    return _solve_square_qep(A, B, C)
end

function _solve_square_qep(A, B, C)
    n = size(A, 1)

    # Companion linearization:
    # [  0   I ] x = ω [ I  0 ] x
    # [ -A  -B ]       [ 0  C ]
    L0 = zeros(ComplexF64, 2n, 2n)
    L1 = zeros(ComplexF64, 2n, 2n)
    L0[1:n, n+1:2n] .= I(n)
    L0[n+1:2n, 1:n] .= -A
    L0[n+1:2n, n+1:2n] .= -B
    L1[1:n, 1:n] .= I(n)
    L1[n+1:2n, n+1:2n] .= C

    return eigen(L0, L1).values
end

"""
    solve_qep_newton(sys, ω₀; max_iter=20) → ω::ComplexF64

Refine a single QEP eigenvalue near ω₀ using Newton on σ_min(D̃(ω)).
Each step costs one SVD. No companion linearization needed.
"""
function solve_qep_newton(sys::METRICSSystem, ω₀::Number; max_iter::Int=20)
    solve_qep_newton(sys.D0, sys.D1, sys.D2, ComplexF64(ω₀); max_iter)
end

function solve_qep_newton(D0::AbstractMatrix, D1::AbstractMatrix, D2::AbstractMatrix,
                           ω::ComplexF64; max_iter::Int=20)
    for _ in 1:max_iter
        P = D0 .+ ω .* D1 .+ ω^2 .* D2
        F = svd(P)
        σ = F.S[end]

        # Converged when σ_min is small relative to σ_max
        σ < 1e-13 * F.S[1] && return ω

        u = F.U[:, end]
        v = F.Vt[end, :]

        # P'(ω) = D1 + 2ωD2
        dP = D1 .+ 2ω .* D2
        deriv = dot(u, dP * v)

        abs(deriv) < 1e-15 && break

        ω -= σ / deriv
    end
    return ω
end

"""
    solve_qep_full_newton(sys, ω₀, v₀; parity=:polar, max_iter=30, tol=1e-12)
        → (ω, v, converged)

Paper's Newton-Raphson: iterate on BOTH (v, ω) simultaneously.
Solves D̃(ω)·v = 0 with one component of v pinned for normalization.

Unlike solve_qep_newton (which iterates on ω only via σ_min),
this uses the full Jacobian J = [D̃(ω)[:,free] | D̃'(ω)·v] and
iterates [Δv; Δω] = -J \\ D̃(ω)·v.  This converges to the specific
mode closest to (ω₀, v₀) in the full (v,ω) space, avoiding spurious
eigenvalues that steal σ_min-based Newton.
"""
function solve_qep_full_newton(sys::METRICSSystem, ω₀::Number,
                                v₀::AbstractVector{<:Number};
                                parity::Symbol=:polar, max_iter::Int=30,
                                tol::Float64=1e-12, verbose::Bool=false)
    N = sys.N
    m = sys.m
    bs = (N + 1)^2
    n_v = 6 * bs

    # Pin one coefficient for normalization (same as compute_jacobian)
    pinned_k = parity == :polar ? 1 : 5
    pinned_idx = (pinned_k - 1) * bs + nl_index(0, abs(m), N, m)
    free_idx = setdiff(1:n_v, pinned_idx)

    ω = ComplexF64(ω₀)
    v = ComplexF64.(v₀)

    # Pin normalization: v[pinned_idx] stays fixed
    v_pinned = v[pinned_idx]

    for iter in 1:max_iter
        # Evaluate D̃(ω)·v
        Dv = Dtilde(sys, ω) * v

        # Residual (all rows — the 10×bs equations)
        res_norm = norm(Dv)
        rel_res = res_norm / max(norm(v), 1.0)

        if verbose && (iter ≤ 5 || iter % 5 == 0)
            @printf("    iter %2d: |D̃·v| = %.2e, |ω - ω₀| = %.2e\n",
                    iter, rel_res, abs(ω - ω₀))
            flush(stdout)
        end

        rel_res < tol && return (ω=ω, v=v, converged=true, iters=iter)

        # Jacobian: J = [D̃(ω)[:,free] | D̃'(ω)·v]
        D_at_ω = Dtilde(sys, ω)
        dD_v = Dtilde_deriv(sys, ω) * v
        J = hcat(D_at_ω[:, free_idx], dD_v)

        # Newton step: [Δv_free; Δω] = -J \ D̃·v
        step = -(J \ Dv)

        # Update
        v[free_idx] .+= step[1:end-1]
        ω += step[end]
        v[pinned_idx] = v_pinned  # re-pin
    end

    return (ω=ω, v=v, converged=false, iters=max_iter)
end

export solve_qep_full_newton

"""
    qep_residual(sys, ω) → Float64

Relative residual σ_min(D̃(ω)) / σ_max(D̃(ω)).  Values < 1e-10 indicate eigenvalue.
"""
function qep_residual(sys::METRICSSystem, ω::Number)
    P = Dtilde(sys, ω)
    svals = svdvals(P)
    svals[1] == 0 && return 0.0
    return svals[end] / svals[1]
end

"""
    validate_qep(sys, eigenvalues; tol=1e-8) → (good, residuals)

Filter eigenvalues by QEP residual.
"""
function validate_qep(sys::METRICSSystem, eigenvalues; tol::Real=1e-8)
    residuals = Float64[]
    good = ComplexF64[]

    for ω in eigenvalues
        if !isfinite(ω)
            push!(residuals, Inf)
            continue
        end
        r = qep_residual(sys, ω)
        push!(residuals, r)
        r < tol && push!(good, ω)
    end

    return (good=good, residuals=residuals)
end

# ═══════════════════════════════════════════════════════════════════════════════
#  QEP solver with eigenvector return + parity classification
# ═══════════════════════════════════════════════════════════════════════════════

"""
    solve_qep_with_vectors(sys; ω₀=0.0, refine=1, normalize=true)
        → (eigenvalues, eigenvectors, Un)

Like solve_qep_svd but also returns the original-space eigenvectors.
Each column of `eigenvectors` corresponds to one eigenvalue.
"""
function solve_qep_with_vectors(sys::METRICSSystem; ω₀::Number=0.0,
                                 refine::Int=1, normalize::Bool=true)
    normalize && normalize_system!(sys)
    D0, D1, D2 = sys.D0, sys.D1, sys.D2
    m_rows, n_cols = size(D0)

    # Initial solve + refinement (same logic as solve_qep_svd)
    ω_proj = ComplexF64(ω₀)
    evals = ComplexF64[]
    for pass in 0:refine
        P0 = D0 .+ ω_proj .* D1 .+ ω_proj^2 .* D2
        F = svd(P0)
        Un = m_rows > n_cols ? F.U[:, 1:n_cols] : F.U

        A = Un' * D0
        B = Un' * D1
        C = Un' * D2

        n = size(A, 1)
        L0 = zeros(ComplexF64, 2n, 2n)
        L1 = zeros(ComplexF64, 2n, 2n)
        L0[1:n, n+1:2n] .= I(n)
        L0[n+1:2n, 1:n] .= -A
        L0[n+1:2n, n+1:2n] .= -B
        L1[1:n, 1:n] .= I(n)
        L1[n+1:2n, n+1:2n] .= C

        E = eigen(L0, L1)
        evals = E.values

        if pass < refine
            # Pick most physical eigenvalue for re-projection
            finite_eigs = filter(isfinite, evals)
            physical = filter(e -> imag(e) < 0, finite_eigs)
            if !isempty(physical)
                ω_proj = physical[argmin(abs.(imag.(physical)))]
            elseif !isempty(finite_eigs)
                ω_proj = finite_eigs[argmin(abs.(finite_eigs))]
            end
        else
            # Final pass: extract eigenvectors
            evecs = zeros(ComplexF64, n_cols, length(evals))
            for i in eachindex(evals)
                evecs[:, i] = E.vectors[1:n, i]
            end
            return (eigenvalues=evals, eigenvectors=evecs)
        end
    end

    # Fallback (refine=0)
    return (eigenvalues=evals, eigenvectors=zeros(ComplexF64, n_cols, 0))
end

"""
    classify_parity(v, N) → :axial or :polar

Classify an eigenvector as axial-led or polar-led based on the norm ratio
of polar (h₁-h₄) vs axial (h₅-h₆) field components.
"""
function classify_parity(v::AbstractVector{<:Number}, N::Int)
    bs = (N + 1)^2
    polar_norm = sum(abs2, v[1:4*bs])
    axial_norm = sum(abs2, v[4*bs+1:6*bs])
    return axial_norm > polar_norm ? :axial : :polar
end
