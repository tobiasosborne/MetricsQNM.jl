# Rectangular QEP solver via SVD compression.
# Adapted from af-tests/examples13/rectangular_qep.jl.
#
# Solves (D₀ + ωD₁ + ω²D₂)v = 0 where D₀,D₁,D₂ ∈ ℂ^{m×n}, m ≥ n.
# Algorithm: SVD compress to n×n, then companion linearization + QZ.

export solve_qep_svd, solve_qep_newton, qep_residual, validate_qep

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
