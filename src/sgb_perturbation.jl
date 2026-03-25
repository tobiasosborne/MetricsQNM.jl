# Eigenvalue perturbation solver for scalar-Gauss-Bonnet corrections.
# Implements Eq. 111 of arxiv:2406.11986:
#   x⁽¹⁾ = -J⁻¹ · [D̃⁽¹⁾(ω⁽⁰⁾) · v⁽⁰⁾]
# where ω⁽¹⁾ = last component of x⁽¹⁾.

export compute_jacobian, solve_sgb_perturbation

"""
    compute_jacobian(sys, ω, v; parity=:polar)

Compute the Newton-Raphson Jacobian J = [D̃(ω)[:,free] | D̃'(ω)·v]
at a converged GR solution (ω, v). This is the Jacobian needed for
eigenvalue perturbation theory (Eq. 111 of 2406.11986).

Returns (J, free_idx, pinned_idx).
"""
function compute_jacobian(sys::METRICSSystem, ω::ComplexF64, v::Vector{ComplexF64};
                          parity::Symbol=:polar)
    N = sys.N
    m = sys.m
    bs = (N + 1)^2
    n_v = 6 * bs

    pinned_k = parity == :polar ? 1 : 5
    pinned_idx = (pinned_k - 1) * bs + nl_index(0, abs(m), N, m)
    free_idx = setdiff(1:n_v, pinned_idx)

    D = Dtilde(sys, ω)
    J_v = D[:, free_idx]
    J_ω = Dtilde_deriv(sys, ω) * v
    J = hcat(J_v, J_ω)

    return J, free_idx, pinned_idx
end

"""
    solve_sgb_perturbation(sys_corr, ω0, v0, J)

Compute the leading-order sGB frequency correction ω⁽¹⁾ via eigenvalue
perturbation theory.

Arguments:
- `sys_corr::METRICSSystem`: The D̃⁽¹⁾ correction matrices (D0, D1, D2 are
  the ζ-derivatives of the D̃ matrices)
- `ω0::ComplexF64`: GR eigenfrequency ω⁽⁰⁾
- `v0::Vector{ComplexF64}`: GR eigenvector v⁽⁰⁾
- `J::Matrix{ComplexF64}`: GR Jacobian at (ω0, v0)

Returns ω⁽¹⁾ ∈ ℂ (the leading-order frequency correction).
"""
function solve_sgb_perturbation(sys_corr::METRICSSystem,
                                 ω0::ComplexF64,
                                 v0::Vector{ComplexF64},
                                 J::Matrix{ComplexF64})
    # Source vector: D̃⁽¹⁾(ω⁰) · v⁰
    source = (sys_corr.D0 + ω0 * sys_corr.D1 + ω0^2 * sys_corr.D2) * v0

    # Solve J · x⁽¹⁾ = -source via pseudoinverse
    x1 = -(pinv(J) * source)

    # ω⁽¹⁾ is the last component (corresponds to the frequency variable)
    ω1 = x1[end]

    return ω1
end

"""
    solve_sgb_perturbation(sys_corr, gr_result; parity=:polar)

Convenience method using the NamedTuple from solve_qnm directly.
"""
function solve_sgb_perturbation(sys_corr::METRICSSystem, gr_result::NamedTuple;
                                 parity::Symbol=:polar)
    if haskey(gr_result, :J)
        return solve_sgb_perturbation(sys_corr, ComplexF64(gr_result.ω),
                                       gr_result.v, gr_result.J)
    else
        error("gr_result does not contain Jacobian J. Use solve_qnm or compute_jacobian.")
    end
end
