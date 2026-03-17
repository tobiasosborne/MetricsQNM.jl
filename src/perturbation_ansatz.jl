# Asymptotic controlling factors A_k(r) and compactified radial coordinate.
# Eqs. 15–17 of arxiv:2312.08435.

# ── Asymptotic divergence parameters (Eq. 16) ────────────────────────────────

rho_H(k::Int) = k == 3 ? 2 : k ∈ (2, 6) ? 1 : 0
rho_inf(k::Int) = k == 4 ? 1 : 2

"""
    asymptotic_factor(r, k, ω, p::KerrParams)

Controlling factor A_k(r) from Eq. 15.  Captures the singular behaviour
at the horizon and spatial infinity so that u_k = h_k / A_k is bounded.
"""
function asymptotic_factor(r, k::Int, ω, p::KerrParams)
    rp  = r_plus(p.a)
    b_  = b(p.a)
    ΩH  = Omega_H(p.a)
    exp(im * ω * r) *
        r^(2im * ω + rho_inf(k)) *
        ((r - rp) / r)^(-im * (ω - p.m * ΩH) * (1 + b_) / b_ - rho_H(k))
end

# ── Compactified radial coordinate (Eq. 17) ──────────────────────────────────

z_of_r(r, rp) = 2rp / r - 1        # r ∈ [r₊, ∞) → z ∈ [-1, +1]
r_of_z(z, rp) = 2rp / (1 + z)      # z ∈ [-1, +1] → r ∈ [r₊, ∞)

dz_dr(r, rp) = -2rp / r^2          # chain rule: dz/dr
dr_dz(z, rp) = -2rp / (1 + z)^2    # chain rule: dr/dz
