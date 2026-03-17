# Kerr black hole background quantities.
# Convention: M = 1 throughout (arxiv:2312.08435 Eq. 2).

"""
    KerrParams(a, m)

Kerr background parameters: dimensionless spin `a` (0 ≤ a < 1)
and azimuthal mode number `m`.
"""
struct KerrParams
    a::Float64
    m::Int
    function KerrParams(a, m)
        0 ≤ a < 1 || throw(DomainError(a, "spin must satisfy 0 ≤ a < 1"))
        new(Float64(a), m)
    end
end

# Derived quantities (Eq. 2)
b(a)       = sqrt(1 - a^2)
r_plus(a)  = 1 + b(a)
r_minus(a) = 1 - b(a)
Omega_H(a) = a / (2 * r_plus(a))
kappa(a)   = b(a) / (2 * r_plus(a))   # surface gravity (M=1)

# Metric functions
Sigma(r, χ, a) = r^2 + a^2 * χ^2
Delta(r, a)    = (r - r_plus(a)) * (r - r_minus(a))
