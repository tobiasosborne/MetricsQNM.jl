@testset "Kerr background" begin
    # ── Schwarzschild limit (a = 0) ──────────────────────────────────────
    @test b(0.0) == 1.0
    @test r_plus(0.0) == 2.0
    @test r_minus(0.0) == 0.0
    @test Omega_H(0.0) == 0.0
    @test kappa(0.0) ≈ 0.25   # surface gravity of Schwarzschild

    # Delta(r, 0) = r² - 2r = r(r-2)
    @test Delta(3.0, 0.0) ≈ 3.0 * 1.0
    @test Delta(2.0, 0.0) ≈ 0.0 atol=1e-15

    # ── Extremal limit (a → 1) ───────────────────────────────────────────
    @test r_plus(0.999) ≈ 1 + sqrt(1 - 0.999^2) atol=1e-10
    @test b(0.999) ≈ sqrt(1 - 0.999^2) atol=1e-10

    # ── General properties ───────────────────────────────────────────────
    for a in [0.1, 0.3, 0.5, 0.7, 0.9]
        # Sigma > 0 outside horizon
        for r in [r_plus(a) + 0.01, 3.0, 10.0, 100.0]
            for χ in [-0.9, 0.0, 0.5, 1.0]
                @test Sigma(r, χ, a) > 0
            end
        end
        # Delta vanishes at horizon
        @test Delta(r_plus(a), a) ≈ 0.0 atol=1e-14
        @test Delta(r_minus(a), a) ≈ 0.0 atol=1e-14

        # Horizon angular velocity is positive for prograde
        @test Omega_H(a) > 0
        # Surface gravity
        @test kappa(a) > 0
        @test kappa(a) ≈ b(a) / (2 * r_plus(a))
    end

    # ── KerrParams validation ────────────────────────────────────────────
    @test KerrParams(0.5, 2).a == 0.5
    @test KerrParams(0.5, 2).m == 2
    @test_throws DomainError KerrParams(1.0, 2)   # a = 1 not allowed
    @test_throws DomainError KerrParams(-0.1, 2)   # negative a
end
