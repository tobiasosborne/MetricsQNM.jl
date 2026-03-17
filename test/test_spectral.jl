using SparseArrays

@testset "Chebyshev basis" begin
    N = 15
    cb = chebyshev_basis(N)

    # ── T_n evaluation at Chebyshev nodes ────────────────────────────────
    # T_n(cos(θ)) = cos(nθ)
    nodes = [cos(π * k / N) for k in 0:N]
    T_at = zeros(N + 1, N + 1)  # T_at[n+1, k+1] = T_n(nodes[k+1])
    for n in 0:N, k in 0:N
        θ = π * k / N
        T_at[n+1, k+1] = cos(n * θ)
    end

    # ── Product rule: z · T_n = (T_{n+1} + T_{n-1})/2 ──────────────────
    Z1 = cb.Z[2]  # multiplication by z
    for n in 0:N-1  # skip n=N where z·T_N involves T_{N+1} beyond truncation
        # Coefficient vector for T_n
        e_n = zeros(N + 1); e_n[n+1] = 1.0
        # z · T_n coefficients
        z_T_n = Z1 * e_n
        # Evaluate both sides at nodes
        for (ki, z) in enumerate(nodes)
            lhs = z * T_at[n+1, ki]
            rhs = sum(z_T_n[j+1] * T_at[j+1, ki] for j in 0:N)
            @test lhs ≈ rhs atol=1e-12
        end
    end

    # ── Derivative: dT_n/dz ──────────────────────────────────────────────
    # dT_n/dz at node z_k via finite differences, compared to D matrix
    D = cb.D
    for n in 2:N-1  # skip endpoints where FD is less accurate
        e_n = zeros(N + 1); e_n[n+1] = 1.0
        dT_coeffs = D * e_n
        for ki in 2:N  # interior nodes
            z = nodes[ki]
            # Analytical: dT_n/dz = n sin(nθ)/sin(θ) where z = cos(θ)
            θ = acos(clamp(z, -1, 1))
            sinθ = sin(θ)
            if abs(sinθ) > 1e-8
                exact = n * sin(n * θ) / sinθ
                approx = sum(dT_coeffs[j+1] * T_at[j+1, ki] for j in 0:N)
                @test exact ≈ approx atol=1e-8
            end
        end
    end

    # ── z^0 is identity ──────────────────────────────────────────────────
    @test Matrix(cb.Z[1]) ≈ I(N + 1)
end

@testset "Associated Legendre basis" begin
    m = 2
    N = 10
    lb = legendre_basis(N, m)

    # ── Product rule at sample points ────────────────────────────────────
    # P_l^m values via recurrence
    function plm_values(χ, m, lmax)
        am = abs(m)
        # P_{|m|}^{|m|} = (-1)^m (2m-1)!! (1-χ²)^{m/2}
        Plm = zeros(lmax - am + 1)
        fact = 1.0
        for k in 1:am
            fact *= (2k - 1)
        end
        Plm[1] = (-1)^am * fact * (1 - χ^2)^(am / 2)

        if lmax > am
            # P_{|m|+1}^{|m|} = χ(2|m|+1) P_{|m|}^{|m|}
            Plm[2] = χ * (2am + 1) * Plm[1]
        end

        for l in (am + 2):lmax
            i = l - am + 1
            Plm[i] = ((2l - 1) * χ * Plm[i-1] - (l + am - 1) * Plm[i-2]) / (l - am)
        end
        return Plm
    end

    X1 = lb.X[2]  # multiplication by χ
    test_points = [-0.8, -0.3, 0.0, 0.4, 0.9]
    lmax = abs(m) + N
    for χ in test_points
        Ps = plm_values(χ, m, lmax)
        for i in 1:N  # skip i=N+1 where χ·P_{lmax}^m involves P_{lmax+1}^m beyond truncation
            e_i = zeros(N + 1); e_i[i] = 1.0
            χP_coeffs = X1 * e_i
            lhs = χ * Ps[i]
            rhs = sum(χP_coeffs[j] * Ps[j] for j in 1:(N + 1))
            @test lhs ≈ rhs atol=1e-10
        end
    end

    # ── χ^0 is identity ─────────────────────────────────────────────────
    @test Matrix(lb.X[1]) ≈ I(N + 1)
end

@testset "SpectralBasis" begin
    sb = spectral_basis(10, 2)
    @test sb.block_size == 121  # 11²
    @test nl_index(0, 2, 10, 2) == 1
    @test nl_index(1, 3, 10, 2) == 13  # 1*11 + (3-2) + 1
end
