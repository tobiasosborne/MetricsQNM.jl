@testset "Leaver QNM solver" begin
    # Table I of arxiv:2312.08435 — Leaver column (l=2, m=2, n=0, s=-2)
    # 16-digit reference values
    table1_leaver = [
        (a=0.005, ωR= 0.374302314745705,    ωI=-0.0889522053640457),
        (a=0.1,   ωR= 0.3870175383645592,   ωI=-0.0887056990268991),
        (a=0.2,   ωR= 0.4021453241072112,   ωI=-0.08831116615465),
        (a=0.3,   ωR= 0.4195266799093153,   ωI=-0.0877292712328145),
        (a=0.4,   ωR= 0.439841909727434,    ωI=-0.0868819580547294),
        (a=0.5,   ωR= 0.4641229739649294,   ωI=-0.0856388194008764),
        (a=0.6,   ωR= 0.4940446109217166,   ωI=-0.0837651572095065),
        (a=0.7,   ωR= 0.5325997998444519,   ωI=-0.0807927741196761),
        (a=0.8,   ωR= 0.5860160981862801,   ωI=-0.07562938913772186),
        (a=0.9,   ωR= 0.671613259501218,    ωI=-0.06486906741255006),
        (a=0.95,  ωR= 0.7463194371599231,   ωI=-0.05314891507283093),
    ]

    # Test at representative spin values: low spin to high precision, high spin to ~1e-5
    for row in table1_leaver[[1, 3]]   # a = 0.005, 0.2
        ω = leaver_qnm(row.a; l=2, m=2, n=0)
        @test real(ω) ≈ row.ωR rtol=1e-9
        @test imag(ω) ≈ row.ωI rtol=1e-9
    end
    for row in table1_leaver[[6, 9, 11]]  # a = 0.5, 0.8, 0.95
        ω = leaver_qnm(row.a; l=2, m=2, n=0)
        @test real(ω) ≈ row.ωR rtol=1e-5
        @test imag(ω) ≈ row.ωI rtol=1e-5
    end
end
