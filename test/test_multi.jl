function createQuadraticModelT(qpdata; T = Double64, name = "qp_pb")
  return QuadraticModel(
    convert(Array{T}, qpdata.c),
    qpdata.qrows,
    qpdata.qcols,
    convert(Array{T}, qpdata.qvals),
    Arows = qpdata.arows,
    Acols = qpdata.acols,
    Avals = convert(Array{T}, qpdata.avals),
    lcon = convert(Array{T}, qpdata.lcon),
    ucon = convert(Array{T}, qpdata.ucon),
    lvar = convert(Array{T}, qpdata.lvar),
    uvar = convert(Array{T}, qpdata.uvar),
    c0 = T(qpdata.c0),
    x0 = zeros(T, length(qpdata.c)),
    name = name,
  )
end

@testset "multi_mode" begin
  stats1 = ripqp(QuadraticModel(qps1), mode = :multi, display = false)
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :first_order

  stats2 = ripqp(QuadraticModel(qps2), mode = :multi, display = false)
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :first_order

  stats3 = ripqp(QuadraticModel(qps3), mode = :multi, display = false)
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :first_order
end

@testset "Float16, Float32, Float128, BigFloat" begin
  qm16 = QuadraticModel(
    Float16.(c),
    Float16.(tril(Q)),
    A = Float16.(A),
    lcon = Float16.(b),
    ucon = Float16.(b),
    lvar = Float16.(l),
    uvar = Float16.(u),
    c0 = Float16(0.0),
    x0 = zeros(Float16, 3),
    name = "QM16",
  )
  stats_dense = ripqp(qm16, itol = InputTol(Float16), display = false, ps = false)
  @test isapprox(stats_dense.objective, 1.1249999990782493, atol = 1e-2)
  @test stats_dense.status == :first_order

  for T ∈ [
    Float32,
    # Double64,
  ]
    qmT_2 = createQuadraticModelT(qps2, T = T)
    stats2 = ripqp(qmT_2, display = false)
    @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
    @test stats2.status == :first_order
  end

  qm128_1 = createQuadraticModelT(qps1, T = BigFloat)
  stats1 = ripqp(
    qm128_1,
    itol = InputTol(BigFloat, ϵ_rb1 = BigFloat(0.1), ϵ_rb2 = BigFloat(0.01)),
    mode = :multi,
    normalize_rtol = false,
    display = false,
  )
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :first_order
end

@testset "multi solvers" begin
  for mode ∈ [:multi, :multizoom, :multiref]
    stats1 = ripqp(
      QuadraticModel(qps1),
      mode = mode,
      solve_method = IPF(),
      sp2 = K2KrylovParams(uplo = :U, kmethod = :gmres, preconditioner = LDL()),
      display = false,
    )
    @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
    @test stats1.status == :first_order
  end

  T128 = BigFloat
  qm128_1 = createQuadraticModelT(qps1, T = T128)
  stats1 = ripqp(
    qm128_1,
    mode = :multi,
    solve_method = IPF(),
    sp = K2KrylovParams(
      uplo = :U,
      form_mat = true,
      equilibrate = false,
      preconditioner = LDL(T = Float64),
      ρ_min = sqrt(eps()),
      δ_min = sqrt(eps()),
    ),
    sp2 = K2KrylovParams{T128}(
      uplo = :U,
      form_mat = true,
      equilibrate = false,
      preconditioner = LDL(T = Float64),
      atol_min = T128(1.0e-18),
      rtol_min = T128(1.0e-18),
      ρ_min = sqrt(eps(T128)),
      δ_min = sqrt(eps(T128)),
    ),
    display = true,
    itol = InputTol(T128, ϵ_rb = T128(1.0e-12), ϵ_rc = T128(1.0e-12)),
  )
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :first_order
end
