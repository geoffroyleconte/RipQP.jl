export K2_5StructuredParams

"""
Type to use the K2.5 formulation with a structured Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl).
This only works for solving Linear Problems.
The outer constructor 

    K2_5StructuredParams(; uplo = :L, kmethod = :trimr, rhs_scale = true,
                         atol0 = 1.0e-4, rtol0 = 1.0e-4,
                         atol_min = 1.0e-10, rtol_min = 1.0e-10,
                         ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5,
                         ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()),
                         itmax = 0, mem = 20, k3_resid = false, cb_only = false)

creates a [`RipQP.SolverParams`](@ref).
The available methods are:
- `:tricg`
- `:trimr`
- `:gpmr`

The `mem` argument sould be used only with `gpmr`.
"""
mutable struct K2_5StructuredParams{T} <: AugmentedParams{T}
  uplo::Symbol
  kmethod::Symbol
  rhs_scale::Bool
  atol0::T
  rtol0::T
  atol_min::T
  rtol_min::T
  ρ0::T
  δ0::T
  ρ_min::T
  δ_min::T
  itmax::Int
  mem::Int
  k3_resid::Bool
  cb_only::Bool # deactivate stop crit for Krylov method, leaving only the callback
end

function K2_5StructuredParams{T}(;
  uplo::Symbol = :L,
  kmethod::Symbol = :trimr,
  rhs_scale::Bool = true,
  atol0::T = eps(T)^(1 / 4),
  rtol0::T = eps(T)^(1 / 4),
  atol_min::T = sqrt(eps(T)),
  rtol_min::T = sqrt(eps(T)),
  ρ0::T = eps(T)^(1 / 4),
  δ0::T = eps(T)^(1 / 4),
  ρ_min::T = T(1e2 * sqrt(eps(T))),
  δ_min::T = T(1e2 * sqrt(eps(T))),
  itmax::Int = 0,
  mem::Int = 20,
  k3_resid = false,
  cb_only = false,
) where {T <: Real}
  @assert uplo == :L
  return K2_5StructuredParams(
    uplo,
    kmethod,
    rhs_scale,
    atol0,
    rtol0,
    atol_min,
    rtol_min,
    ρ0,
    δ0,
    ρ_min,
    δ_min,
    itmax,
    mem,
    k3_resid,
    cb_only,
  )
end

K2_5StructuredParams(; kwargs...) = K2_5StructuredParams{Float64}(; kwargs...)

mutable struct K2_5StructuredToK3Residuals{T, S <: AbstractVector{T}, M}
  AsqrtX1X2::M
  sqrtX1X2::S
  E::S
  regu::Regularization{T}
  ξ1::S
  ξ2::S
  atol::T
  rtol::T
  ϵ_d::S
  ϵ_p::S
  ϵ_l::S
  ϵ_u::S
  ϵK2_5::S
  ξ_l::S
  ξ_u::S
  Δs_l::S
  Δs_u::S
  s_l::S
  s_u::S
  x_m_lvar::S
  uvar_m_x::S
  μ0::T
  rbNorm0::T
  rcNorm0::T
  nvar::Int
  ncon::Int
  ilow::Vector{Int}
  iupp::Vector{Int}
  nlow::Int
  nupp::Int
end

ToK3Residuals(
  AsqrtX1X2::M,
  sqrtX1X2::S,
  E::S,
  regu::Regularization{T},
  ξ1::S,
  ξ2::S,
  itd::IterData{T, S},
  pt::Point{T, S},
  id::QM_IntData,
  sp::K2_5StructuredParams,
) where {T, S, M} = K2_5StructuredToK3Residuals{T, S, M}(
  AsqrtX1X2,
  sqrtX1X2,
  E,
  regu,
  ξ1,
  ξ2,
  T(sp.atol_min),
  T(sp.rtol_min),
  S(undef, id.nvar),
  S(undef, id.ncon),
  S(undef, id.nlow),
  S(undef, id.nupp),
  S(undef, id.nvar + id.ncon),
  S(undef, id.nlow),
  S(undef, id.nupp),
  S(undef, id.nlow),
  S(undef, id.nupp),
  pt.s_l,
  pt.s_u,
  itd.x_m_lvar,
  itd.uvar_m_x,
  zero(T),
  zero(T),
  zero(T),
  id.nvar,
  id.ncon,
  id.ilow,
  id.iupp,
  id.nlow,
  id.nupp,
)

function set_rd_init_res!(rd::K2_5StructuredToK3Residuals{T}, μ::T, rbNorm0::T, rcNorm0::T) where {T}
  rd.μ0 = μ
  rd.rbNorm0 = rbNorm0
  rd.rcNorm0 = rcNorm0
end

function (rd::K2_5StructuredToK3Residuals)(solver::KrylovSolver{T}, σ::T, μ::T, rbNorm::T, rcNorm::T) where {T}
  @views mul!(rd.ϵK2_5[1:rd.nvar], rd.AsqrtX1X2', solver.y)
  @. rd.ϵK2_5[1:rd.nvar] += -rd.E * solver.x - rd.ξ1
  @views mul!(rd.ϵK2_5[(rd.nvar + 1):end], rd.AsqrtX1X2, solver.x)
  @. rd.ϵK2_5[(rd.nvar + 1):end] += rd.regu.δ * solver.y - rd.ξ2
  rd.ϵ_p .= @views rd.ϵK2_5[(rd.nvar+1): end]
  @. rd.ξ_l = -rd.s_l * rd.x_m_lvar + σ * μ
  @. rd.ξ_u = -rd.s_u * rd.uvar_m_x + σ * μ
  @. rd.Δs_l = @views (σ * μ - rd.s_l * solver.x[rd.ilow] / rd.sqrtX1X2[rd.ilow]) / rd.x_m_lvar - rd.s_l
  @. rd.Δs_u = @views (σ * μ + rd.s_u * solver.x[rd.iupp] / rd.sqrtX1X2[rd.iupp]) / rd.uvar_m_x - rd.s_u
  @. rd.ϵ_l = @views rd.s_l * solver.x[rd.ilow] / rd.sqrtX1X2[rd.ilow] + rd.x_m_lvar * rd.Δs_l + rd.s_l * rd.x_m_lvar - σ * μ
  @. rd.ϵ_u = @views -rd.s_u * solver.x[rd.iupp] / rd.sqrtX1X2[rd.iupp] + rd.uvar_m_x * rd.Δs_u + rd.s_u * rd.uvar_m_x - σ * μ
  rd.ϵ_d .= @views rd.ϵK2_5[1:rd.nvar]
  @. rd.ϵ_d[rd.ilow] += rd.ϵ_l / rd.x_m_lvar
  @. rd.ϵ_d[rd.iupp] -= rd.ϵ_u / rd.uvar_m_x
  rd.ϵ_d ./= rd.sqrtX1X2
  return (norm(rd.ϵ_d, Inf) ≤ min(rd.atol, rd.rtol * rd.rcNorm0 * μ / rd.μ0) &&
          norm(rd.ϵ_p, Inf) ≤ min(rd.atol, rd.rtol * rd.rbNorm0 * μ / rd.μ0) &&
          norm(rd.ϵ_l, Inf) ≤ min(rd.atol, rd.rtol * norm(rd.ξ_l, Inf) * μ / rd.μ0) &&
          norm(rd.ϵ_u, Inf) ≤ min(rd.atol, rd.rtol * norm(rd.ξ_u, Inf) * μ / rd.μ0))
end

mutable struct PreallocatedDataK2_5Structured{
  T <: Real,
  S,
  Ksol <: KrylovSolver,
  L <: AbstractLinearOperator{T},
  R <: Union{K2_5StructuredToK3Residuals{T, S}, Int},
} <: PreallocatedDataAugmentedKrylovStructured{T, S}
  E::S                                  # temporary top-left diagonal
  invE::S
  sqrtX1X2::S # vector to scale K2 to K2.5
  AsqrtX1X2::L
  ξ1::S
  ξ2::S
  rhs_scale::Bool
  regu::Regularization{T}
  KS::Ksol
  kiter::Int
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
  itmax::Int
  rd::R
  cb_only::Bool
end

function opAsqrtX1X2tprod!(res, A, v, α, β, sqrtX1X2)
  mul!(res, transpose(A), v, α, β)
  res .*= sqrtX1X2
end

function PreallocatedData(
  sp::K2_5StructuredParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  E = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), :classic)
    E .= T(1.0e0) / 2
  else
    regu =
      Regularization(T(sp.ρ0), T(sp.δ0), T(sqrt(eps(T)) * 1e0), T(sqrt(eps(T)) * 1e0), :classic)
    E .= T(1.0e-2)
  end
  if regu.δ_min == zero(T) # gsp for gpmr
    regu.δ = zero(T)
  end
  invE = similar(E)
  @. invE = one(T) / E

  sqrtX1X2 = fill!(similar(fd.c), one(T))
  ξ1 = similar(fd.c, id.nvar)
  ξ2 = similar(fd.c, id.ncon)

  KS = init_Ksolver(fd.A', fd.b, sp)

  AsqrtX1X2 = LinearOperator(
    T,
    id.ncon,
    id.nvar,
    false,
    false,
    (res, v, α, β) -> mul!(res, fd.A, v .* sqrtX1X2, α, β),
    (res, v, α, β) -> opAsqrtX1X2tprod!(res, fd.A, v, α, β, sqrtX1X2),
  )

  rd = sp.k3_resid ? ToK3Residuals(AsqrtX1X2, sqrtX1X2, E, regu, ξ1, ξ2, itd, pt, id, sp) : 0

  return PreallocatedDataK2_5Structured(
    E,
    invE,
    sqrtX1X2,
    AsqrtX1X2,
    ξ1,
    ξ2,
    sp.rhs_scale,
    regu,
    KS,
    0,
    T(sp.atol0),
    T(sp.rtol0),
    T(sp.atol_min),
    T(sp.rtol_min),
    sp.itmax,
    rd,
    sp.cb_only,
  )
end

function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK2_5Structured{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  step::Symbol,
) where {T <: Real}
  pad.ξ1 .= @views step == :init ? fd.c : dd[1:(id.nvar)] .* pad.sqrtX1X2
  pad.ξ2 .= @views (step == :init && all(dd[(id.nvar + 1):end] .== zero(T))) ? one(T) :
         dd[(id.nvar + 1):end]
  if pad.rhs_scale
    rhsNorm = sqrt(norm(pad.ξ1)^2 + norm(pad.ξ2)^2)
    pad.ξ1 ./= rhsNorm
    pad.ξ2 ./= rhsNorm
  end
  (step !== :cc) && (pad.kiter = 0)
  ksolve!(
    pad.KS,
    pad.AsqrtX1X2',
    pad.ξ1,
    pad.ξ2,
    Diagonal(pad.invE),
    (one(T) / pad.regu.δ) .* I,
    verbose = 0,
    atol = pad.atol,
    rtol = pad.rtol,
    gsp = (pad.regu.δ == zero(T)),
    itmax = pad.itmax,
    callback = pad.rd == 0 ? solver -> false : solver -> pad.rd(solver, itd.σ, itd.μ, res.rbNorm, res.rcNorm),
    cb_only = pad.cb_only,
  )
  update_kresiduals_history!(
    res,
    pad.E,
    fd.A,
    pad.regu.δ,
    pad.KS.x,
    pad.KS.y,
    pad.ξ1,
    pad.ξ2,
    id.nvar,
  )
  pad.kiter += niterations(pad.KS)
  if pad.rhs_scale
    kunscale!(pad.KS.x, rhsNorm)
    kunscale!(pad.KS.y, rhsNorm)
  end

  @. dd[1:(id.nvar)] = pad.KS.x * pad.sqrtX1X2
  dd[(id.nvar + 1):end] .= pad.KS.y

  return 0
end

function update_pad!(
  pad::PreallocatedDataK2_5Structured{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
) where {T <: Real}
  if cnts.k != 0
    update_regu!(pad.regu)
  else
    pad.rd != 0 && set_rd_init_res!(pad.rd, itd.μ, res.rbNorm, res.rcNorm)
  end

  update_krylov_tol!(pad)

  pad.sqrtX1X2 .= one(T)
  @. pad.sqrtX1X2[id.ilow] *= sqrt(itd.x_m_lvar)
  @. pad.sqrtX1X2[id.iupp] *= sqrt(itd.uvar_m_x)
  pad.E .= pad.regu.ρ
  @. pad.E[id.ilow] += pt.s_l / itd.x_m_lvar
  @. pad.E[id.iupp] += pt.s_u / itd.uvar_m_x
  @. pad.E *= pad.sqrtX1X2^2
  @. pad.invE = one(T) / pad.E

  return 0
end
