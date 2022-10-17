export K2StructuredParams

"""
Type to use the K2 formulation with a structured Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl).
This only works for solving Linear Problems.
The outer constructor 

    K2StructuredParams(; uplo = :L, kmethod = :trimr, rhs_scale = true, 
                       atol0 = 1.0e-4, rtol0 = 1.0e-4,
                       atol_min = 1.0e-10, rtol_min = 1.0e-10, 
                       ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()),
                       itmax = 0, mem = 20, k3_resid = false, cb_only = false)

creates a [`RipQP.SolverParams`](@ref).
The available methods are:
- `:tricg`
- `:trimr`
- `:gpmr`

The `mem` argument sould be used only with `gpmr`.
"""
mutable struct K2StructuredParams{T} <: AugmentedParams{T}
  uplo::Symbol
  kmethod::Symbol
  rhs_scale::Bool
  atol0::T
  rtol0::T
  atol_min::T
  rtol_min::T
  ρ_min::T
  δ_min::T
  itmax::Int
  mem::Int
  k3_resid::Bool
  cb_only::Bool # deactivate stop crit for Krylov method, leaving only the callback
end

function K2StructuredParams{T}(;
  uplo::Symbol = :L,
  kmethod::Symbol = :trimr,
  rhs_scale::Bool = true,
  atol0::T = eps(T)^(1 / 4),
  rtol0::T = eps(T)^(1 / 4),
  atol_min::T = sqrt(eps(T)),
  rtol_min::T = sqrt(eps(T)),
  ρ_min::T = T(1e2 * sqrt(eps(T))),
  δ_min::T = T(1e2 * sqrt(eps(T))),
  itmax::Int = 0,
  mem::Int = 20,
  k3_resid = false,
  cb_only = false,
) where {T <: Real}
  @assert uplo == :L
  return K2StructuredParams(
    uplo,
    kmethod,
    rhs_scale,
    atol0,
    rtol0,
    atol_min,
    rtol_min,
    ρ_min,
    δ_min,
    itmax,
    mem,
    k3_resid,
    cb_only,
  )
end

K2StructuredParams(; kwargs...) = K2StructuredParams{Float64}(; kwargs...)

mutable struct K2StructuredToK3Residuals{T, S <: AbstractVector{T}, M}
  A::M
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
  ϵK2::S
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
  A::M,
  E::S,
  regu::Regularization{T},
  ξ1::S,
  ξ2::S,
  itd::IterData{T, S},
  pt::Point{T, S},
  id::QM_IntData,
  sp::K2StructuredParams,
) where {T, S, M} = K2StructuredToK3Residuals{T, S, M}(
  A,
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

function set_rd_init_res!(rd::K2StructuredToK3Residuals{T}, μ::T, rbNorm0::T, rcNorm0::T) where {T}
  rd.μ0 = μ
  rd.rbNorm0 = rbNorm0
  rd.rcNorm0 = rcNorm0
end

function (rd::K2StructuredToK3Residuals)(solver::KrylovSolver{T}, σ::T, μ::T, rbNorm::T, rcNorm::T) where {T}
  @views mul!(rd.ϵK2[1:rd.nvar], rd.A', solver.y)
  @. rd.ϵK2[1:rd.nvar] += -rd.E * solver.x - rd.ξ1
  @views mul!(rd.ϵK2[(rd.nvar + 1):end], rd.A, solver.x)
  @. rd.ϵK2[(rd.nvar + 1):end] += rd.regu.δ * solver.y - rd.ξ2
  rd.ϵ_p .= @views rd.ϵK2[(rd.nvar+1): end]
  @. rd.ξ_l = -rd.s_l * rd.x_m_lvar + σ * μ
  @. rd.ξ_u = -rd.s_u * rd.uvar_m_x + σ * μ
  @. rd.Δs_l = @views (σ * μ - rd.s_l * solver.x[rd.ilow]) / rd.x_m_lvar - rd.s_l
  @. rd.Δs_u = @views (σ * μ + rd.s_u * solver.x[rd.iupp]) / rd.uvar_m_x - rd.s_u
  @. rd.ϵ_l = @views rd.s_l * solver.x[rd.ilow] + rd.x_m_lvar * rd.Δs_l + rd.s_l * rd.x_m_lvar - σ * μ
  @. rd.ϵ_u = @views -rd.s_u * solver.x[rd.iupp] + rd.uvar_m_x * rd.Δs_u + rd.s_u * rd.uvar_m_x - σ * μ
  rd.ϵ_d .= @views rd.ϵK2[1:rd.nvar]
  @. rd.ϵ_d[rd.ilow] += rd.ϵ_l / rd.x_m_lvar
  @. rd.ϵ_d[rd.iupp] -= rd.ϵ_u / rd.uvar_m_x
  return (norm(rd.ϵ_d, Inf) ≤ min(rd.atol, rd.rtol * rd.rcNorm0 * μ / rd.μ0) &&
          norm(rd.ϵ_p, Inf) ≤ min(rd.atol, rd.rtol * rd.rbNorm0 * μ / rd.μ0) &&
          norm(rd.ϵ_l, Inf) ≤ min(rd.atol, rd.rtol * norm(rd.ξ_l, Inf) * μ / rd.μ0) &&
          norm(rd.ϵ_u, Inf) ≤ min(rd.atol, rd.rtol * norm(rd.ξ_u, Inf) * μ / rd.μ0))
end

mutable struct PreallocatedDataK2Structured{T <: Real, S, Ksol <: KrylovSolver,
  R <: Union{K2StructuredToK3Residuals{T, S}, Int}} <:
      PreallocatedDataAugmentedKrylovStructured{T, S}
  E::S  # temporary top-left diagonal
  invE::S
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

function PreallocatedData(
  sp::K2StructuredParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  E = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu =
      Regularization(T(sqrt(eps()) * 1e5), T(sqrt(eps()) * 1e5), T(sp.ρ_min), T(sp.δ_min), :classic)
    E .= T(1.0e0) / 2
  else
    regu = Regularization(
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps(T)) * 1e0),
      T(sqrt(eps(T)) * 1e0),
      :classic,
    )
    E .= T(1.0e-2)
  end
  if regu.δ_min == zero(T) # gsp for gpmr
    regu.δ = zero(T)
  end
  invE = similar(E)
  invE .= one(T) ./ E

  ξ1 = similar(fd.c, id.nvar)
  ξ2 = similar(fd.c, id.ncon)

  KS = init_Ksolver(fd.A', fd.b, sp)
  rd = sp.k3_resid ? ToK3Residuals(fd.A, E, regu, ξ1, ξ2, itd, pt, id, sp) : 0

  return PreallocatedDataK2Structured(
    E,
    invE,
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

function update_kresiduals_history!(
  res::AbstractResiduals{T},
  E::AbstractVector{T},
  A::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  δ::T,
  solx::AbstractVector{T},
  soly::AbstractVector{T},
  ξ1::AbstractVector{T},
  ξ2::AbstractVector{T},
  nvar::Int,
) where {T <: Real}
  if typeof(res) <: ResidualsHistory
    @views mul!(res.Kres[1:nvar], A', soly)
    @. res.Kres[1:nvar] += -E * solx - ξ1
    @views mul!(res.Kres[(nvar + 1):end], A, solx)
    @. res.Kres[(nvar + 1):end] += δ * soly - ξ2
  end
end

function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK2Structured{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  step::Symbol,
) where {T <: Real}
  pad.ξ1 .= @views step == :init ? fd.c : dd[1:(id.nvar)]
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
    fd.A',
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
  dd[1:(id.nvar)] .= pad.KS.x
  dd[(id.nvar + 1):end] .= pad.KS.y

  return 0
end

function update_pad!(
  pad::PreallocatedDataK2Structured{T},
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

  pad.E .= pad.regu.ρ
  @. pad.E[id.ilow] += pt.s_l / itd.x_m_lvar
  @. pad.E[id.iupp] += pt.s_u / itd.uvar_m_x
  @. pad.invE = one(T) / pad.E

  return 0
end
