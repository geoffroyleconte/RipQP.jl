# (A D⁻¹ Aᵀ + δI) Δy = A D⁻¹ ξ₁ + ξ₂ 
# where D = s_l (x - lvar)⁻¹ + s_u (uvar - x)⁻¹ + ρI,
# and the right hand side of K2 is rhs = [ξ₁]
#                                        [ξ₂] 
export K1KrylovParams

"""
Type to use the K1 formulation with a Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K1KrylovParams(; uplo = :L, kmethod = :cg, preconditioner = Identity(),
                   rhs_scale = true,
                   atol0 = 1.0e-4, rtol0 = 1.0e-4, 
                   atol_min = 1.0e-10, rtol_min = 1.0e-10,
                   ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5, 
                   ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()),
                   itmax = 0, mem = 20, k3_resid = false, cb_only = false)

creates a [`RipQP.SolverParams`](@ref).
The available methods are:
- `:cg`
- `:cg_lanczos`
- `:cr`
- `:minres`
- `:minres_qlp`
- `:symmlq`

"""
mutable struct K1KrylovParams{T, PT} <: NormalKrylovParams{T, PT}
  uplo::Symbol
  kmethod::Symbol
  preconditioner::PT
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

function K1KrylovParams{T}(;
  uplo::Symbol = :L,
  kmethod::Symbol = :cg,
  preconditioner::AbstractPreconditioner = Identity(),
  rhs_scale::Bool = true,
  atol0::T = eps(T)^(1 / 4),
  rtol0::T = eps(T)^(1 / 4),
  atol_min::T = sqrt(eps(T)),
  rtol_min::T = sqrt(eps(T)),
  ρ0::T = T(sqrt(eps()) * 1e5),
  δ0::T = T(sqrt(eps()) * 1e5),
  ρ_min::T = T(1e3 * sqrt(eps())),
  δ_min::T = T(1e4 * sqrt(eps())),
  itmax::Int = 0,
  mem::Int = 20,
  k3_resid = false,
  cb_only = false,
) where {T <: Real}
  return K1KrylovParams(
    uplo,
    kmethod,
    preconditioner,
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

K1KrylovParams(; kwargs...) = K1KrylovParams{Float64}(; kwargs...)

mutable struct K1ToK3Residuals{T, S <: AbstractVector{T}, M, MA}
  K::M
  rhs::S
  atol::T
  rtol::T
  ϵK1::S
  ϵ_d::S
  ϵ_p::S
  ϵ_l::S
  ϵ_u::S
  ϵ1::S
  ξ1::S
  ξ2::S
  ξ_l::S
  ξ_u::S
  Δx::S
  Δs_l::S
  Δs_u::S
  D::S
  uplo::Symbol
  A::MA
  s_l::S
  s_u::S
  x_m_lvar::S
  uvar_m_x::S
  ilow::Vector{Int}
  iupp::Vector{Int}
  μ0::T
  rbNorm0::T
  rcNorm0::T
  nvar::Int
  ncon::Int
  nlow::Int
  nupp::Int
end

ToK3Residuals(
  K,
  rhs::S,
  itd::IterData{T, S},
  pt::Point{T, S},
  id::QM_IntData,
  sp::K1KrylovParams,
  fd::Abstract_QM_FloatData{T, S},
  D::S,
) where {T, S} = K1ToK3Residuals(
  K,
  rhs,
  T(sp.atol_min),
  T(sp.rtol_min),
  S(undef, id.ncon),
  S(undef, id.nvar),
  S(undef, id.ncon),
  S(undef, id.nlow),
  S(undef, id.nupp),
  S(undef, id.nvar),
  S(undef, id.nvar),
  S(undef, id.ncon),
  S(undef, id.nlow),
  S(undef, id.nupp),
  S(undef, id.nvar),
  S(undef, id.nlow),
  S(undef, id.nupp),
  D,
  fd.uplo,
  fd.A,
  pt.s_l,
  pt.s_u,
  itd.x_m_lvar,
  itd.uvar_m_x,
  id.ilow,
  id.iupp,
  zero(T),
  zero(T),
  zero(T),
  id.nvar,
  id.ncon,
  id.nlow,
  id.nupp,
)

function set_rd_init_res!(rd::K1ToK3Residuals{T}, μ::T, rbNorm0::T, rcNorm0::T) where {T}
  rd.μ0 = μ
  rd.rbNorm0 = rbNorm0
  rd.rcNorm0 = rcNorm0
end

function (rd::K1ToK3Residuals)(solver::KrylovSolver{T}, σ::T, μ::T, rbNorm::T, rcNorm::T, dd::AbstractVector{T}) where {T}
  mul!(rd.ϵK1, rd.K, solver.x)
  rd.ϵK1 .-= rd.rhs
  rd.ξ1 .= @views dd[1:rd.nvar]
  rd.ξ2 .= @views dd[rd.nvar+1: end]
  @. rd.ξ_l = -rd.s_l * rd.x_m_lvar + σ * μ
  @. rd.ξ_u = -rd.s_u * rd.uvar_m_x + σ * μ
  if rd.uplo == :L
    mul!(rd.Δx, rd.A', solver.x)
  else
    mul!(rd.Δx, rd.A, solver.x)
  end
  rd.ϵ1 .= rd.Δx
  @. rd.Δx = (-rd.ξ1 + rd.Δx) / rd.D
  @. rd.ϵ1 = -rd.D * rd.Δx + rd.ϵ1 - rd.ξ1
  @. rd.Δs_l = @views (σ * μ - rd.s_l * rd.Δx[rd.ilow]) / rd.x_m_lvar - rd.s_l
  @. rd.Δs_u = @views (σ * μ + rd.s_u * rd.Δx[rd.iupp]) / rd.uvar_m_x - rd.s_u
  @. rd.ϵ_l = @views rd.s_l * rd.Δx[rd.ilow] + rd.x_m_lvar * rd.Δs_l + rd.s_l * rd.x_m_lvar - σ * μ
  @. rd.ϵ_u = @views -rd.s_u * rd.Δx[rd.iupp] + rd.uvar_m_x * rd.Δs_u + rd.s_u * rd.uvar_m_x - σ * μ
  rd.ϵ_d .= rd.ϵ1
  @. rd.ϵ_d[rd.ilow] += rd.ϵ_l / rd.x_m_lvar
  @. rd.ϵ_d[rd.iupp] -= rd.ϵ_u / rd.uvar_m_x
  rd.ϵ1 ./= rd.D
  if rd.uplo == :L
    mul!(rd.ϵ_p, rd.A, rd.ϵ1)
  else
    mul!(rd.ϵ_p, rd.A', rd.ϵ1)
  end
  @. rd.ϵ_p = rd.ϵK1 - rd.ϵ_p
  return (norm(rd.ϵ_d, Inf) ≤ min(rd.atol, rd.rtol * rd.rcNorm0 * μ / rd.μ0) &&
          norm(rd.ϵ_p, Inf) ≤ min(rd.atol, rd.rtol * rd.rbNorm0 * μ / rd.μ0) &&
          norm(rd.ϵ_l, Inf) ≤ min(rd.atol, rd.rtol * norm(rd.ξ_l, Inf) * μ / rd.μ0) &&
          norm(rd.ϵ_u, Inf) ≤ min(rd.atol, rd.rtol * norm(rd.ξ_u, Inf) * μ / rd.μ0))
end


mutable struct PreallocatedDataK1Krylov{T <: Real, S, L <: LinearOperator, Ksol <: KrylovSolver,
  Pr <: PreconditionerData,
  R <: Union{K1ToK3Residuals{T, S}, Int}} <: PreallocatedDataNormalKrylov{T, S}
  pdat::Pr
  D::S
  rhs::S
  rhs_scale::Bool
  regu::Regularization{T}
  δv::Vector{T}
  K::L # augmented matrix (LinearOperator)         
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

function opK1prod!(
  res::AbstractVector{T},
  D::AbstractVector{T},
  A::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  vtmp::AbstractVector{T},
  α::T,
  β::T,
  uplo::Symbol,
) where {T}
  if uplo == :U
    mul!(vtmp, A, v)
    vtmp ./= D
    mul!(res, A', vtmp, α, β)
    res .+= (α * δv[1]) .* v
  else
    mul!(vtmp, A', v)
    vtmp ./= D
    mul!(res, A, vtmp, α, β)
    res .+= (α * δv[1]) .* v
  end
end

function PreallocatedData(
  sp::K1KrylovParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}
  D = similar(fd.c, id.nvar)
  # init Regularization values
  if iconf.mode == :mono
    regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), :classic)
    # Regularization(T(0.), T(0.), T(sp.ρ_min), T(sp.δ_min), :classic)
    D .= T(1.0e0) / 2
  else
    regu =
      Regularization(T(sp.ρ0), T(sp.δ0), T(sqrt(eps(T)) * 1e0), T(sqrt(eps(T)) * 1e0), :classic)
    D .= T(1.0e-2)
  end

  δv = [regu.δ] # put it in a Vector so that we can modify it without modifying opK2prod!
  K = LinearOperator(
    T,
    id.ncon,
    id.ncon,
    true,
    true,
    (res, v, α, β) -> opK1prod!(res, D, fd.A, δv, v, similar(fd.c), α, β, fd.uplo),
  )

  rhs = similar(fd.c, id.ncon)
  KS = init_Ksolver(K, rhs, sp)
  pdat = PreconditionerData(sp, id, fd, regu, D, K)
  rd = sp.k3_resid ? ToK3Residuals(K, rhs, itd, pt, id, sp, fd, D) : 0

  return PreallocatedDataK1Krylov(
    pdat,
    D,
    rhs,
    sp.rhs_scale,
    regu,
    δv,
    K, #K
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
  pad::PreallocatedDataK1Krylov{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  step::Symbol,
) where {T <: Real}
  pad.rhs .= @views dd[(id.nvar + 1):end]
  if fd.uplo == :U
    @views mul!(pad.rhs, fd.A', dd[1:(id.nvar)] ./ pad.D, one(T), one(T))
  else
    @views mul!(pad.rhs, fd.A, dd[1:(id.nvar)] ./ pad.D, one(T), one(T))
  end
  if pad.rhs_scale
    rhsNorm = kscale!(pad.rhs)
  end
  (step !== :cc) && (pad.kiter = 0)
  ksolve!(
    pad.KS,
    pad.K,
    pad.rhs,
    pad.pdat.P,
    verbose = 0,
    atol = pad.atol,
    rtol = pad.rtol,
    itmax = pad.itmax,
    callback = pad.rd == 0 ? solver -> false : solver -> pad.rd(solver, itd.σ, itd.μ, res.rbNorm, res.rcNorm, dd),
    cb_only = pad.cb_only,
  )
  update_kresiduals_history!(res, pad.K, pad.KS.x, pad.rhs)
  pad.kiter += niterations(pad.KS)
  if pad.rhs_scale
    kunscale!(pad.KS.x, rhsNorm)
  end

  if fd.uplo == :U
    @views mul!(dd[1:(id.nvar)], fd.A, pad.KS.x, one(T), -one(T))
  else
    @views mul!(dd[1:(id.nvar)], fd.A', pad.KS.x, one(T), -one(T))
  end
  dd[1:(id.nvar)] ./= pad.D
  dd[(id.nvar + 1):end] .= pad.KS.x
  return 0
end

function update_pad!(
  pad::PreallocatedDataK1Krylov{T},
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

  pad.δv[1] = pad.regu.δ

  pad.D .= pad.regu.ρ
  @. pad.D[id.ilow] += pt.s_l / itd.x_m_lvar
  @. pad.D[id.iupp] += pt.s_u / itd.uvar_m_x
  pad.δv[1] = pad.regu.δ

  update_preconditioner!(pad.pdat, pad, itd, pt, id, fd, cnts)

  return 0
end
