# (A D⁻¹ Aᵀ + δI) Δy = A D⁻¹ ξ₁ + ξ₂ 
# where D = s_l (x - lvar)⁻¹ + s_u (uvar - x)⁻¹ + ρI,
# and the right hand side of K2 is rhs = [ξ₁]
#                                        [ξ₂]
# With the least-squares formulation:
# (A D⁻¹ Aᵀ + δI) Δ̃y = A D⁻¹ (ξ₁ - Aᵀ ξ₂ / δ),
# where Δ̃y = Δy - ξ₂ / δ
export K1LSKrylovParams

"""
Type to use the K1 formulation with a Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K1LSKrylovParams(; uplo = :L, kmethod = :lsqr, preconditioner = :Identity, 
                   ratol = 1.0e-10, rrtol = 1.0e-10)

creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
The available methods are:
- `:lsqr`

"""
struct K1LSKrylovParams <: SolverParams
  uplo::Symbol
  kmethod::Symbol
  preconditioner::Symbol
  atol0::Float64
  rtol0::Float64
  atol_min::Float64
  rtol_min::Float64
  ρ_min::Float64
  δ_min::Float64
end

function K1LSKrylovParams(;
  uplo::Symbol = :L,
  kmethod::Symbol = :cg,
  preconditioner::Symbol = :Identity,
  atol0::T = 1.0e-4,
  rtol0::T = 1.0e-4,
  atol_min::T = 1.0e-10,
  rtol_min::T = 1.0e-10,
  ρ_min::T = 1e3 * sqrt(eps()),
  δ_min::T = 1e4 * sqrt(eps()),
) where {T <: Real}
  return K1LSKrylovParams(
    uplo,
    kmethod,
    preconditioner,
    atol0,
    rtol0,
    atol_min,
    rtol_min,
    ρ_min,
    δ_min,
  )
end

mutable struct PreallocatedDataK1LSKrylov{T <: Real, S, L <: LinearOperator, Ksol <: KrylovSolver} <:
               PreallocatedDataNormalKrylov{T, S}
  D::S
  rhs::S
  vtmp::S
  regu::Regularization{T}
  δv::Vector{T}
  K::L # augmented matrix (LinearOperator)         
  KS::Ksol
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
end

function opK1LSprod!(
  res::AbstractVector{T},
  D::AbstractVector{T},
  A::AbstractMatrix{T},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  α::T,
  β::T,
  uplo::Symbol,
) where {T}
  if uplo == :L
    mul!(res, A, v ./ sqrt.(D), α, β)
  end
end

function opK1LStprod!(
  res::AbstractVector{T},
  D::AbstractVector{T},
  A::AbstractMatrix{T},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  vtmp::AbstractVector{T},
  α::T,
  β::T,
  uplo::Symbol,
) where {T}
  if uplo == :L
    mul!(vtmp, A', v)
    if β == zero(T)
      res .= α .* vtmp ./ sqrt.(D)
    else
      res .= α .* vtmp ./ sqrt.(D) .+ β .* res
    end
  end
end

function PreallocatedData(
  sp::K1LSKrylovParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}
  D = similar(fd.c, id.nvar)
  # init Regularization values
  if iconf.mode == :mono
    regu =
      Regularization(T(sqrt(eps()) * 1e5), T(sqrt(eps()) * 1e5), T(sp.ρ_min), T(sp.δ_min), :classic)
    D .= T(1.0e0) / 2
  else
    regu = Regularization(
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps(T)) * 1e0),
      T(sqrt(eps(T)) * 1e0),
      :classic,
    )
    D .= T(1.0e-2)
  end

  δv = [regu.δ] # put it in a Vector so that we can modify it without modifying opK2prod!

  vtmp = similar(fd.c)
  K = LinearOperator(
    T,
    id.ncon,
    id.nvar,
    false,
    false,
    (res, v, α, β) -> opK1LSprod!(res, D, fd.A, δv, v, α, β, fd.uplo),
    (res, v, α, β) -> opK1LStprod!(res, D, fd.A, δv, v, vtmp, α, β, fd.uplo),
  )

  rhs = similar(fd.c, id.nvar)
  kstring = string(sp.kmethod)
  KS = eval(KSolver(sp.kmethod))(K', rhs)

  return PreallocatedDataK1LSKrylov(
    D,
    rhs,
    vtmp,
    regu,
    δv,
    K, #K
    KS,
    sp.atol0,
    sp.rtol0,
    sp.atol_min,
    sp.rtol_min,
  )
end

function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK1LSKrylov{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  T0::DataType,
  step::Symbol,
) where {T <: Real}
  pad.rhs .= @views dd[1:id.nvar] ./ sqrt.(pad.D)
  @views mul!(pad.rhs, pad.K', dd[(id.nvar + 1):end], -one(T) /pad.regu.δ, one(T))
  pad.K.nprod = 0
  rhsNorm = kscale!(pad.rhs)
  ksolve!(pad.KS, pad.K', pad.rhs, I(id.nvar), verbose = 0, atol = pad.atol, rtol = pad.rtol, λ = sqrt(pad.regu.δ))
  update_kresiduals_history_LS!(res, pad.K, pad.KS.x, pad.rhs, pad.vtmp, pad.regu.δ)
  kunscale!(pad.KS.x, rhsNorm)
  pad.KS.x .+= dd[(id.nvar + 1):end] ./ pad.regu.δ
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
  pad::PreallocatedDataK1LSKrylov{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  T0::DataType,
) where {T <: Real}
  if cnts.k != 0
    update_regu!(pad.regu)
  end

  if pad.atol > pad.atol_min
    pad.atol /= 10
  end
  if pad.rtol > pad.rtol_min
    pad.rtol /= 10
  end

  pad.δv[1] = pad.regu.δ

  pad.D .= pad.regu.ρ
  pad.D[id.ilow] .+= pt.s_l ./ itd.x_m_lvar
  pad.D[id.iupp] .+= pt.s_u ./ itd.uvar_m_x
  pad.δv[1] = pad.regu.δ

  # update_preconditioner!(pad.pdat, pad, itd, pt, id, fd, cnts)

  return 0
end
