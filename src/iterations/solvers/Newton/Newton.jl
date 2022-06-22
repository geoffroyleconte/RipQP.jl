abstract type NewtonParams{T} <: SolverParams{T} end

abstract type NewtonKrylovParams{T, PT <: AbstractPreconditioner} <: NewtonParams{T} end

abstract type PreallocatedDataNewton{T <: Real, S} <: PreallocatedData{T, S} end

abstract type PreallocatedDataNewtonKrylov{T <: Real, S} <: PreallocatedDataNewton{T, S} end

uses_krylov(pad::PreallocatedDataNewtonKrylov) = true

abstract type PreallocatedDataNewtonKrylovStructured{T <: Real, S} <: PreallocatedDataNewton{T, S} end

mutable struct NewtonToK3Residuals{T, S <: AbstractVector{T}, M}
  K::M
  rhs::S
  formul::Symbol
  atol::T
  rtol::T
  ϵK3::S
  μ0::T
  rbNorm0::T
  rcNorm0::T
  nvar::Int
  ncon::Int
  nlow::Int
  nupp::Int
end

ToK3Residuals(
  K::M,
  rhs::S,
  itd::IterData{T, S},
  pt::Point{T, S},
  id::QM_IntData,
  sp::NewtonParams,
) where {T, S, M} = NewtonToK3Residuals{T, S, M}(
  K,
  rhs,
  :K3,
  T(sp.atol_min),
  T(sp.rtol_min),
  S(undef, id.nvar + id.ncon + id.nlow + id.nupp),
  zero(T),
  zero(T),
  zero(T),
  id.nvar,
  id.ncon,
  id.nlow,
  id.nupp,
)

function set_rd_init_res!(rd::NewtonToK3Residuals{T}, μ::T, rbNorm0::T, rcNorm0::T) where {T}
  rd.μ0 = μ
  rd.rbNorm0 = rbNorm0
  rd.rcNorm0 = rcNorm0
end

function (rd::NewtonToK3Residuals)(solver::KrylovSolver{T}, σ::T, μ::T, rbNorm::T, rcNorm::T) where {T}
  mul!(rd.ϵK3, rd.K, solver.x)
  rd.ϵK3 .-= rd.rhs
  ϵ_dNorm = @views norm(rd.ϵK3[1:rd.nvar], Inf)
  ϵ_pNorm = @views norm(rd.ϵK3[(rd.nvar + 1): (rd.nvar + rd.ncon)], Inf)
  ϵ_lNorm = @views norm(rd.ϵK3[(rd.nvar + rd.ncon + 1): (rd.nvar + rd.ncon + rd.nlow)], Inf)
  ϵ_uNorm = @views norm(rd.ϵK3[(rd.nvar + rd.ncon + rd.nlow + 1): (rd.nvar + rd.ncon + rd.nlow + rd.nupp)], Inf)
  ξ_lNorm = @views norm(rd.rhs[(rd.nvar + rd.ncon + 1): (rd.nvar + rd.ncon + rd.nlow)], Inf)
  ξ_uNorm = @views norm(rd.rhs[(rd.nvar + rd.ncon + rd.nlow + 1): (rd.nvar + rd.ncon + rd.nlow + rd.nupp)], Inf)
  return (ϵ_dNorm ≤ min(T(1.0e-4), rd.rtol * rd.rcNorm0 * μ / rd.μ0) &&
          ϵ_pNorm ≤ min(T(1.0e-4), rd.rtol * rd.rbNorm0 * μ / rd.μ0) &&
          ϵ_lNorm ≤ min(T(1.0e-4), rd.rtol * ξ_lNorm * μ / rd.μ0) &&
          ϵ_uNorm ≤ min(T(1.0e-4), rd.rtol * ξ_uNorm * μ / rd.μ0))
end

include("K3Krylov.jl")
include("K3SKrylov.jl")
include("K3_5Krylov.jl")
# utils for K3_5 gpmr
include("K3_5gpmr_utils.jl")
include("K3SStructured.jl")
include("K3_5Structured.jl")
