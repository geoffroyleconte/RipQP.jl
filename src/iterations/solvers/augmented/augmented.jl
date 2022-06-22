abstract type AugmentedParams{T} <: SolverParams{T} end

abstract type AugmentedKrylovParams{T, PT <: AbstractPreconditioner} <: AugmentedParams{T} end

abstract type PreallocatedDataAugmented{T <: Real, S} <: PreallocatedData{T, S} end

abstract type PreallocatedDataAugmentedLDL{T <: Real, S} <: PreallocatedDataAugmented{T, S} end

uses_krylov(pad::PreallocatedDataAugmentedLDL) = false

include("K2LDL.jl")
include("K2_5LDL.jl")
include("K2LDLDense.jl")

abstract type PreallocatedDataAugmentedKrylov{T <: Real, S} <: PreallocatedDataAugmented{T, S} end

uses_krylov(pad::PreallocatedDataAugmentedKrylov) = true

abstract type PreallocatedDataAugmentedKrylovStructured{T <: Real, S} <:
              PreallocatedDataAugmented{T, S} end

mutable struct AugmentedToK3Residuals{T, S <: AbstractVector{T}, M}
  K::M
  rhs::S
  formul::Symbol
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
  K::M,
  rhs::S,
  itd::IterData{T, S},
  pt::Point{T, S},
  id::QM_IntData,
  sp::AugmentedParams,
) where {T, S, M} = AugmentedToK3Residuals{T, S, M}(
  K,
  rhs,
  :K2,
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

function set_rd_init_res!(rd::AugmentedToK3Residuals{T}, μ::T, rbNorm0::T, rcNorm0::T) where {T}
  rd.μ0 = μ
  rd.rbNorm0 = rbNorm0
  rd.rcNorm0 = rcNorm0
end

function (rd::AugmentedToK3Residuals)(solver::KrylovSolver{T}, σ::T, μ::T, 
                                      rbNorm::T, rcNorm::T) where {T}
  mul!(rd.ϵK2, rd.K, solver.x)
  rd.ϵK2 .-= rd.rhs
  rd.ϵ_p .= @views rd.ϵK2[(rd.nvar+1): end]
  @. rd.ξ_l = -rd.s_l * rd.x_m_lvar + σ * μ
  @. rd.ξ_u = -rd.s_u * rd.uvar_m_x + σ * μ
  @. rd.Δs_l = @views (σ * μ - rd.s_l * solver.x[rd.ilow]) / rd.x_m_lvar - rd.s_l
  @. rd.Δs_u = @views (σ * μ + rd.s_u * solver.x[rd.iupp]) / rd.uvar_m_x - rd.s_u
  @. rd.ϵ_l = @views rd.s_l * solver.x[rd.ilow] + rd.x_m_lvar * rd.Δs_l + rd.s_l * rd.x_m_lvar - σ * μ
  @. rd.ϵ_u = @views -rd.s_u * solver.x[rd.iupp] + rd.uvar_m_x * rd.Δs_u + rd.s_u * rd.uvar_m_x - σ * μ
  if rd.formul == :K2
    rd.ϵ_d .= @views rd.ϵK2[1:rd.nvar]
    @. rd.ϵ_d[rd.ilow] += rd.ϵ_l / rd.x_m_lvar
    @. rd.ϵ_d[rd.iupp] -= rd.ϵ_u / rd.uvar_m_x
  end
  # println("rNorm true = ", norm(rd.ϵK2, 2), " xNorm true = ", norm(solver.x))
  # println("ϵd = ", norm(rd.ϵ_d, Inf), " ϵ_p = ", norm(rd.ϵ_p, Inf), " ϵ_l = ", norm(rd.ϵ_l, Inf),
  #   " ϵ_u = ", norm(rd.ϵ_u, Inf))
  return (norm(rd.ϵ_d, Inf) ≤ min(T(1.0e-4), rd.rtol * rd.rcNorm0 * μ / rd.μ0) &&
          norm(rd.ϵ_p, Inf) ≤ min(T(1.0e-4), rd.rtol * rd.rbNorm0 * μ / rd.μ0) &&
          norm(rd.ϵ_l, Inf) ≤ min(T(1.0e-4), rd.rtol * norm(rd.ξ_l, Inf) * μ / rd.μ0) &&
          norm(rd.ϵ_u, Inf) ≤ min(T(1.0e-4), rd.rtol * norm(rd.ξ_u, Inf) * μ / rd.μ0))
end

include("K2Krylov.jl")
include("K2_5Krylov.jl")
include("K2Structured.jl")
include("K2_5Structured.jl")
