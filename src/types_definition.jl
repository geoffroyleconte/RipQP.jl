import Base: convert

export InputConfig, InputTol, SystemWrite, SolverParams, PreallocatedData

# problem: min 1/2 x'Qx + c'x + c0     s.t.  Ax = b,  lvar ≤ x ≤ uvar
abstract type Abstract_QM_FloatData{T <: Real, S, Ssp} end

mutable struct QM_FloatData{T <: Real, S, Ssp} <: Abstract_QM_FloatData{T, S, Ssp}
  Q::Ssp # size nvar * nvar
  AT::Ssp # size ncon * nvar, using Aᵀ is easier to form systems
  b::S # size ncon
  c::S # size nvar
  c0::T
  lvar::S # size nvar
  uvar::S # size nvar
end

mutable struct QM_IntData
  ilow::Vector{Int} # indices of finite elements in lvar
  iupp::Vector{Int} # indices of finite elements in uvar
  irng::Vector{Int} # indices of finite elements in both lvar and uvar
  ifree::Vector{Int} # indices of infinite elements in both lvar and uvar
  ncon::Int # number of equality constraints after SlackModel! (= size of b)
  nvar::Int # number of variables
  nlow::Int # length(ilow)
  nupp::Int # length(iupp)
end

"""
Abstract type for tuning the parameters of the different solvers. 
Each solver has its own `SolverParams` type.

The `SolverParams` currently implemented within RipQP are:

- [`RipQP.K2LDLParams`](@ref)
- [`RipQP.K2_5LDLParams`](@ref)

"""
abstract type SolverParams end

"""
Type to write the matrix (.mtx format) and the right hand side (.rhs format) of the system to solve at each iteration.

- `write::Bool`: activate/deactivate writing of the system 
- `name::String`: name of the sytem to solve 
- `kfirst::Int`: first iteration where a system should be written
- `kgap::Int`: iteration gap between two problem writings

The constructor

    SystemWrite(; write::Bool = false, name::String = "", 
                kfirst::Int = 0, kgap::Int = 1)

returns a `SystemWrite` structure that should be used to tell RipQP to save the system. 
See the tutorial for more information. 
"""
struct SystemWrite
  write::Bool
  name::String
  kfirst::Int
  kgap::Int
end

SystemWrite(; write::Bool = false, name::String = "", kfirst::Int = 0, kgap::Int = 1) =
  SystemWrite(write, name, kfirst, kgap)

"""
Type to specify the configuration used by RipQP.

- `mode :: Symbol`: should be `:mono` to use the mono-precision mode, or `:multi` to use
    the multi-precision mode (start in single precision and gradually transitions
    to `T0`)
- `scaling :: Bool`: activate/deactivate scaling of A and Q in `QM0`
- `normalize_rtol :: Bool = true` : if `true`, the primal and dual tolerance for the stopping criteria 
    are normalized by the initial primal and dual residuals
- `kc :: Int`: number of centrality corrections (set to `-1` for automatic computation)
- `refinement :: Symbol` : should be `:zoom` to use the zoom procedure, `:multizoom` to use the zoom procedure 
    with multi-precision (then `mode` should be `:multi`), `ref` to use the QP refinement procedure, `multiref` 
    to use the QP refinement procedure with multi_precision (then `mode` should be `:multi`), or `none` to avoid 
    refinements
- `sp :: SolverParams` : choose a solver to solve linear systems that occurs at each iteration and during the 
    initialization, see [`RipQP.SolverParams`](@ref)
- `solve_method :: Symbol` : method used to solve the system at each iteration, use `solve_method = :PC` to 
    use the Predictor-Corrector algorithm (default), and use `solve_method = :IPF` to use the Infeasible Path 
    Following algorithm
- `w :: SystemWrite`: configure writing of the systems to solve (no writing is done by default), see [`RipQP.SystemWrite`](@ref)

The constructor

    iconf = InputConfig(; mode :: Symbol = :mono, scaling :: Bool = true, 
                        normalize_rtol :: Bool = true, kc :: I = 0, 
                        refinement :: Symbol = :none, max_ref :: I = 0, 
                        sp :: SolverParams = K2LDLParams(),
                        solve_method :: Symbol = :PC, 
                        w :: SystemWrite = SystemWrite()) where {I<:Integer}

returns a `InputConfig` struct that shall be used to solve the input `QuadraticModel` with RipQP.
"""
struct InputConfig{I <: Integer}
  mode::Symbol
  scaling::Bool
  normalize_rtol::Bool # normalize the primal and dual tolerance to the initial starting primal and dual residuals
  kc::I # multiple centrality corrections, -1 = automatic computation

  # QP refinement 
  refinement::Symbol
  max_ref::I # maximum number of refinements

  # Functions to choose formulations
  sp::SolverParams
  solve_method::Symbol

  # write systems 
  w::SystemWrite
end

function InputConfig(;
  mode::Symbol = :mono,
  scaling::Bool = true,
  normalize_rtol::Bool = true,
  kc::I = 0,
  refinement::Symbol = :none,
  max_ref::I = 0,
  sp::SolverParams = K2LDLParams(),
  solve_method::Symbol = :PC,
  w::SystemWrite = SystemWrite(),
) where {I <: Integer}
  mode == :mono || mode == :multi || error("mode should be :mono or :multi")
  refinement == :zoom ||
    refinement == :multizoom ||
    refinement == :ref ||
    refinement == :multiref ||
    refinement == :none ||
    error("not a valid refinement parameter")
  solve_method == :IPF &&
    kc != 0 &&
    error("IPF method should not be used with centrality corrections")

  return InputConfig{I}(mode, scaling, normalize_rtol, kc, refinement, max_ref, sp, solve_method, w)
end

"""
Type to specify the tolerances used by RipQP.

- `max_iter :: Int`: maximum number of iterations
- `ϵ_pdd`: relative primal-dual difference tolerance
- `ϵ_rb`: primal tolerance
- `ϵ_rc`: dual tolerance
- `max_iter32`, `ϵ_pdd32`, `ϵ_rb32`, `ϵ_rc32`: same as `max_iter`, `ϵ_pdd`, `ϵ_rb` and
    `ϵ_rc`, but used for switching from single precision to double precision. They are
    only usefull when `mode=:multi`
- `max_iter64`, `ϵ_pdd64`, `ϵ_rb64`, `ϵ_rc64`: same as `max_iter`, `ϵ_pdd`, `ϵ_rb` and
    `ϵ_rc`, but used for switching from double precision to quadruple precision. They
    are only usefull when `mode=:multi` and `T0=Float128`
- `ϵ_rbz` : primal transition tolerance for the zoom procedure, (used only if `refinement=:zoom`)
- `ϵ_Δx`: step tolerance for the current point estimate (note: this criterion
    is currently disabled)
- `ϵ_μ`: duality measure tolerance (note: this criterion is currently disabled)
- `max_time`: maximum time to solve the QP

The constructor

    itol = InputTol(;max_iter :: I = 200, max_iter32 :: I = 40, max_iter64 :: I = 180, 
                     ϵ_pdd :: T = 1e-8, ϵ_pdd32 :: T = 1e-2, ϵ_pdd64 :: T = 1e-4, 
                     ϵ_rb :: T = 1e-6, ϵ_rb32 :: T = 1e-4, ϵ_rb64 :: T = 1e-5, ϵ_rbz :: T = 1e-3,
                     ϵ_rc :: T = 1e-6, ϵ_rc32 :: T = 1e-4, ϵ_rc64 :: T = 1e-5,
                     ϵ_Δx :: T = 1e-16, ϵ_μ :: T = 1e-9) where {T<:Real, I<:Integer}

returns a `InputTol` struct that initializes the stopping criteria for RipQP. 
The 32 and 64 characters refer to the stopping criteria in `:multi` mode for the transitions from `Float32` to `Float64` 
and `Float64` to `Float128` (if the input `QuadraticModel` is in `Float128`) respectively.
"""
struct InputTol{T <: Real, I <: Integer}
  # maximum number of iterations
  max_iter::I
  max_iter32::I # only in multi mode
  max_iter64::I # only in multi mode with T0 = Float128

  # relative primal-dual gap tolerance
  ϵ_pdd::T
  ϵ_pdd32::T # only in multi mode 
  ϵ_pdd64::T # only in multi mode with T0 = Float128

  # primal residual
  ϵ_rb::T
  ϵ_rb32::T # only in multi mode 
  ϵ_rb64::T # only in multi mode with T0 = Float128
  ϵ_rbz::T # only when using zoom refinement

  # dual residual
  ϵ_rc::T
  ϵ_rc32::T # only in multi mode 
  ϵ_rc64::T # only in multi mode with T0 = Float128

  # unused residuals (for now)
  ϵ_μ::T
  ϵ_Δx::T

  # maximum time for resolution
  max_time::T
end

function InputTol(;
  max_iter::I = 200,
  max_iter32::I = 40,
  max_iter64::I = 180,
  ϵ_pdd::T = 1e-8,
  ϵ_pdd32::T = 1e-2,
  ϵ_pdd64::T = 1e-4,
  ϵ_rb::T = 1e-6,
  ϵ_rb32::T = 1e-4,
  ϵ_rb64::T = 1e-5,
  ϵ_rbz::T = 1e-5,
  ϵ_rc::T = 1e-6,
  ϵ_rc32::T = 1e-4,
  ϵ_rc64::T = 1e-5,
  ϵ_Δx::T = 1e-16,
  ϵ_μ::T = 1e-9,
  max_time::T = 1200.0,
) where {T <: Real, I <: Integer}
  return InputTol{T, I}(
    max_iter,
    max_iter32,
    max_iter64,
    ϵ_pdd,
    ϵ_pdd32,
    ϵ_pdd64,
    ϵ_rb,
    ϵ_rb32,
    ϵ_rb64,
    ϵ_rbz,
    ϵ_rc,
    ϵ_rc32,
    ϵ_rc64,
    ϵ_μ,
    ϵ_Δx,
    max_time,
  )
end

mutable struct Tolerances{T <: Real}
  pdd::T  # primal-dual difference (relative)
  rb::T  # primal residuals tolerance
  rc::T  # dual residuals tolerance
  tol_rb::T  # ϵ_rb * (1 + ||r_b0||)
  tol_rc::T  # ϵ_rc * (1 + ||r_c0||)
  μ::T  # duality measure
  Δx::T
  normalize_rtol::Bool # true if normalize_rtol=true, then tol_rb, tol_rc = ϵ_rb, ϵ_rc
end

mutable struct Point{T <: Real, S}
  x::S # size nvar
  y::S # size ncon
  s_l::S # size nlow (useless zeros corresponding to infinite lower bounds are not stored)
  s_u::S # size nupp (useless zeros corresponding to infinite upper bounds are not stored)
  function Point(
    x::AbstractVector{T},
    y::AbstractVector{T},
    s_l::AbstractVector{T},
    s_u::AbstractVector{T},
  ) where {T <: Real}
    S = typeof(x)
    return new{T, S}(x, y, s_l, s_u)
  end
end

convert(::Type{Point{T, S}}, pt) where {T <: Real, S} = Point(
  convert(S.name.wrapper{T, 1}, pt.x),
  convert(S.name.wrapper{T, 1}, pt.y),
  convert(S.name.wrapper{T, 1}, pt.s_l),
  convert(S.name.wrapper{T, 1}, pt.s_u),
)

mutable struct Residuals{T <: Real, S}
  rb::S # primal residuals Ax - b
  rc::S # dual residuals -Qx + Aᵀy + s_l - s_u
  rbNorm::T # ||rb||
  rcNorm::T # ||rc||
  function Residuals(
    rb::AbstractVector{T},
    rc::AbstractVector{T},
    rbNorm::T,
    rcNorm::T,
  ) where {T <: Real}
    S = typeof(rb)
    return new{T, S}(rb, rc, rbNorm, rcNorm)
  end
end

convert(::Type{Residuals{T, S}}, res) where {T <: Real, S} = Residuals(
  convert(S.name.wrapper{T, 1}, res.rb),
  convert(S.name.wrapper{T, 1}, res.rc),
  convert(T, res.rbNorm),
  convert(T, res.rcNorm),
  convert(T, res.n_Δx),
)

# LDLFactorization conversion function
convertldl(T::DataType, K_fact) = LDLFactorizations.LDLFactorization(
  K_fact.__analyzed,
  K_fact.__factorized,
  K_fact.__upper,
  K_fact.n,
  K_fact.parent,
  K_fact.Lnz,
  K_fact.flag,
  K_fact.P,
  K_fact.pinv,
  K_fact.Lp,
  K_fact.Cp,
  K_fact.Ci,
  K_fact.Li,
  convert(Array{T}, K_fact.Lx),
  convert(Array{T}, K_fact.d),
  convert(Array{T}, K_fact.Y),
  K_fact.pattern,
  T(K_fact.r1),
  T(K_fact.r2),
  T(K_fact.tol),
  K_fact.n_d,
)

mutable struct Regularization{T <: Real}
  ρ::T       # curent top-left regularization parameter
  δ::T       # cureent bottom-right regularization parameter
  ρ_min::T       # ρ minimum value 
  δ_min::T       # δ minimum value 
  regul::Symbol  # regularization mode (:classic, :dynamic, or :none)
end

convert(::Type{Regularization{T}}, regu::Regularization{T0}) where {T <: Real, T0 <: Real} =
  Regularization(T(regu.ρ), T(regu.δ), T(regu.ρ_min), T(regu.δ_min), regu.regul)

abstract type IterData{T <: Real, S} end

mutable struct IterDataCPU{T <: Real, S} <: IterData{T, S}
  Δxy::S # Newton step [Δx; Δy]
  Δs_l::S
  Δs_u::S
  x_m_lvar::S # x - lvar
  uvar_m_x::S # uvar - x
  Qx::S
  ATy::S # Aᵀy
  Ax::S
  xTQx_2::T # xᵀQx
  cTx::T # cᵀx
  pri_obj::T # 1/2 xᵀQx + cᵀx + c0                                             
  dual_obj::T # -1/2 xᵀQx + yᵀb + s_lᵀlvar - s_uᵀuvar + c0
  μ::T # duality measure (s_lᵀ(x-lvar) + s_uᵀ(uvar-x)) / (nlow+nupp)
  pdd::T # primal dual difference (relative) pri_obj - dual_obj / pri_obj
  l_pdd::Vector{T} # list of the 5 last pdd
  mean_pdd::T # mean of the 5 last pdd
  qp::Bool # true if qp false if lp
end

mutable struct IterDataGPU{T <: Real, S} <: IterData{T, S}
  Δxy::S # Newton step [Δx; Δy]
  Δs_l::S
  Δs_u::S
  x_m_lvar::S # x - lvar
  uvar_m_x::S # uvar - x
  Qx::S
  ATy::S # Aᵀy
  Ax::S
  xTQx_2::T # xᵀQx
  cTx::T # cᵀx
  pri_obj::T # 1/2 xᵀQx + cᵀx + c0                                             
  dual_obj::T # -1/2 xᵀQx + yᵀb + s_lᵀlvar - s_uᵀuvar + c0
  μ::T # duality measure (s_lᵀ(x-lvar) + s_uᵀ(uvar-x)) / (nlow+nupp)
  pdd::T # primal dual difference (relative) pri_obj - dual_obj / pri_obj
  l_pdd::Vector{T} # list of the 5 last pdd
  mean_pdd::T # mean of the 5 last pdd
  qp::Bool # true if qp false if lp
  store_vpri::S
  store_vdual_l::S
  store_vdual_u::S
  IterDataGPU(Δxy::S, Δs_l::S, Δs_u::S, x_m_lvar::S, uvar_m_x::S, Qx::S, ATy::S, 
              Ax::S, xTQx_2::T, cTx::T, 
              pri_obj::T, dual_obj::T, 
              μ::T, pdd::T, l_pdd::Vector{T}, 
              mean_pdd::T, qp::Bool) where {T <: Real, S} = 
    new{T, S}(Δxy, Δs_l, Δs_u, x_m_lvar, uvar_m_x, Qx, ATy, Ax, xTQx_2, cTx, 
              pri_obj, dual_obj, μ, pdd, l_pdd, mean_pdd, qp, 
              similar(Qx), similar(Δs_l), similar(Δs_u))
end

function IterData(Δxy, Δs_l, Δs_u, x_m_lvar, uvar_m_x, Qx, ATy, Ax, xTQx_2, cTx, 
                  pri_obj, dual_obj, μ, pdd, l_pdd, mean_pdd, qp)
  if typeof(Δxy) <: Vector
    return IterDataCPU(Δxy, Δs_l, Δs_u, x_m_lvar, uvar_m_x, Qx, ATy, Ax, xTQx_2, cTx, 
                       pri_obj, dual_obj, μ, pdd, l_pdd, mean_pdd, qp)
  else
    return IterDataGPU(Δxy, Δs_l, Δs_u, x_m_lvar, uvar_m_x, Qx, ATy, Ax, xTQx_2, cTx, 
                       pri_obj, dual_obj, μ, pdd, l_pdd, mean_pdd, qp)
  end
end

convert(::Type{IterData{T, S}}, itd::IterData{T0, S0}) where {T <: Real, S, T0 <: Real, S0} =
  IterData(
    convert(S.name.wrapper{T, 1}, itd.Δxy),
    convert(S.name.wrapper{T, 1}, itd.Δs_l),
    convert(S.name.wrapper{T, 1}, itd.Δs_u),
    convert(S.name.wrapper{T, 1}, itd.x_m_lvar),
    convert(S.name.wrapper{T, 1}, itd.uvar_m_x),
    convert(S.name.wrapper{T, 1}, itd.Qx),
    convert(S.name.wrapper{T, 1}, itd.ATy),
    convert(S.name.wrapper{T, 1}, itd.Ax),
    convert(T, itd.xTQx_2),
    convert(T, itd.cTx),
    convert(T, itd.pri_obj),
    convert(T, itd.dual_obj),
    convert(T, itd.μ),
    convert(T, itd.pdd),
    convert(Array{T, 1}, itd.l_pdd),
    convert(T, itd.mean_pdd),
    itd.qp,
  )

abstract type PreallocatedData{T <: Real, S} end

mutable struct StopCrit{T}
  optimal::Bool
  small_μ::Bool
  tired::Bool
  max_iter::Int
  max_time::T
  start_time::T
  Δt::T
end

mutable struct Counters
  c_catch::Int # safety try:cath
  c_pdd::Int # maximum number of δ_min reductions when pdd does not change
  k::Int # iter count
  km::Int # iter relative to precision: if k+=1 and T==Float128, km +=16  (km+=4 if T==Float64 and km+=1 if T==Float32)
  kc::Int # maximum corrector steps
  max_ref::Int # maximum number of refinements
  c_ref::Int # current number of refinements
  w::SystemWrite # store SystemWrite data
end
