export K2_5KrylovParams

"""
Type to use the K2.5 formulation with a Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K2_5KrylovParams(; uplo = :L, kmethod = :minres, preconditioner = Identity(),
                     rhs_scale = true,
                     atol0 = 1.0e-4, rtol0 = 1.0e-4, 
                     atol_min = 1.0e-10, rtol_min = 1.0e-10,
                     ρ0 = sqrt(eps()) * 1e5, δ0 = sqrt(eps()) * 1e5, 
                     ρ_min = 1e2 * sqrt(eps()), δ_min = 1e2 * sqrt(eps()),
                     itmax = 0, mem = 20)

creates a [`RipQP.SolverParams`](@ref).
The available methods are:
- `:minres`
- `:minres_qlp`
- `:symmlq`
"""
mutable struct K2_5KrylovParams{T, PT} <: AugmentedKrylovParams{T, PT}
  uplo::Symbol
  kmethod::Symbol
  preconditioner::PT
  rhs_scale::Bool
  form_mat::Bool
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
end

function K2_5KrylovParams{T}(;
  uplo::Symbol = :L,
  kmethod::Symbol = :minres,
  preconditioner::AbstractPreconditioner = Identity(),
  rhs_scale::Bool = true,
  form_mat::Bool= false,
  atol0::T = T(1.0e-4),
  rtol0::T = T(1.0e-4),
  atol_min::T = T(1.0e-10),
  rtol_min::T = T(1.0e-10),
  ρ0::T = T(sqrt(eps()) * 1e5),
  δ0::T = T(sqrt(eps()) * 1e5),
  ρ_min::T = T(1e2 * sqrt(eps())),
  δ_min::T = T(1e3 * sqrt(eps())),
  itmax::Int = 0,
  mem::Int = 20,
) where {T <: Real}
  if typeof(preconditioner) <: LDL
    if !form_mat
      form_mat = true
      @info "changed form_mat to true to use this preconditioner"
    end
    uplo_fact = get_uplo(preconditioner.fact_alg)
    if uplo != uplo_fact
      uplo = uplo_fact
      @info "changed uplo to :$uplo_fact to use this preconditioner"
    end
  end
  return K2_5KrylovParams(
    uplo,
    kmethod,
    preconditioner,
    rhs_scale,
    form_mat,
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
  )
end

K2_5KrylovParams(; kwargs...) = K2_5KrylovParams{Float64}(; kwargs...)

mutable struct PreallocatedDataK2_5Krylov{
  T <: Real,
  S,
  M <: Union{LinearOperator{T}, AbstractMatrix{T}},
  MT <: Union{MatrixTools{T}, Int},
  Pr <: PreconditionerData,
  Ksol <: KrylovSolver,
} <: PreallocatedDataAugmentedKrylov{T, S}
  pdat::Pr
  D::S                                  # temporary top-left diagonal
  sqrtX1X2::S # vector to scale K2 to K2.5
  tmp1::S # temporary vector for products
  tmp2::S # temporary vector for products
  rhs::S
  rhs_scale::Bool
  regu::Regularization{T}
  δv::Vector{T}
  K::M # augmented matrix
  K_scaled::Bool # only useful with form_mat = true
  mt::MT      
  KS::Ksol
  kiter::Int
  atol::T
  rtol::T
  atol_min::T
  rtol_min::T
  itmax::Int
end

function opK2_5prod!(
  res::AbstractVector{T},
  nvar::Int,
  Q::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  D::AbstractVector{T},
  A::Union{AbstractMatrix{T}, AbstractLinearOperator{T}},
  sqrtX1X2::AbstractVector{T},
  tmp1::AbstractVector{T},
  tmp2::AbstractVector{T},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  α::T,
  β::T,
  uplo::Symbol,
) where {T}
  tmp2 .= @views sqrtX1X2 .* v[1:nvar]
  mul!(tmp1, Q, tmp2, -α, zero(T))
  tmp1 .= @views sqrtX1X2 .* tmp1 .+ α .* D .* v[1:nvar]
  if β == zero(T)
    res[1:nvar] .= tmp1
  else
    res[1:nvar] .= @views tmp1 .+ β .* res[1:nvar]
  end
  if uplo == :U
    @views mul!(tmp1, A, v[(nvar + 1):end], α, zero(T))
    res[1:nvar] .+= sqrtX1X2 .* tmp1
    @views mul!(res[(nvar + 1):end], A', tmp2, α, β)
  else
    @views mul!(tmp1, A', v[(nvar + 1):end], α, zero(T))
    res[1:nvar] .+= sqrtX1X2 .* tmp1
    @views mul!(res[(nvar + 1):end], A, tmp2, α, β)
  end
  res[(nvar + 1):end] .+= @views (α * δv[1]) .* v[(nvar + 1):end]
end

function PreallocatedData(
  sp::K2_5KrylovParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  pt::Point{T},
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  D = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu = Regularization(T(sp.ρ0), T(sp.δ0), T(sp.ρ_min), T(sp.δ_min), :classic)
    D .= -T(1.0e0) / 2
  else
    regu =
      Regularization(T(sp.ρ0), T(sp.δ0), T(sqrt(eps(T)) * 1e0), T(sqrt(eps(T)) * 1e0), :classic)
    D .= -T(1.0e-2)
  end
  sqrtX1X2 = fill!(similar(D), one(T))
  tmp1 = similar(D)
  tmp2 = similar(D)
  δv = [regu.δ] # put it in a Vector so that we can modify it without modifying opK2prod!
  if sp.form_mat
    diag_Q = get_diag_Q(fd.Q)
    K = create_K2(id, D, fd.Q.data, fd.A, diag_Q, regu, fd.uplo)
    diagind_K = get_diagind_K(K, sp.uplo)
    Deq = Diagonal(Vector{T}(undef, 0))
    C_eq = Diagonal(Vector{T}(undef, 0))
    mt = MatrixTools(diag_Q, diagind_K, Deq, C_eq)
  else
    K = LinearOperator(
      T,
      id.nvar + id.ncon,
      id.nvar + id.ncon,
      true,
      true,
      (res, v, α, β) ->
        opK2_5prod!(res, id.nvar, fd.Q, D, fd.A, sqrtX1X2, tmp1, tmp2, δv, v, α, β, fd.uplo),
    )
    mt = 0
  end

  rhs = similar(fd.c, id.nvar + id.ncon)
  KS = @timeit_debug to "krylov solver setup" init_Ksolver(K, rhs, sp)
  pdat = @timeit_debug to "preconditioner setup" PreconditionerData(sp, id, fd, regu, D, K)

  return PreallocatedDataK2_5Krylov(
    pdat,
    D,
    sqrtX1X2,
    tmp1,
    tmp2,
    rhs,
    sp.rhs_scale,
    regu,
    δv,
    K, #K
    false,
    mt,
    KS, #K_fact
    0,
    T(sp.atol0),
    T(sp.rtol0),
    T(sp.atol_min),
    T(sp.rtol_min),
    sp.itmax,
  )
end

function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedDataK2_5Krylov{T},
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

  # erase dda.Δxy_aff only for affine predictor step with PC method
  pad.rhs[1:(id.nvar)] .= @views dd[1:(id.nvar)] .* pad.sqrtX1X2
  pad.rhs[(id.nvar + 1):end] .= @views dd[(id.nvar + 1):end]
  if pad.rhs_scale
    rhsNorm = kscale!(pad.rhs)
  end
  if step !== :cc
    @timeit_debug to "preconditioner update" update_preconditioner!(
      pad.pdat,
      pad,
      itd,
      pt,
      id,
      fd,
      cnts,
    )
    pad.kiter = 0
  end
  @timeit_debug to "Krylov solve" ksolve!(
    pad.KS,
    pad.K,
    pad.rhs,
    pad.pdat.P,
    verbose = 0,
    atol = pad.atol,
    rtol = pad.rtol,
    itmax = pad.itmax,
  )
  update_kresiduals_history!(res, pad.K, pad.KS.x, pad.rhs)
  pad.kiter += niterations(pad.KS)
  if pad.rhs_scale
    kunscale!(pad.KS.x, rhsNorm)
  end
  pad.KS.x[1:(id.nvar)] .*= pad.sqrtX1X2

  dd .= pad.KS.x

   # update regularization and restore K. Cannot be done in update_pad since x-lvar and uvar-x will change.
  if typeof(pad.K) <: Symmetric{T, <:Union{SparseMatrixCSC{T}, SparseMatrixCOO{T}}} &&
    (step == :cc || step == :IPF) && pad.K_scaled
    # restore J for next iteration
    # pad.D .= one(T)
    # pad.D[id.ilow] ./= sqrt.(itd.x_m_lvar)
    # pad.D[id.iupp] ./= sqrt.(itd.uvar_m_x)
    lrdiv_K!(pad.K, pad.sqrtX1X2, id.nvar)
    pad.K_scaled = false
  end

  return 0
end

function update_pad!(
  pad::PreallocatedDataK2_5Krylov{T},
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

  # K2.5
  pad.sqrtX1X2 .= one(T)
  pad.sqrtX1X2[id.ilow] .*= sqrt.(itd.x_m_lvar)
  pad.sqrtX1X2[id.iupp] .*= sqrt.(itd.uvar_m_x)
  pad.D .= -pad.regu.ρ
  pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
  pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x

  pad.δv[1] = pad.regu.δ

  if typeof(pad.K) <: Symmetric{T, <:Union{SparseMatrixCSC{T}, SparseMatrixCOO{T}}}
    pad.D[pad.mt.diag_Q.nzind] .-= pad.mt.diag_Q.nzval
    update_diag_K11!(pad.K, pad.D, pad.mt.diagind_K, id.nvar)
    update_diag_K22!(pad.K, pad.regu.δ, pad.mt.diagind_K, id.nvar, id.ncon)
    lrmultilply_K!(pad.K, pad.sqrtX1X2, id.nvar)
    pad.K_scaled = true
  else
    pad.D .*= pad.sqrtX1X2 .^2
  end

  return 0
end

function convertpad(
  ::Type{<:PreallocatedData{T}},
  pad::PreallocatedDataK2_5Krylov{T_old},
  sp_old::K2_5KrylovParams,
  sp_new::K2_5KrylovParams,
  id::QM_IntData,
  fd::Abstract_QM_FloatData,
  T0::DataType,
) where {T <: Real, T_old <: Real}
  D = convert(Array{T}, pad.D)
  sqrtX1X2 = convert(Array{T}, pad.sqrtX1X2)
  tmp1 = convert(Array{T}, pad.tmp1)
  tmp2 = convert(Array{T}, pad.tmp2)
  regu = convert(Regularization{T}, pad.regu)
  regu.ρ_min = T(sp_new.ρ_min)
  regu.δ_min = T(sp_new.δ_min)
  K = Symmetric(convert(eval(typeof(pad.K.data).name.name){T, Int}, pad.K.data), sp_new.uplo)
  rhs = similar(D, id.nvar + id.ncon)
  δv = [regu.δ]
  mt = convert(MatrixTools{T}, pad.mt)
  mt.Deq.diag .= one(T)
  regu_precond = convert(Regularization{sp_new.preconditioner.T}, pad.regu)
  regu_precond.regul = :dynamic
  K_fact =
    (sp_new.preconditioner.T != sp_old.preconditioner.T) ?
    convertldl(sp_new.preconditioner.T, pad.pdat.K_fact) : pad.pdat.K_fact
  pdat = PreconditionerData(sp_new, K_fact, id.nvar, id.ncon, regu_precond, K)
  KS = init_Ksolver(K, rhs, sp_new)

  return PreallocatedDataK2_5Krylov(
    pdat,
    D,
    sqrtX1X2,
    tmp1,
    tmp2,
    rhs,
    sp_new.rhs_scale,
    regu,
    δv,
    K, #K
    pad.K_scaled,
    mt,
    KS,
    0,
    T(sp_new.atol0),
    T(sp_new.rtol0),
    T(sp_new.atol_min),
    T(sp_new.rtol_min),
    sp_new.itmax,
  )
end

