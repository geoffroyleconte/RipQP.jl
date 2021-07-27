export K2KrylovReguParams

"""
Type to use the K2 formulation with a Krylov method, using the package 
[`Krylov.jl`](https://github.com/JuliaSmoothOptimizers/Krylov.jl). 
The outer constructor 

    K2KrylovParams(; kmethod = :minres, preconditioner = :Jacobi, 
                   ratol = 1.0e-10, rrtol = 1.0e-10)

creates a [`RipQP.SolverParams`](@ref) that should be used to create a [`RipQP.InputConfig`](@ref).
The available methods are:
- `:minres`
- `:minres_qlp`

The list of available preconditioners for this solver is displayed here: [`RipQP.PreconditionerDataK2`](@ref).
"""
struct K2KrylovReguParams <: SolverParams
  kmethod::Symbol
  preconditioner::Symbol
  atol0::Float64
  rtol0::Float64
  atol_min::Float64
  rtol_min::Float64
  ρ_min::Float64
  δ_min::Float64
end

function K2KrylovReguParams(;
  kmethod::Symbol = :minres,
  preconditioner::Symbol = :Jacobi,
  atol0::T = 1.0e-4,
  rtol0::T = 1.0e-4,
  atol_min::T = 1.0e-10,
  rtol_min::T = 1.0e-10,
  ρ_min::T = 1e3 * sqrt(eps()),
  δ_min::T = 1e4 * sqrt(eps()),
) where {T <: Real}
  return K2KrylovReguParams(kmethod, preconditioner, atol0, rtol0, atol_min, rtol_min, ρ_min, δ_min)
end

function solveK2basic(AT0, rhs1, rhs2, M0, δ, atol, rtol)

  T = typeof(δ)
  D = Diagonal(1 ./ sqrt.(diag(M0)))
  AT = D * AT0
  A = AT' .+ zero(T)
  rhs1D = D * rhs1
  M = D * M0 * D

  n, m = size(AT)
  Abis = [A √δ*I]
  Mbis = [M spzeros(n, m); spzeros(m,n) I]
  nnzA = length(AT.nzval)

  # Attention : Geoffroy maxvolume_basis s'applique sur A et non Abis !!!
  # La permutation basis s'applique directement sur A si A est de rang plein sinon c'est sur Abis.
  # Le Abis de BASICLU est :
  # m, n = size(A)
  # rowmax = maximum(abs.(A), dims=2)[:]
  # rowmax = max.(rowmax, 1.)
  # AI = [A spdiagm(0 => lindeptol.*rowmax)] où lindeptol = 1e-8 par défaut
  basis, F = maxvolume_basis(A, volumetol=1.1, maxpass=2)
  rank_deficiency = count(>(n), basis)
  # println(rank_deficiency)

  notbasis = setdiff(1:m+n, basis)
  B = Abis[:, basis]
  N = Abis[:, notbasis]
  M11 = Diagonal(Mbis[basis,basis])
  M11 = Diagonal(Vector(M11.diag))
  M22 = Diagonal(Mbis[notbasis,notbasis])
  M22 = Diagonal(Vector(M22.diag))
  T = B * inv(M11) * B'
  Kbis = [-M22 N'; N T]

  rhs1bis = [rhs1D; zeros(m)][notbasis]
  rhs2bis = rhs2 + B * inv(M11) * [rhs1D; zeros(m)][basis]
  rhsbis = [rhs1bis; rhs2bis]

  lu_B = lu(B)
  opM = LinearOperator(Float64, n, n, true, true, (y, v, α, β) -> y .= M22 \ v)
  opB = LinearOperator(Float64, m, m, true, true, (y, w, α, β) -> y .= lu_B' \ (M11 * (lu_B \ w)))
  opD = BlockDiagonalOperator(opM, opB)

  x2N, stats2 = minres_qlp(Symmetric(Kbis, :U), rhsbis, M = opD, atol=atol, rtol=rtol, 
                          history=false, verbose=0)
  println(norm(rhsbis - Kbis * x2N))

  x2 = zeros(n+m)
  x2[notbasis] .= x2N[1:length(notbasis)]
  x2[basis] .= inv(M11) * (B' * x2N[length(notbasis)+1: end] - [rhs1D; zeros(m)][basis]) 
  return x2[1:n] .* D.diag[1:n], x2N[length(notbasis)+1: end]
end

function opK2prod!(
  res::AbstractVector{T},
  nvar::Int,
  Q::AbstractMatrix{T},
  D::AbstractVector{T},
  AT::AbstractMatrix{T},
  δv::AbstractVector{T},
  v::AbstractVector{T},
  α::T,
  β::T,
) where {T}
  @views mul!(res[1:nvar], Q, v[1:nvar], -α, β)
  res[1:nvar] .+= α .* D .* v[1:nvar]
  @views mul!(res[1:nvar], AT, v[(nvar + 1):end], α, one(T))
  @views mul!(res[(nvar + 1):end], AT', v[1:nvar], α, β)
  res[(nvar + 1):end] .+= @views (α * δv[1]) .* v[(nvar + 1):end]
end

function PreallocatedData(
  sp::K2KrylovReguParams,
  fd::QM_FloatData{T},
  id::QM_IntData,
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  D = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu =
      Regularization(T(sqrt(eps()) * 1e5), T(sqrt(eps()) * 1e5), T(sp.ρ_min), T(sp.δ_min), :classic)
    D .= -T(1.0e0) / 2
  else
    regu = Regularization(
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps(T)) * 1e0),
      T(sqrt(eps(T)) * 1e0),
      :classic,
    )
    D .= -T(1.0e-2)
  end
  δv = [regu.δ] # put it in a Vector so that we can modify it without modifying opK2prod!
  K = LinearOperator(
    T,
    id.nvar + id.ncon,
    id.nvar + id.ncon,
    true,
    true,
    (res, v, α, β) -> opK2prod!(res, id.nvar, fd.Q, D, fd.AT, δv, v, α, β),
  )

  rhs = similar(fd.c, id.nvar + id.ncon)
  kstring = string(sp.kmethod)
  KS = eval(KSolver(sp.kmethod))(K, rhs)

  pdat = eval(sp.preconditioner)(id, fd, regu, D, K)

  return eval(:PreallocatedData_K2minresRegu)(
    pdat,
    D,
    rhs,
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
  pad::PreallocatedData_K2minresRegu{T},
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
  pad.rhs .= (step == :aff) ? dda.Δxy_aff : pad.rhs .= itd.Δxy
  # rhsNorm = norm(pad.rhs)
  # if rhsNorm != zero(T)
  #   pad.rhs ./= rhsNorm
  # end
  # 
  if cnts.k > 5
    dx, dy = solveK2basic(fd.AT, pad.rhs[1:id.nvar], pad.rhs[id.nvar+1: end], 
                          fd.Q - Diagonal(pad.D), pad.regu.δ, pad.atol, pad.rtol)
    if step == :aff
      dda.Δxy_aff[1: id.nvar] .= dx
      dda.Δxy_aff[id.nvar+1 : end] .= dy
    else
      itd.Δxy[1: id.nvar] .= dx
      itd.Δxy[id.nvar+1 : end] .= dy
    end
  else
    ksolve!(pad.KS, pad.K, pad.rhs, pad.pdat.P, verbose = 0, atol = pad.atol, rtol = pad.rtol)
    if step == :aff
      dda.Δxy_aff .= pad.KS.x
    else
      itd.Δxy .= pad.KS.x
    end
  # if typeof(res) <: ResidualsHistory
  #   mul!(res.KΔxy, pad.K, pad.KS.x) # krylov residuals
  #   res.Kres = res.KΔxy .- pad.rhs
  # end
  # if rhsNorm != zero(T)
  #   pad.KS.x .*= rhsNorm
  # end
  end

  return 0
end

function update_pad!(
  pad::PreallocatedData_K2minresRegu{T},
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

  pad.D .= -pad.regu.ρ
  pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
  pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x
  pad.δv[1] = pad.regu.δ

  update_preconditioner!(pad.pdat, pad, itd, pt, id, fd, cnts)

  return 0
end
