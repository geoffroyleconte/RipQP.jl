export QRSep

struct QRSep <: SolverParams
  regul::Symbol
end

function QRSep(; regul::Symbol = :classic)
  regul == :classic ||
    regul == :dynamic ||
    regul == :none ||
    error("regul should be :classic or :dynamic or :none")
  return QRSep(regul)
end

mutable struct PreallocatedData_QRSep{T <: Real, S} <: PreallocatedData_LDL{T, S}
  D::S # temporary top-left diagonal
  regu::Regularization{T}
  diag_Q::SparseVector{T, Int} # Q diagonal
  K::SparseMatrixCSC{T, Int} # augmented matrix 
  K_fact::LDLFactorizations.LDLFactorization{T, Int, Int, Int} # factorized matrix
  fact_fail::Bool # true if factorization failed 
  diagind_K::Vector{Int} # diagonal indices of J
  closetobnd::Vector{Bool}
end

# outer constructor
function PreallocatedData(
  sp::QRSep,
  fd::QM_FloatData{T},
  id::QM_IntData,
  iconf::InputConfig{Tconf},
) where {T <: Real, Tconf <: Real}

  # init Regularization values
  D = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu = Regularization(
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps()) * 1e5),
      1e-5 * sqrt(eps(T)),
      1e0 * sqrt(eps(T)),
      sp.regul,
    )
    D .= -T(1.0e0) / 2
  else
    regu = Regularization(
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps(T)) * 1e0),
      T(sqrt(eps(T)) * 1e0),
      sp.regul,
    )
    D .= -T(1.0e-2)
  end
  diag_Q = get_diag_Q(fd.Q.colptr, fd.Q.rowval, fd.Q.nzval, id.nvar)
  K = create_K2(id, D, fd.Q, fd.A, diag_Q, regu)

  diagind_K = get_diag_sparseCSC(K.colptr, id.ncon + id.nvar)
  K_fact = ldl_analyze(Symmetric(K, :U))
  if regu.regul == :dynamic
    Amax = @views norm(K.nzval[diagind_K], Inf)
    regu.ρ, regu.δ = -T(eps(T)^(3 / 4)), T(eps(T)^(0.45))
    K_fact.r1, K_fact.r2 = regu.ρ, regu.δ
    K_fact.tol = Amax * T(eps(T))
    K_fact.n_d = id.nvar
  elseif regu.regul == :none
    regu.ρ, regu.δ = zero(T), zero(T)
  end
  K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
  K_fact.__factorized = true
  closetobnd = fill(false, id.nvar)

  return PreallocatedData_QRSep(
    D,
    regu,
    diag_Q, #diag_Q
    K, #K
    K_fact, #K_fact
    false,
    diagind_K, #diagind_K
    closetobnd,
  )
end

# function used to solve problems
# solver LDLFactorization
function solver!(
  dd::AbstractVector{T},
  pad::PreallocatedData_QRSep{T},
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

  if all(pad.closetobnd .== false)
    # ldiv!(pad.K_fact, dd)
    rhs = copy(dd)
    rhsNorm = norm(rhs)
    if rhsNorm != zero(T)
      rhs ./= rhsNorm
    end
    res, stats = minres(Symmetric(pad.K,:U), rhs, verbose = 0)
    if rhsNorm != zero(T)
      res .*= rhsNorm
    end
    dd .= res
  else
    println("qr sum(closetobnd) = $(sum(pad.closetobnd))")
    Kb, Kn = get_sub_data(pad, fd, id)
    if rem(cnts.k, 2) == 0
      rhsidx = [pad.closetobnd; fill(false, id.ncon)]
      invertrhsidx = invertbool.(rhsidx)
      rhs = dd[invertrhsidx]
      rhsNorm = norm(rhs)
      if rhsNorm != zero(T)
        rhs ./= rhsNorm
      end
      ddb, stats = minres(Kb, rhs, verbose = 0)
      if rhsNorm != zero(T)
        ddb .*= rhsNorm
      end
      println(norm((Kb * ddb) - dd[invertrhsidx]))
      println(sum(invertrhsidx))
      dd[invertrhsidx] .= ddb
      dd[rhsidx] .= zero(T)
    else
      rhsidx = [pad.closetobnd; fill(true, id.ncon)]
      invertrhsidx = invertbool.(rhsidx)
      rhs = dd[rhsidx]
      rhsNorm = norm(rhs)
      if rhsNorm != zero(T)
        rhs ./= rhsNorm
      end
      ddn, stats = minres(Kn, rhs, verbose = 0)
      if rhsNorm != zero(T)
        ddn .*= rhsNorm
      end
      println(norm((Kn * ddn) - dd[rhsidx]))
      println(sum(invertrhsidx))
      dd[rhsidx] .= ddn
      dd[invertrhsidx] .= zero(T)
    end
  end

  return 0
end

function update_pad!(
  pad::PreallocatedData_QRSep{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  cnts::Counters,
  T0::DataType,
) where {T <: Real}
  if pad.regu.regul == :classic && cnts.k != 0 # update ρ and δ values, check K diag magnitude 
    out = update_regu_diagK2!(
      pad.regu,
      pad.K.nzval,
      pad.diagind_K,
      id.nvar,
      itd.pdd,
      itd.l_pdd,
      itd.mean_pdd,
      cnts,
      T,
      T0,
    )
    out == 1 && return out
  end

  out = factorize_K2!(
    pad.K,
    pad.K_fact,
    pad.D,
    pad.diag_Q,
    pad.diagind_K,
    pad.regu,
    pt.s_l,
    pt.s_u,
    itd.x_m_lvar,
    itd.uvar_m_x,
    id.ilow,
    id.iupp,
    id.ncon,
    id.nvar,
    cnts,
    itd.qp,
    T,
    T0,
  ) # update D and factorize K

  if out == 1
    pad.fact_fail = true
    return out
  end

  if rem(cnts.k, 2) == 0
    check_dist_bnd!(pad.closetobnd, itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, id.ilow, id.iupp, id.nlow, id.nupp, itd.μ)
  end

  return 0
end

function check_dist_bnd!(closetobnd::Vector{Bool}, x_m_lvar::AbstractVector{T}, uvar_m_x::AbstractVector{T}, 
                         s_l::AbstractVector{T}, s_u::AbstractVector{T}, ilow, iupp, nlow, nupp, μ::T) where {T <: Real}

  if μ ≤ 1e-2
    for i=1:nlow
      if x_m_lvar[i] / s_l[i] ≤ μ
        closetobnd[ilow[i]] = true
      else
        closetobnd[ilow[i]] = false
      end
    end
    for i=1:nupp
      if uvar_m_x[i] / s_u[i] ≤ μ
        closetobnd[iupp[i]] = true
      else
        closetobnd[iupp[i]] = false
      end
    end
  end
end

invertbool(b::Bool) = (b == true) ? false : true

function get_sub_data(pad::PreallocatedData{T}, fd, id) where T # lp only for now
  Kmat = pad.K + pad.K' - Diagonal(pad.K)
  Kb = Kmat[[invertbool.(pad.closetobnd); fill(true, id.ncon)], [invertbool.(pad.closetobnd); fill(true, id.ncon)]]
  nbtrue = sum(pad.closetobnd)
  nvar = length(pad.closetobnd)
  println(size([(-fd.Q + Diagonal(pad.D))[pad.closetobnd, pad.closetobnd];
  spzeros(T, nvar - nbtrue, nbtrue)] ))
  println(size(fd.A))
  notclosetobnd = invertbool.(pad.closetobnd)
  # Kb = [[(-fd.Q + Diagonal(pad.D))[notclosetobnd, notclosetobnd];
  #         spzeros(T, nbtrue, nvar - nbtrue)]                [fd.A[notclosetobnd,:]; fd.A[pad.closetobnd,:]];
  #       fd.A'[:, notclosetobnd]  pad.regu.δ * I]
  Kn = Kmat[[pad.closetobnd; fill(true, id.ncon)], [pad.closetobnd; fill(true, id.ncon)]]
  # Kn = [[spzeros(T, nbtrue, nvar - nbtrue);
  #         (-fd.Q + Diagonal(pad.D))[invertbool.(pad.closetobnd), invertbool.(pad.closetobnd)]]   fd.A;
  #       fd.A'[:, invertbool.(pad.closetobnd)]             pad.regu.δ * I]
  return Kb, Kn
end