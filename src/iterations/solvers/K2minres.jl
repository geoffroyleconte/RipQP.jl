export K2minresParams

struct K2minresParams <: SolverParams
    preconditioner :: Symbol
    atol :: Float64 
    rtol :: Float64
end

function K2minresParams(; preconditioner = :LLDL, atol :: T = 1.0e-6, rtol :: T = 1.0e-6) where {T<:Real} 
    return K2minresParams(preconditioner, atol, rtol)
end

mutable struct PreallocatedData_K2minres{T<:Real} <: PreallocatedData{T} 
    # pdat             :: PreconditionerDataK2{T}
    # P                :: LinearOperator{T}
    D                :: Vector{T}                                        # temporary top-left diagonal
    rhs              :: Vector{T}
    regu             :: Regularization{T}
    diagind_Q        :: StepRange{Int64, Int64} # Q diagonal indices
    K                :: SparseMatrixCSC{T,Int} # augmented matrix          
    MS               :: MinresSolver{T, Vector{T}}
    diagind_K        :: Vector{Int} # diagonal indices of J
end


function PreallocatedData(sp :: K2minresParams, fd :: QM_FloatData{T}, id :: QM_IntData, 
                          iconf :: InputConfig{Tconf}) where {T<:Real, Tconf<:Real}

  # init Regularization values
  D = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu = Regularization(
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps()) * 1e5),
      1e1 * sqrt(eps(T)),
      1e1 * sqrt(eps(T)),
      :classic,
    )
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
  # diag_Q = get_diag_Q(fd.Q.colptr, fd.Q.rowval, fd.Q.nzval, id.nvar)
  diagind_Q = diagind(fd.Q)
  K = [.-fd.Q .+ D                       fd.AT;
       spzeros(T, id.ncon, id.nvar)  regu.δ * I]
  diagind_K = get_diag_sparseCSC(K.colptr, id.ncon+id.nvar)
  
  rhs = zeros(T, id.nvar+id.ncon)
  MS = MinresSolver(K, rhs)

  # pdat, P = eval(sp.preconditioner)(id, regu, D, K)

  return PreallocatedData_K2minres(#pdat,
                                  #  P,
                                   D,
                                   rhs, 
                                   regu,
                                   diagind_Q, #diag_Q
                                   K, #K
                                   MS, #K_fact
                                   diagind_K, #diagind_K
                                   )
end

function solver!(pad :: PreallocatedData_K2minres{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
                 fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, cnts :: Counters, T0 :: DataType, 
                 step :: Symbol) where {T<:Real} 

  # erase dda.Δxy_aff only for affine predictor step with PC method
  if step == :aff 
    pad.rhs .= dda.Δxy_aff 
  else
    pad.rhs .= itd.Δxy
  end

  rhsNorm = norm(pad.rhs)
  if rhsNorm != zero(T)
    pad.rhs ./= rhsNorm
  end
  (pad.MS.x, pad.MS.stats) = minres!(pad.MS, Symmetric(pad.K, :U), pad.rhs)#, M=pad.P)
  if rhsNorm != zero(T)
    pad.MS.x .*= rhsNorm
  end

  if step == :aff 
    dda.Δxy_aff .= pad.MS.x
  else
    itd.Δxy .= pad.MS.x
  end

  return 0
end

function update_pad!(pad :: PreallocatedData_K2minres{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
                     fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, cnts :: Counters, T0 :: DataType) where {T<:Real}

    if cnts.k != 0
        update_regu!(pad.regu) 
    end

    pad.D .= -pad.regu.ρ .- fd.Q[pad.diagind_Q]
    pad.K.nzval[view(pad.diagind_K, id.nvar+1:id.ncon+id.nvar)] .= pad.regu.δ
    pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
    pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x
    # pad.D[pad.diag_Q.nzind] .-= pad.diag_Q.nzval
    pad.K.nzval[view(pad.diagind_K,1:id.nvar)] = pad.D 

    # update_preconditioner!(pad.pdat, pad, itd, pt, id, cnts)
    
    # pad.opK = PreallocatedLinearOperator(pad.y_opK, Symmetric(pad.K, :U))
    # pad.opK = PreallocatedLinearOperator(pad.y_opK, Symmetric(pad.K+pad.K'-Diagonal(pad.K), :U))

    return 0
end