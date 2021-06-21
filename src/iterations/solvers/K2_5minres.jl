export K2_5minresParams

struct K2_5minresParams <: SolverParams
    preconditioner :: Symbol
    ratol           :: Float64 
    rrtol           :: Float64
end

function K2_5minresParams(; preconditioner = :Schur, ratol :: T = 1.0e-10, rrtol :: T = 1.0e-10) where {T<:Real} 
    return K2_5minresParams(preconditioner, ratol, rrtol)
end

mutable struct PreallocatedData_K2_5minres{T<:Real} <: PreallocatedData{T} 
    pdat             :: PreconditionerDataK2{T}
    D                :: Vector{T}                                        # temporary top-left diagonal
    rhs              :: Vector{T}
    regu             :: Regularization{T}
    diagind_Q        :: StepRange{Int64, Int64} # Q diagonal indices
    K                :: SparseMatrixCSC{T,Int} # augmented matrix          
    MS               :: MinresSolver{T, Vector{T}}
    diagind_K        :: Vector{Int} # diagonal indices of J
    ratol            :: T
    rrtol            :: T
    K_scaled         :: Bool
    X1X2             :: Vector{T}
end


function PreallocatedData(sp :: K2_5minresParams, fd :: QM_FloatData{T}, id :: QM_IntData, 
                          iconf :: InputConfig{Tconf}) where {T<:Real, Tconf<:Real}

  # init Regularization values
  D = similar(fd.c, id.nvar)
  if iconf.mode == :mono
    regu = Regularization(
      T(sqrt(eps()) * 1e5),
      T(sqrt(eps()) * 1e5),
      1e3 * sqrt(eps(T)),
      1e4 * sqrt(eps(T)),
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
  diagind_Q = diagind(fd.Q)
  K = [.-fd.Q .+ Diagonal(D)                       fd.AT;
       spzeros(T, id.ncon, id.nvar)  regu.δ * I]
  diagind_K = get_diag_sparseCSC(K.colptr, id.ncon+id.nvar)
  
  rhs = similar(fd.c, id.nvar+id.ncon)
  MS = MinresSolver(K, rhs)

  pdat = eval(sp.preconditioner)(id, fd, regu, D, K)

  return PreallocatedData_K2_5minres(pdat,
                                     D,
                                     rhs, 
                                     regu,
                                     diagind_Q, #diag_Q
                                     K, #K
                                     MS, #K_fact
                                     diagind_K, #diagind_K
                                     sp.ratol,
                                     sp.rrtol,
                                     false,
                                     similar(D),
                                     )
end

function solver!(pad :: PreallocatedData_K2_5minres{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
                 fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, cnts :: Counters, T0 :: DataType, 
                 step :: Symbol) where {T<:Real} 

  # erase dda.Δxy_aff only for affine predictor step with PC method
  if step == :aff 
    pad.rhs[1:id.nvar] .= @views dda.Δxy_aff[1:id.nvar] .* pad.X1X2
    pad.rhs[id.nvar+1: end] .= @views dda.Δxy_aff[id.nvar+1: end]
  else
    if pad.K_scaled
      pad.rhs[1:id.nvar] .= @views itd.Δxy[1:id.nvar] .* pad.X1X2
      pad.rhs[id.nvar+1: end] .= @views itd.Δxy[id.nvar+1: end]
    else
      pad.rhs .= itd.Δxy
    end
  end

  rhsNorm = norm(pad.rhs)
  if rhsNorm != zero(T)
    pad.rhs ./= rhsNorm
  end
  (pad.MS.x, pad.MS.stats) = minres!(pad.MS, Symmetric(pad.K, :U), pad.rhs, M=pad.pdat.P, 
                                     verbose=0, atol=zero(T), rtol=zero(T), ratol=pad.ratol, rrtol=pad.rrtol)
  if rhsNorm != zero(T)
    pad.MS.x .*= rhsNorm
  end

  if pad.K_scaled
    pad.MS.x[1: id.nvar] .*= pad.X1X2
  end

  if step == :aff 
    dda.Δxy_aff .= pad.MS.x
  else
    itd.Δxy .= pad.MS.x
  end

  if (step == :cc || step == :IPF) && pad.K_scaled

    out = 0
    if pad.regu.regul == :classic # update ρ and δ values, check K diag magnitude 
        out = update_regu_diagK2_5!(pad.regu, pad.X1X2, itd.pdd, itd.l_pdd, itd.mean_pdd, cnts, T, T0) 
        # update_regu!(pad.regu)
    end

    pad.X1X2 .= one(T) 
    pad.X1X2[id.ilow] ./= sqrt.(itd.x_m_lvar)
    pad.X1X2[id.iupp] ./= sqrt.(itd.uvar_m_x)
    lrmultilply_J!(pad.K.colptr, pad.K.rowval, pad.K.nzval, pad.X1X2, id.nvar)
    pad.K_scaled = false
  end

  return 0
end

function update_pad!(pad :: PreallocatedData_K2_5minres{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
                     fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, cnts :: Counters, T0 :: DataType) where {T<:Real}

    if cnts.k != 0
        update_regu!(pad.regu) 
    end

    pad.D .= -pad.regu.ρ .- fd.Q[pad.diagind_Q]
    pad.K.nzval[view(pad.diagind_K, id.nvar+1:id.ncon+id.nvar)] .= pad.regu.δ
    pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
    pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x
    pad.K.nzval[view(pad.diagind_K,1:id.nvar)] = pad.D 

    # scale K
    pad.X1X2 .= one(T)
    pad.X1X2[id.ilow] .*= sqrt.(itd.x_m_lvar)
    pad.X1X2[id.iupp] .*= sqrt.(itd.uvar_m_x)
    lrmultilply_J!(pad.K.colptr, pad.K.rowval, pad.K.nzval, pad.X1X2, id.nvar) 
    # pad.K.nzval[view(pad.diagind_K,1:id.nvar)] .-= pad.regu.ρ 

    pad.K_scaled = true

    update_preconditioner!(pad.pdat, pad, itd, pt, id, fd, cnts)

    return 0
end