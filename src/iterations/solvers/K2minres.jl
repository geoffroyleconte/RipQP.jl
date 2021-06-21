export K2minresParams

struct K2minresParams <: SolverParams
    preconditioner :: Symbol
    ratol           :: Float64 
    rrtol           :: Float64
end

function K2minresParams(; preconditioner = :Schur, ratol :: T = 1.0e-10, rrtol :: T = 1.0e-10) where {T<:Real} 
    return K2minresParams(preconditioner, ratol, rrtol)
end

mutable struct PreallocatedData_K2minres{T<:Real, S, Ssp} <: PreallocatedData{T, S} 
    pdat             :: PreconditionerDataK2{T, S}
    D                :: S                                  # temporary top-left diagonal
    rhs              :: S
    regu             :: Regularization{T}
    diagind_Q        :: StepRange{Int64, Int64} # Q diagonal indices
    K                :: Ssp # augmented matrix          
    MS               :: MinresSolver{T, S}
    diagind_K        :: Vector{Int} # diagonal indices of J
    ratol            :: T
    rrtol            :: T
end


function PreallocatedData(sp :: K2minresParams, fd :: QM_FloatData{T}, id :: QM_IntData, 
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

  return PreallocatedData_K2minres(pdat,
                                   D,
                                   rhs, 
                                   regu,
                                   diagind_Q, #diag_Q
                                   K, #K
                                   MS, #K_fact
                                   diagind_K, #diagind_K
                                   sp.ratol,
                                   sp.rrtol,
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
  (pad.MS.x, pad.MS.stats) = minres!(pad.MS, Symmetric(pad.K, :U), pad.rhs, M=pad.pdat.P, 
                                     verbose=0, atol=zero(T), rtol=zero(T), ratol=pad.ratol, rrtol=pad.rrtol)
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
    pad.K.nzval[view(pad.diagind_K,1:id.nvar)] = pad.D 

    update_preconditioner!(pad.pdat, pad, itd, pt, id, fd, cnts)

    return 0
end