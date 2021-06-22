export K2bicgstabParams

struct K2bicgstabParams <: SolverParams
  preconditioner :: Symbol
  ratol           :: Float64 
  rrtol           :: Float64       
end

function K2bicgstabParams(; preconditioner = :BlockJacobi, ratol :: T = 1.0e-10, rrtol :: T = 1.0e-10, gpu :: Bool = false) where {T<:Real} 
    return K2bicgstabParams(preconditioner, ratol, rrtol)
end

mutable struct PreallocatedData_K2bicgstab{T<:Real, S, Ssp} <: PreallocatedData{T, S} 
  pdat             :: PreconditionerDataK2{T, S}
  D                :: S                                      # temporary top-left diagonal
  rhs              :: S
  regu             :: Regularization{T}
  diag_Q           :: S # Q diagonal indices
  K                :: Ssp # augmented matrix          
  MS               :: BicgstabSolver{T, S}
  diagind_K1       :: Vector{Int}
  diagind_K2       :: Vector{Int}
  ratol            :: T
  rrtol            :: T
end


function PreallocatedData(sp :: K2bicgstabParams, fd :: QM_FloatData{T}, id :: QM_IntData, 
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

  diag_Q = similar(fd.c)
  if typeof(fd.c) <: Vector
    diag_Q .= fd.Q[diagind(fd.Q)]
    K = [.-fd.Q .+ Diagonal(D)                       fd.AT;
        spzeros(T, id.ncon, id.nvar)  regu.δ * I]
    K .= K .+ K' .- Diagonal(K)
    diagind_K1, diagind_K2 = get_diagK_ptr(K.colptr, K.rowval, id.nvar, id.ncon)
  else
    diag_Q .= get_diag_Q(fd.Q)
    K, diagind_K1, diagind_K2 = create_K_GPU(fd.Q, D, regu.δ, fd.AT, id.nvar, id.ncon)
  end
  
  rhs = similar(fd.c, id.nvar+id.ncon)
  MS = BicgstabSolver(K, rhs)

  pdat = eval(sp.preconditioner)(id, fd, regu, D, K)

  return PreallocatedData_K2bicgstab(pdat,
                                     D,
                                     rhs, 
                                     regu,
                                     diag_Q, #diag_Q
                                     K, #K
                                     MS, #K_fact
                                     diagind_K1,
                                     diagind_K2,
                                     sp.ratol,
                                     sp.rrtol,
                                     )
end

function solver!(pad :: PreallocatedData_K2bicgstab{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
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
  (pad.MS.x, stats) = bicgstab!(pad.MS, Symmetric(pad.K, :U), pad.rhs, M=pad.pdat.P, 
                                     verbose=0)#, atol=zero(T), rtol=zero(T), ratol=pad.ratol, rrtol=pad.rrtol)
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

function update_pad!(pad :: PreallocatedData_K2bicgstab{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
                     fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, cnts :: Counters, T0 :: DataType) where {T<:Real}

    if cnts.k != 0
        update_regu!(pad.regu) 
    end

    pad.D .= -pad.regu.ρ .- pad.diag_Q
    pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
    pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x
    if typeof(fd.c) <: Vector
      pad.K.nzval[pad.diagind_K1] = pad.D
      pad.K.nzval[pad.diagind_K2] .= pad.regu.δ
    else
      update_diagK_gpu!(pad.K, pad.diagind_K1, pad.diagind_K2, pad.D, pad.regu.δ)
    end

    update_preconditioner!(pad.pdat, pad, itd, pt, id, fd, cnts)

    return 0
end

function get_diagK_ptr(K_colptr, K_rowval, nvar, ncon)
  diagindK1, diagindK2 = zeros(Int, nvar), zeros(Int, ncon)
  for j=1:nvar
    for k=K_colptr[j]: (K_colptr[j+1]-1)
      if j == K_rowval[k]
        diagindK1[j] = k
      end
    end
  end
  for j=(nvar+1): (nvar+ncon)
    for k=K_colptr[j]: (K_colptr[j+1]-1)
      if j == K_rowval[k]
        diagindK2[j-nvar] = k
      end
    end
  end
  return diagindK1, diagindK2
end