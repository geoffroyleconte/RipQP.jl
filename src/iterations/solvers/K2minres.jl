export K2minresParams

struct K2minresParams <: SolverParams
    preconditioner :: Symbol
    atol :: Float64 
    rtol :: Float64
end

function K2minresParams(; preconditioner = :LLDLData, atol :: T = 1.0e-6, rtol :: T = 1.0e-6) where {T<:Real} 
    return K2minresParams(preconditioner, atol, rtol)
end

mutable struct PreallocatedData_K2minres{T<:Real} <: PreallocatedData{T} 
    pdat             :: PreconditionerDataK2{T}
    P                :: LinearOperator{T}
    D                :: Vector{T}                                        # temporary top-left diagonal
    rhs              :: Vector{T}
    regu             :: Regularization{T}
    diag_Q           :: SparseVector{T,Int} # Q diagonal
    K                :: SparseMatrixCSC{T,Int} # augmented matrix          
    opK              :: PreallocatedLinearOperator{T}
    y_opK            :: Vector{T} # preallocated vector for the LinearOperator
    MS               :: MinresSolver{T, Vector{T}}
    diagind_K        :: Vector{Int} # diagonal indices of J
end

function PreallocatedData(sp :: K2minresParams, fd :: QM_FloatData{T}, id :: QM_IntData, 
                          iconf :: InputConfig{Tconf}) where {T<:Real, Tconf<:Real}

    # init Regularization values
    if iconf.mode == :mono
        regu = Regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), 1e0*sqrt(eps(T)), 1e0*sqrt(eps(T)), :classic)
        D = -T(1.0e0)/2 .* ones(T, id.nvar)
    else
        regu = Regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), T(sqrt(eps(T))*1e0), T(sqrt(eps(T))*1e0), :classic)
        D = -T(1.0e-2) .* ones(T, id.nvar)
    end
    diag_Q = get_diag_Q(fd.Q.colptr, fd.Q.rowval, fd.Q.nzval, id.nvar)
    K = create_K2(id, D, fd.Q, fd.AT, diag_Q, regu)
    diagind_K = get_diag_sparseCSC(K.colptr, id.ncon+id.nvar)
    y_opK = zeros(T, id.nvar+id.ncon)
    opK = PreallocatedLinearOperator(y_opK, Symmetric(K, :U))
    
    rhs = zeros(T, id.nvar+id.ncon)
    MS = MinresSolver(K, rhs)

    pdat, P = eval(sp.preconditioner)(id, regu, D, K)

    return PreallocatedData_K2minres(pdat,
                                     P,
                                     D,
                                     rhs, 
                                     regu,
                                     diag_Q, #diag_Q
                                     K, #K
                                     opK,
                                     y_opK,
                                     MS, #K_fact
                                     diagind_K #diagind_K
                                     )
end

function solver!(pad :: PreallocatedData_K2minres{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
                 fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, cnts :: Counters, T0 :: DataType, 
                 step :: Symbol) where {T<:Real} 

    # erase dda.Δxy_aff only for affine predictor step with PC method
    if step == :aff 
        pad.rhs .= dda.Δxy_aff 
        (pad.MS.x, pad.MS.stats) = minres!(pad.MS, pad.opK, pad.rhs, M=pad.P)
        dda.Δxy_aff .= pad.MS.x
        println(norm(Symmetric(pad.K, :U) * dda.Δxy_aff - pad.rhs))
    else
        pad.rhs .= itd.Δxy
        # (pad.MS.x, pad.MS.stats) = minres!(pad.MS, pad.opK, pad.rhs, M=pad.opiLLDL)
        (pad.MS.x, pad.MS.stats) = minres!(pad.MS, pad.opK, pad.rhs, M=pad.P)
        itd.Δxy .= pad.MS.x
        println(norm(Symmetric(pad.K, :U) * itd.Δxy - pad.rhs))
    end
    return 0
end

function update_pad!(pad :: PreallocatedData_K2minres{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
                     fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, cnts :: Counters, T0 :: DataType) where {T<:Real}

    if cnts.k != 0
        update_regu!(pad.regu) 
    end

    pad.D .= -pad.regu.ρ
    pad.K.nzval[view(pad.diagind_K, id.nvar+1:id.ncon+id.nvar)] .= pad.regu.δ
    pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
    pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x
    pad.D[pad.diag_Q.nzind] .-= pad.diag_Q.nzval
    pad.K.nzval[view(pad.diagind_K,1:id.nvar)] = pad.D 

    update_preconditioner!(pad.pdat, pad, itd, pt, id)
    
    # pad.opK = PreallocatedLinearOperator(pad.y_opK, Symmetric(pad.K, :U))
    pad.opK = PreallocatedLinearOperator(pad.y_opK, Symmetric(pad.K+pad.K'-Diagonal(pad.K), :U))

    return 0
end