export K2_5minresParams

struct K2_5minresParams <: SolverParams
    atol :: Float64 
    rtol :: Float64
end

function K2_5minresParams(; atol :: T = 1.0e-6, rtol :: T = 1.0e-6) where {T<:Real} 
    return K2_5minresParams(atol, rtol)
end

mutable struct PreallocatedData_K2_5minres{T<:Real} <: PreallocatedData{T} 
    D                :: Vector{T}                                        # temporary top-left diagonal
    rhs              :: Vector{T}
    regu             :: Regularization{T}
    diag_Q           :: SparseVector{T,Int} # Q diagonal
    K                :: SparseMatrixCSC{T,Int} # augmented matrix 
    invDiagK         :: Vector{T}
    opK              :: PreallocatedLinearOperator{T}
    y_op             :: Vector{T} # preallocated vector for the LinearOperator
    MS               :: MinresSolver{T, Vector{T}}
    diagind_K        :: Vector{Int} # diagonal indices of J
    K_scaled         :: Bool
end


function PreallocatedData(sp :: K2_5minresParams, fd :: QM_FloatData{T}, id :: QM_IntData, 
                          iconf :: InputConfig{Tconf}) where {T<:Real, Tconf<:Real}

    # init Regularization values
    if iconf.mode == :mono
        regu = Regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), sqrt(eps(T)), sqrt(eps(T)), :classic)
        D = -T(1.0e0)/2 .* ones(T, id.nvar)
    else
        regu = Regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), T(sqrt(eps(T))*1e0), T(sqrt(eps(T))*1e0), :classic)
        D = -T(1.0e-2) .* ones(T, id.nvar)
    end
    diag_Q = get_diag_Q(fd.Q.colptr, fd.Q.rowval, fd.Q.nzval, id.nvar)
    K = create_K2(id, D, fd.Q, fd.AT, diag_Q, regu)
    y_op = zeros(T, id.nvar+id.ncon)
    opK = PreallocatedLinearOperator(y_op, Symmetric(K, :U))

    diagind_K = get_diag_sparseCSC(K.colptr, id.ncon+id.nvar)
    invDiagK = regu.δ.*ones(T, id.nvar+id.ncon)
    invDiagK[1:id.nvar] = .-D
    
    rhs = zeros(T, id.nvar+id.ncon)
    MS = MinresSolver(K, rhs)

    return PreallocatedData_K2_5minres(D,
                                       rhs, 
                                       regu,
                                       diag_Q, #diag_Q
                                       K, #K
                                       invDiagK,
                                       opK,
                                       y_op,
                                       MS, #K_fact
                                       diagind_K, #diagind_K
                                       false
                                       )
end

function solver!(pad :: PreallocatedData_K2_5minres{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
                 fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, cnts :: Counters, T0 :: DataType, 
                 step :: Symbol) where {T<:Real} 

    # erase dda.Δxy_aff only for affine predictor step with PC method
    if step == :aff 
        pad.rhs[1:id.nvar] .= @views dda.Δxy_aff[1:id.nvar] .* pad.D
        pad.rhs[id.nvar+1: end] .= @views dda.Δxy_aff[id.nvar+1: end]
        dda.Δxy_aff, pad.MS.stats = minres!(pad.MS, pad.opK, pad.rhs, M=opDiagonal(pad.invDiagK))
        dda.Δxy_aff[1:id.nvar] .*= pad.D
        LDL = ldl(Symmetric(pad.K, :U))
        println(norm(dda.Δxy_aff - LDL\pad.rhs))
    else
        if pad.K_scaled
            pad.rhs[1:id.nvar] .= @views itd.Δxy[1:id.nvar] .* pad.D
            pad.rhs[id.nvar+1: end] .= @views itd.Δxy[id.nvar+1: end]
            itd.Δxy, pad.MS.stats = minres!(pad.MS, pad.opK, pad.rhs, M=opDiagonal(pad.invDiagK))
            itd.Δxy[1:id.nvar] .*= pad.D
        else
            pad.rhs .= itd.Δxy
            itd.Δxy, pad.MS.stats = minres!(pad.MS, pad.opK, pad.rhs, M=opDiagonal(pad.invDiagK))
        end
    end

    if (step == :cc || step == :IPF) && pad.K_scaled
        pad.D .= one(T) 
        pad.D[id.ilow] ./= sqrt.(itd.x_m_lvar)
        pad.D[id.iupp] ./= sqrt.(itd.uvar_m_x)
        lrmultilply_J!(pad.K.colptr, pad.K.rowval, pad.K.nzval, pad.D, id.nvar)
        pad.K_scaled = false
    end

    return 0
end

function update_pad!(pad :: PreallocatedData_K2_5minres{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
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

    pad.invDiagK[1:id.nvar] .= .-one(T) ./ pad.D 
    pad.invDiagK[id.nvar+1:end] .= pad.regu.δ 

    # scale K
    pad.D .= one(T)
    pad.D[id.ilow] .*= sqrt.(itd.x_m_lvar)
    pad.D[id.iupp] .*= sqrt.(itd.uvar_m_x)
    lrmultilply_J!(pad.K.colptr, pad.K.rowval, pad.K.nzval, pad.D, id.nvar)
    pad.invDiagK[1:id.nvar] ./= pad.D.^2
    pad.K_scaled = true

    pad.opK = PreallocatedLinearOperator(pad.y_op, Symmetric(pad.K, :U))

    return 0
end