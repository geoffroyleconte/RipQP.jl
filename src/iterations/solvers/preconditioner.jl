abstract type PreconditionerDataK2{T<:Real} end

mutable struct JacobiData{T<:Real} <: PreconditionerDataK2{T}
    invDiagK :: Vector{T}
end

function JacobiData(id :: QM_IntData, regu :: Regularization{T}, D :: Vector{T}, K::SparseMatrixCSC{T, Int}) where {T<:Real} 
    invDiagK = (one(T)/regu.δ) .* ones(T, id.nvar+id.ncon)
    invDiagK[1:id.nvar] .= .-one(T) ./ D
    P = opDiagonal(invDiagK)
    return JacobiData{T}(invDiagK), P
end 

function update_preconditioner!(pdat :: JacobiData{T}, pad :: PreallocatedData{T}, itd :: IterData{T}, 
                                pt :: Point{T}, id :: QM_IntData) where {T<:Real}

    # pad.pdat.invDiagK[1:id.nvar] .= .-one(T) ./ pad.D 
    # pad.pdat.invDiagK[id.ilow] ./= itd.x_m_lvar
    # pad.pdat.invDiagK[id.iupp] ./= itd.uvar_m_x
    # pad.pdat.invDiagK[id.nvar+1:end] .= (one(T) / pad.regu.δ )
    pad.pdat.invDiagK .= @views abs.(one(T) ./ pad.K.nzval[pad.diagind_K])
    pad.P = opDiagonal(pad.pdat.invDiagK)
end

mutable struct LLDLData{T<:Real} <: PreconditionerDataK2{T}
    LLDL             :: LimitedLDLFactorization{T,Int}
    y_opiLLDL        :: Vector{T}
end

function LLDLData(id :: QM_IntData, regu :: Regularization{T}, D :: Vector{T}, K :: SparseMatrixCSC{T, Int}) where {T<:Real}
    
    Krows, Kcols, Kvals = findnz(K)
    Kl = sparse(Kcols, Krows, Kvals)
    LLDL = lldl(Kl, memory=20)
    y_opiLLDL = zeros(T, id.nvar+id.ncon)
    P = opilldl(LLDL, y_opiLLDL)
    return LLDLData{T}(LLDL, y_opiLLDL), P
end 

function update_preconditioner!(pdat :: LLDLData{T}, pad :: PreallocatedData{T}, itd :: IterData{T}, 
                                pt :: Point{T}, id :: QM_IntData) where {T<:Real}

    Krows, Kcols, Kvals = findnz(pad.K)
    Kl = sparse(Kcols, Krows, Kvals)
    pad.pdat.LLDL = lldl(Kl, collect(1: id.nvar+id.ncon), memory=20)
    pad.P = opilldl(pad.pdat.LLDL, pad.pdat.y_opiLLDL)
end

mutable struct ActiveCLDLData{T<:Real} <: PreconditionerDataK2{T}
    Kp              :: SparseMatrixCSC{T, Int}
    LDL             :: LDLFactorizations.LDLFactorization{T,Int,Int,Int}
    y_opiLDL        :: Vector{T}
    i_active        :: Vector{Bool}
end

function ActiveCLDLData(id :: QM_IntData, regu :: Regularization{T}, D :: Vector{T}, K :: SparseMatrixCSC{T, Int}) where {T<:Real}
    
    Kp = copy(K)
    LDL = ldl_analyze(Symmetric(K, :U))
    i_active = fill(false, id.nvar)
    ldl_factorize!(Symmetric(K, :U), LDL)
    y_opiLDL = zeros(T, id.nvar + id.ncon)
    LDL.D = abs.(LDL.D)
    P = LinearOperator(T, n, n, true, true, v -> ldiv!(y_opiLDL, LDL, v))

    return ActiveCLDLData{T}(Kp, LDL, y_opiLLDL, i_active), P
end 

function check_active_constr!(i_active, x_m_lvar, uvar_m_x, s_l, s_u, μ, nvar, T)  
    β = T(0.1)
    @inbounds @simd for i=1:nvar
        if min(x_m_lvar, uvar_m_x) ≤ μ^(1-β) 
            i_active = true 
        else
            i_active = false
        end
    end
end

function remove_active_constr!(K_colptr, K_rowval, K_nzval, i_active, nvar, ncon, T)
    
    for j=1:nvar+ncon
        for k=K_colptr[j]: K_colptr[j+1]-1
            i = K_rowval[k]
            if i_active[i]
                K_nzval[k] = zero(T)
            end
        end
    end
end

function update_preconditioner!(pdat :: LLDLData{T}, pad :: PreallocatedData{T}, itd :: IterData{T}, 
                                pt :: Point{T}, id :: QM_IntData) where {T<:Real}

    check_active_constr!(pad.pdat.i_active, itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, itd.μ, id.nvar, T) 


    ldl_factorize!(Symmetric(K, :U), LDL)                            
end