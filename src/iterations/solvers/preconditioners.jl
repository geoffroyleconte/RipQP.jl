abstract type PreconditionerDataK2{T<:Real} end

mutable struct JacobiData{T<:Real} <: PreconditionerDataK2{T}
    invDiagK :: Vector{T}
end

function Jacobi(id :: QM_IntData, regu :: Regularization{T}, D :: Vector{T}, K::SparseMatrixCSC{T, Int}) where {T<:Real} 
    invDiagK = (one(T)/regu.δ) .* ones(T, id.nvar+id.ncon)
    invDiagK[1:id.nvar] .= .-one(T) ./ D
    P = opDiagonal(invDiagK)
    return JacobiData{T}(invDiagK), P
end 

function update_preconditioner!(pdat :: JacobiData{T}, pad :: PreallocatedData{T}, itd :: IterData{T}, 
                                pt :: Point{T}, id :: QM_IntData, cnts :: Counters) where {T<:Real}

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

function LLDL(id :: QM_IntData, regu :: Regularization{T}, D :: Vector{T}, K :: SparseMatrixCSC{T, Int}) where {T<:Real}
    
    Krows, Kcols, Kvals = findnz(K)
    Kl = sparse(Kcols, Krows, Kvals)
    LLDL = lldl(Kl, memory=40)
    y_opiLLDL = zeros(T, id.nvar+id.ncon)
    P = opilldl(LLDL, y_opiLLDL)
    return LLDLData{T}(LLDL, y_opiLLDL), P
end 

function update_preconditioner!(pdat :: LLDLData{T}, pad :: PreallocatedData{T}, itd :: IterData{T}, 
                                pt :: Point{T}, id :: QM_IntData, cnts :: Counters) where {T<:Real}

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

function ActiveCLDL(id :: QM_IntData, regu :: Regularization{T}, D :: Vector{T}, K :: SparseMatrixCSC{T, Int}) where {T<:Real}
    
    Kp = copy(K)
    LDL = ldl_analyze(Symmetric(Kp, :U))
    i_active = fill(false, id.nvar)
    ldl_factorize!(Symmetric(Kp, :U), LDL)
    y_opiLDL = zeros(T, id.nvar + id.ncon)
    LDL.d = abs.(LDL.d)
    P = LinearOperator(T, id.nvar + id.ncon, id.nvar + id.ncon, true, true, v -> ldiv!(y_opiLDL, LDL, v))

    return ActiveCLDLData{T}(Kp, LDL, y_opiLDL, i_active), P
end 

function check_active_constr!(i_active, x_m_lvar, uvar_m_x, μ, ilow, iupp, nlow, nupp, nvar, T)  
    β = T(0.01)
    tol_active = min(μ^(1-β), one(T)) 
    c_low, c_upp = 1, 1
    for i=1:nvar
        if c_low < nlow && ilow[c_low] < i
            c_low += 1
        end
        if c_upp < nupp && iupp[c_upp] < i
            c_upp += 1
        end
        if c_low ≤ nlow && ilow[c_low] == i && c_upp ≤ nupp && iupp[c_upp] == i && min(x_m_lvar[c_low], uvar_m_x[c_upp]) ≤ tol_active
            i_active[i] = true 
        elseif c_low ≤ nlow && ilow[c_low] == i && x_m_lvar[c_low] ≤ tol_active
            i_active[i] = true 
        elseif c_upp ≤ nupp && iupp[c_upp] == i && uvar_m_x[c_upp] ≤ tol_active
            i_active[i] = true
        else 
            i_active[i] = false
        end
    end
    # println(sum(i_active))
end

function remove_active_constr!(K_colptr, K_rowval, K_nzval, x_m_lvar, uvar_m_x, s_l, s_u, ρ, i_active, ilow, iupp, nlow, nupp, 
                               nvar, ncon, T)

    c_low, c_upp = 1, 1
    for j=1:nvar+ncon
        if c_low < nlow && ilow[c_low] < j
            c_low += 1
        end
        if c_upp < nupp && iupp[c_upp] < j
            c_upp += 1
        end
        for k=K_colptr[j]: K_colptr[j+1]-1
            i = K_rowval[k]
            if i ≤ nvar
                if i != j && i_active[i]
                    K_nzval[k] = zero(T)
                elseif i != j && j ≤ nvar && i_active[j]
                    K_nzval[k] = zero(T)
                elseif i == j && i_active[i]
                    if c_low ≤ nlow && ilow[c_low] == i && c_upp ≤ nupp && iupp[c_upp] == i
                        K_nzval[k] = -s_l[c_low] * uvar_m_x[c_upp] - s_u[c_upp] * x_m_lvar[c_low] - ρ 
                    elseif c_low ≤ nlow && ilow[c_low] == i
                        K_nzval[k] = -s_l[c_low] - ρ
                    elseif c_upp ≤ nupp && iupp[c_upp] == i
                        K_nzval[k] = -s_u[c_upp] - ρ
                    end
                end
            end
        end
    end
end

# function remove_active_constr!(D, i_active, nvar)
#     for i=1:nvar 
#         if i_active == true
#             D[i] = 0
#         end
#     end
# end 

function update_preconditioner!(pdat :: ActiveCLDLData{T}, pad :: PreallocatedData{T}, itd :: IterData{T}, 
                                pt :: Point{T}, id :: QM_IntData, cnts :: Counters) where {T<:Real}

    pad.pdat.Kp.nzval .= pad.K.nzval
    check_active_constr!(pad.pdat.i_active, itd.x_m_lvar, itd.uvar_m_x, itd.μ, id.ilow, id.iupp, id.nlow, id.nupp, id.nvar, T) 
    # remove_active_constr!(D, i_active, nvar)
    remove_active_constr!(pad.pdat.Kp.colptr, pad.pdat.Kp.rowval, pad.pdat.Kp.nzval, itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, 
                          pad.regu.ρ, pad.pdat.i_active, id.ilow, id.iupp, id.nlow, id.nupp, id.nvar, id.ncon, T)

    # Amax = @views norm(pad.pdat.Kp.nzval[pad.diagind_K], Inf)
    # pad.regu.ρ, pad.regu.δ = T(eps(T)^(3/4)), T(eps(T)^(0.45))
    # pad.pdat.LDL.r1, pad.pdat.LDL.r2 = -pad.regu.ρ, pad.regu.δ
    # pad.pdat.LDL.tol = Amax*T(eps(T))
    # pad.pdat.LDL.n_d = id.nvar

    ldl_factorize!(Symmetric(pad.pdat.Kp, :U), pad.pdat.LDL) 
    while !factorized(pad.pdat.LDL)
        println("error fact")
        out = update_regu_trycatch!(pad.regu, cnts, T, T)
        out == 1 && return out
        cnts.c_catch += 1
        cnts.c_catch >= 4 && return 1
        # pad.D .= -pad.regu.ρ
        # D[ilow] .-= s_l ./ x_m_lvar
        # D[iupp] .-= s_u ./ uvar_m_x
        # D[diag_Q.nzind] .-= diag_Q.nzval
        pad.pdat.Kp.nzval[view(pad.diagind_K,1: id.nvar)] .-= pad.regu.ρ
        pad.pdat.Kp.nzval[view(pad.diagind_K, id.nvar+1: id.ncon+id.nvar)] .= pad.regu.δ
        pad.K.nzval[view(pad.diagind_K,1: id.nvar)] .-= pad.regu.ρ
        pad.K.nzval[view(pad.diagind_K, id.nvar+1: id.ncon+id.nvar)] .= pad.regu.δ
        # println(norm(pad.K.nzval[pad.diagind_K] - pad.pdat.Kp.nzval[pad.diagind_K]))
        ldl_factorize!(Symmetric(pad.pdat.Kp, :U), pad.pdat.LDL)
    end
    # println("norm diff prec", norm(pdat.Kp - pad.K))
    pad.pdat.LDL.d .= abs.(pad.pdat.LDL.d)
    # display(Matrix(pad.pdat.Kp))
    pad.P = LinearOperator(T, id.ncon+id.nvar, id.ncon+id.nvar, true, true, v -> ldiv!(pad.pdat.y_opiLDL, pad.pdat.LDL, v)) 
    Minv = invop(pad.P, T, id.ncon+id.nvar)
    # eigs = real.(eigvals(Matrix(Minv * Symmetric(pad.K,:U))))
    # println(unique(trunc.(eigs, digits = 2)))  
    # println("||K|| ", norm(Symmetric(pad.K, :U)))                   
end

function invop(M, T, n)
    ei = zeros(T, n)
    Minv = zeros(T, n, n)
    for i=1:n
        ei[i] = one(T)
        Minv[:, i] = M * ei 
        ei[i] = zero(T)
    end
    return Minv 
end