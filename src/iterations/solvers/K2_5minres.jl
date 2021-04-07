export K2_5minresParams

struct K2_5minresParams <: SolverParams
    preconditioner  :: Symbol
    atol            :: Float64 
    rtol            :: Float64
    conlim            :: Float64
end

function K2_5minresParams(; preconditioner :: Symbol = :Jacobi, atol :: T = 1.0e-6, rtol :: T = 1.0e-6,
                          conlim :: Float64 = 1.0e-6) where {T<:Real} 
    return K2_5minresParams(preconditioner, atol, rtol, conlim)
end

mutable struct PreallocatedData_K2_5minres{T<:Real} <: PreallocatedData{T} 
    pdat             :: PreconditionerDataK2{T}
    atol             :: T
    rtol             :: T
    conlim           :: T
    P                :: LinearOperator{T} # Preconditioner
    D                :: Vector{T}                                        # temporary top-left diagonal
    rhs              :: Vector{T}
    regu             :: Regularization{T}
    diag_Q           :: SparseVector{T,Int} # Q diagonal
    K                :: SparseMatrixCSC{T,Int} # augmented matrix 
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
        regu = Regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), T(1e-5*sqrt(eps(T))), T(1e0*sqrt(eps(T))), :classic)
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
    
    rhs = zeros(T, id.nvar+id.ncon)
    MS = MinresSolver(K, rhs)
    pdat, P = eval(sp.preconditioner)(id, regu, D, K)

    return PreallocatedData_K2_5minres(pdat,
                                       T(sp.atol),
                                       T(sp.rtol),
                                       T(sp.conlim),
                                       P,
                                       D,
                                       rhs, 
                                       regu,
                                       diag_Q, #diag_Q
                                       K, #K
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
    # LDL = ldl(Symmetric(pad.K, :U))
    if step == :aff 
        pad.rhs[1:id.nvar] .= @views dda.Δxy_aff[1:id.nvar] .* pad.D
        pad.rhs[id.nvar+1: end] .= @views dda.Δxy_aff[id.nvar+1: end]
        rhsNorm = norm(pad.rhs)
        pad.rhs ./= rhsNorm
        (pad.MS.x, pad.MS.stats) = minres!(pad.MS, pad.opK, pad.rhs, M=pad.P, verbose=0, atol=zero(T), rtol=zero(T))
        pad.MS.x .*= rhsNorm
        dda.Δxy_aff .= pad.MS.x
        # ldiv!(dda.Δxy_aff, LDL, pad.rhs)
        # println(norm(Symmetric(pad.K, :U) * dda.Δxy_aff - pad.rhs.*rhsNorm) / rhsNorm)
        # println("norm rhs = ", norm(pad.rhs))
        dda.Δxy_aff[1:id.nvar] .*= pad.D
        
    
    else
        if pad.K_scaled
            pad.rhs[1:id.nvar] .= @views itd.Δxy[1:id.nvar] .* pad.D
            pad.rhs[id.nvar+1: end] .= @views itd.Δxy[id.nvar+1: end]
            rhsNorm = norm(pad.rhs)
            pad.rhs ./= rhsNorm
            (pad.MS.x, pad.MS.stats) = minres!(pad.MS, pad.opK, pad.rhs, M=pad.P, verbose=0, atol=zero(T), rtol=zero(T))
            pad.MS.x .*= rhsNorm
            itd.Δxy .= pad.MS.x
            # println(norm(Symmetric(pad.K, :U) * itd.Δxy - pad.rhs.*rhsNorm) / rhsNorm)
            # println("norm rhs = ", norm(pad.rhs))
            # ldiv!(itd.Δxy, LDL, pad.rhs)
            itd.Δxy[1:id.nvar] .*= pad.D
        else
            pad.rhs .= itd.Δxy
            (pad.MS.x, pad.MS.stats) = minres!(pad.MS, pad.opK, pad.rhs, M=pad.P, atol=zero(T), rtol=zero(T))
            itd.Δxy .= pad.MS.x
            # ldiv!(itd.Δxy, LDL, pad.rhs)
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

    # scale K
    pad.D .= one(T)
    pad.D[id.ilow] .*= sqrt.(itd.x_m_lvar)
    pad.D[id.iupp] .*= sqrt.(itd.uvar_m_x)
    lrmultilply_J!(pad.K.colptr, pad.K.rowval, pad.K.nzval, pad.D, id.nvar) 
    # pad.K.nzval[view(pad.diagind_K,1:id.nvar)] .-= pad.regu.ρ 

    pad.K_scaled = true
    update_preconditioner!(pad.pdat, pad, itd, pt, id, cnts)
    # test = Symmetric(pad.K, :U) * Diagonal(pad.pdat.invDiagK)
    # println(test[diagind(test)])

    pad.opK = PreallocatedLinearOperator(pad.y_op, Symmetric(pad.K, :U))

    return 0
end