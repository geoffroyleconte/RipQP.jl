# Formulation K2: (if regul==:classic, adds additional Regularization parmeters -ρ (top left) and δ (bottom right))
# [-Q_X - D      sqrt(X1X2)A' ] [̃x] = rhs
# [ A sqrt(X1x2)         0    ] [y]
# where Q_X = sqrt(X1X2) Q sqrt(X1X2) and D = s_l X2 + s_u X1
# and  Δ x = sqrt(X1 X2) Δ ̃x
export K2_5hybridParams

"""
    K2_5hybridParams
"""
struct K2_5hybridParams <: SolverParams
    regul           :: Symbol
    preconditioner  :: Symbol
    ratol           :: Float64 
    rrtol           :: Float64
end

function K2_5hybridParams(; regul :: Symbol = :classic, preconditioner :: Symbol = :Jacobi, 
                          ratol :: T = sqrt(eps())*1.0e2, rrtol :: T = sqrt(eps())) where {T<:Real}

    regul == :classic || regul == :dynamic || regul == :none || error("regul should be :classic or :dynamic or :none")
    return K2_5hybridParams(regul, preconditioner, ratol, rrtol)
end
 
mutable struct PreallocatedData_K2_5hybrid{T<:Real} <: PreallocatedData{T} 
    D                :: Vector{T}                                        # temporary top-left diagonal
    regu             :: Regularization{T}
    diag_Q           :: SparseVector{T,Int} # Q diagonal
    K                :: SparseMatrixCSC{T,Int} # augmented matrix 
    K_fact           :: LDLFactorizations.LDLFactorization{T,Int,Int,Int} # factorized matrix
    fact_fail        :: Bool # true if factorization failed 
    diagind_K        :: Vector{Int} # diagonal indices of J
    K_scaled         :: Bool # true if K is scaled with X1X2
    # Krylov Solver data
    pdat             :: PreconditionerDataK2{T}
    ratol            :: T
    rrtol            :: T
    P                :: LinearOperator{T} # Preconditioner
    rhs              :: Vector{T}
    opK              :: PreallocatedLinearOperator{T}
    y_op             :: Vector{T} # preallocated vector for the LinearOperator
    MS               :: MinresSolver{T, Vector{T}}
    ac_rm            :: Bool # indicates if active constraints are removed (if false then LDLFactorization is used, if true Krylov solver)
end

function PreallocatedData(sp :: K2_5hybridParams, fd :: QM_FloatData{T}, id :: QM_IntData, 
                          iconf :: InputConfig{Tconf}) where {T<:Real, Tconf<:Real} 
    # init Regularization values
    if iconf.mode == :mono
        regu = Regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), 1e-5*sqrt(eps(T)), 1e0*sqrt(eps(T)), sp.regul)
        D = -T(1.0e0)/2 .* ones(T, id.nvar)
    else
        regu = Regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), T(sqrt(eps(T))*1e0), T(sqrt(eps(T))*1e0), sp.regul)
        D = -T(1.0e-2) .* ones(T, id.nvar)
    end
    diag_Q = get_diag_Q(fd.Q.colptr, fd.Q.rowval, fd.Q.nzval, id.nvar)
    K = create_K2(id, D, fd.Q, fd.AT, diag_Q, regu)

    diagind_K = get_diag_sparseCSC(K.colptr, id.ncon+id.nvar)
    K_fact = ldl_analyze(Symmetric(K, :U))
    if regu.regul == :dynamic
        Amax = @views norm(K.nzval[diagind_K], Inf)
        regu.ρ, regu.δ = -T(eps(T)^(3/4)), T(eps(T)^(0.45))
        K_fact.r1, K_fact.r2 = regu.ρ, regu.δ
        K_fact.tol = Amax*T(eps(T))
        K_fact.n_d = id.nvar
    elseif regu.regul == :none
        regu.ρ, regu.δ = zero(T), zero(T)
    end
    K_fact = ldl_factorize!(Symmetric(K, :U), K_fact)
    K_fact.__factorized = true

    # Krylov solver 
    y_op = zeros(T, id.nvar+id.ncon)
    opK = PreallocatedLinearOperator(y_op, Symmetric(K, :U))
    
    rhs = zeros(T, id.nvar+id.ncon)
    MS = MinresSolver(K, rhs)
    pdat, P = eval(sp.preconditioner)(id, regu, D, K)

    return PreallocatedData_K2_5hybrid(D,
                                       regu,
                                       diag_Q, #diag_Q
                                       K, #K
                                       K_fact, #K_fact
                                       false,
                                       diagind_K, #diagind_K
                                       false,
                                       pdat,
                                       T(sp.ratol),
                                       T(sp.rrtol),
                                       P,
                                       rhs,
                                       opK, 
                                       y_op,
                                       MS,
                                       false
                                       )
end

# solver LDLFactorization
function solver!(pad :: PreallocatedData_K2_5hybrid{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
                 fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, cnts :: Counters, T0 :: DataType, 
                 step :: Symbol) where {T<:Real} 

    if !pad.ac_rm # LDL factorization
        if step == :aff 
            dda.Δxy_aff[1:id.nvar] .*= pad.D
            LDLFactorizations.ldiv!(pad.K_fact, dda.Δxy_aff) 
            dda.Δxy_aff[1:id.nvar] .*= pad.D
        else
            if pad.K_scaled
                itd.Δxy[1:id.nvar] .*= pad.D
                LDLFactorizations.ldiv!(pad.K_fact, itd.Δxy)
                itd.Δxy[1:id.nvar] .*= pad.D
            else
                LDLFactorizations.ldiv!(pad.K_fact, itd.Δxy)
            end
        end

    else # Krylov Solver
        if step == :aff 
            pad.rhs[1:id.nvar] .= @views dda.Δxy_aff[1:id.nvar] .* pad.D
            pad.rhs[id.nvar+1: end] .= @views dda.Δxy_aff[id.nvar+1: end]
            rhsNorm = norm(pad.rhs)
            if rhsNorm != zero(T)
                pad.rhs ./= rhsNorm
            end
            (pad.MS.x, pad.MS.stats) = minres!(pad.MS, pad.opK, pad.rhs, M=pad.P, verbose=0, atol=zero(T), rtol=zero(T), 
                                            ratol=pad.ratol, rrtol=pad.rrtol)
            if rhsNorm != zero(T)
                pad.MS.x .*= rhsNorm
            end
            dda.Δxy_aff .= pad.MS.x
            dda.Δxy_aff[1:id.nvar] .*= pad.D
            
        
        else
            if pad.K_scaled
                pad.rhs[1:id.nvar] .= @views itd.Δxy[1:id.nvar] .* pad.D
                pad.rhs[id.nvar+1: end] .= @views itd.Δxy[id.nvar+1: end]
                rhsNorm = norm(pad.rhs)
                if rhsNorm != zero(T)
                    pad.rhs ./= rhsNorm
                end
                (pad.MS.x, pad.MS.stats) = minres!(pad.MS, pad.opK, pad.rhs, M=pad.P, verbose=0, atol=zero(T), rtol=zero(T), 
                                                ratol=pad.ratol, rrtol=pad.rrtol)
                if rhsNorm != zero(T)
                    pad.MS.x .*= rhsNorm
                end
                itd.Δxy .= pad.MS.x
                itd.Δxy[1:id.nvar] .*= pad.D
            else
                pad.rhs .= itd.Δxy
                rhsNorm = norm(pad.rhs)
                if rhsNorm != zero(T)
                    pad.rhs ./= rhsNorm
                end
                (pad.MS.x, pad.MS.stats) = minres!(pad.MS, pad.opK, pad.rhs, M=pad.P, atol=zero(T), rtol=zero(T),
                                                ratol=pad.ratol, rrtol=pad.rrtol)
                if rhsNorm != zero(T)
                    pad.MS.x .*= rhsNorm
                end
                itd.Δxy .= pad.MS.x
            end
        end
    end

    if (step == :cc || step == :IPF) && pad.K_scaled # update regulariation, restore J

        out = 0
        if pad.regu.regul == :classic # update ρ and δ values, check K diag magnitude 
            out = update_regu_diagK2_5!(pad.regu, pad.D, itd.pdd, itd.l_pdd, itd.mean_pdd, cnts, T, T0) 
            # update_regu!(pad.regu)
        end

        pad.D .= one(T) 
        pad.D[id.ilow] ./= sqrt.(itd.x_m_lvar)
        pad.D[id.iupp] ./= sqrt.(itd.uvar_m_x)
        lrmultilply_J!(pad.K.colptr, pad.K.rowval, pad.K.nzval, pad.D, id.nvar)
        pad.K_scaled = false
    end

    return 0
end

function update_pad!(pad :: PreallocatedData_K2_5hybrid{T}, dda :: DescentDirectionAllocs{T}, pt :: Point{T}, itd :: IterData{T}, 
                     fd :: Abstract_QM_FloatData{T}, id :: QM_IntData, res :: Residuals{T}, cnts :: Counters, T0 :: DataType) where {T<:Real}

    out = 0
    if !pad.ac_rm && itd.μ > one(T)/10 # LDL Fact
        pad.K_scaled = false
        out = factorize_K2_5!(pad.K, pad.K_fact, pad.D, pad.diag_Q, pad.diagind_K, pad.regu, 
                            pt.s_l, pt.s_u, itd.x_m_lvar, itd.uvar_m_x, id.ilow, id.iupp, 
                            id.ncon, id.nvar, cnts, itd.qp, T, T0)
        out == 1 && return out
        pad.K_scaled = true

    else
        if !pad.ac_rm # increase reglarizations for the transition
            pad.regu.ρ, pad. regu.ρ_min = T(1e2*sqrt(eps(T))), T(1e3*sqrt(eps(T)))
            pad.regu.δ, pad.regu.δ_min = T(1e3*sqrt(eps(T))), T(1e4*sqrt(eps(T)))
        end
        if pad.regu.regul == :classic
            pad.D .= -pad.regu.ρ 
            pad.K.nzval[view(pad.diagind_K, id.nvar+1:id.ncon+id.nvar)] .= pad.regu.δ
        else
            pad.D .= zero(T)
        end 
        pad.D[id.ilow] .-= pt.s_l ./ itd.x_m_lvar
        pad.D[id.iupp] .-= pt.s_u ./ itd.uvar_m_x
        pad.D[pad.diag_Q.nzind] .-= pad.diag_Q.nzval
        pad.K.nzval[view(pad.diagind_K,1:id.nvar)] = pad.D
    
        # scale K
        pad.D .= one(T)
        pad.D[id.ilow] .*= sqrt.(itd.x_m_lvar)
        pad.D[id.iupp] .*= sqrt.(itd.uvar_m_x)
        lrmultilply_J!(pad.K.colptr, pad.K.rowval, pad.K.nzval, pad.D, id.nvar)  
    
        pad.K_scaled = true
        update_preconditioner!(pad.pdat, pad, itd, pt, id, cnts)
    
        pad.opK = PreallocatedLinearOperator(pad.y_op, Symmetric(pad.K, :U))
    end

    return out
end

