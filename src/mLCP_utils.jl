function starting_points_mLCP(mLCP, itd, Δ_xλ)

    T = eltype(mLCP.q1)
    Δ_xλ[mLCP.n_cols+1: end] .= .-mLCP.q2
    Δ_xλ = ldiv!(itd.J_fact, Δ_xλ)
    pt0 = point(Δ_xλ[1:mLCP.n_cols], Δ_xλ[mLCP.n_cols+1:end], zeros(T, mLCP.n_cols), zeros(T, mLCP.n_cols))
    itd.M11x = mul!(itd.M11x, mLCP.M11, pt0.x)
    itd.M12λ = mul!(itd.M12λ, mLCP.M12, pt0.λ)
    dual_val = itd.M11x .- itd.M12λ .+ mLCP.q1
    pt0.s_l[mLCP.ilow] = @views dual_val[mLCP.ilow]
    pt0.s_u[mLCP.iupp] = @views -dual_val[mLCP.iupp]
    itd.x_m_lvar .= @views pt0.x[mLCP.ilow] .- mLCP.lvar[mLCP.ilow]
    itd.uvar_m_x .= @views mLCP.uvar[mLCP.iupp] .- pt0.x[mLCP.iupp]
    if mLCP.n_low == 0
        δx_l1, δs_l1 = zero(T), zero(T)
    else
        δx_l1 = max(-T(1.5)*minimum(itd.x_m_lvar), T(1.e-2))
        δs_l1 = @views max(-T(1.5)*minimum(pt0.s_l[mLCP.ilow]), T(1.e-4))
    end

    if mLCP.n_upp == 0
        δx_u1, δs_u1 = zero(T), zero(T)
    else
        δx_u1 = max(-T(1.5)*minimum(itd.uvar_m_x), T(1.e-2))
        δs_u1 = @views max(-T(1.5)*minimum(pt0.s_u[mLCP.iupp]), T(1.e-4))
    end

    itd.x_m_lvar .+= δx_l1
    itd.uvar_m_x .+= δx_u1
    s0_l1 = @views pt0.s_l[mLCP.ilow] .+ δs_l1
    s0_u1 = @views pt0.s_u[mLCP.iupp] .+ δs_u1
    xs_l1, xs_u1 = s0_l1' * itd.x_m_lvar, s0_u1' * itd.uvar_m_x
    if mLCP.n_low == 0
        δx_l2, δs_l2 = zero(T), zero(T)
    else
        δx_l2 = δx_l1 + xs_l1 / sum(s0_l1) / 2
        δs_l2 = @views δs_l1 + xs_l1 / sum(itd.x_m_lvar) / 2
    end
    if mLCP.n_upp == 0
        δx_u2, δs_u2 = zero(T), zero(T)
    else
        δx_u2 = δx_u1 + xs_u1 / sum(s0_u1) / 2
        δs_u2 = @views δs_u1 + xs_u1 / sum(itd.uvar_m_x) / 2
    end

    δx = max(δx_l2, δx_u2)
    δs = max(δs_l2, δs_u2)
    pt0.x[mLCP.ilow] .+= δx
    pt0.x[mLCP.iupp] .-= δx
    pt0.s_l[mLCP.ilow] .= @views pt0.s_l[mLCP.ilow] .+ δs
    pt0.s_u[mLCP.iupp] .= @views pt0.s_u[mLCP.iupp] .+ δs

    @inbounds @simd for i in mLCP.irng
        if mLCP.lvar[i] >= pt0.x[i]
            pt0.x[i] = mLCP.lvar[i] + T(1e-4)
        end
        if pt0.x[i] >= mLCP.uvar[i]
            pt0.x[i] = mLCP.uvar[i] - T(1e-4)
        end
        if (mLCP.lvar[i] < pt0.x[i] < mLCP.uvar[i]) == false
            pt0.x[i] = (mLCP.lvar[i] + mLCP.uvar[i]) / 2
        end
    end

    itd.x_m_lvar .= @views pt0.x[mLCP.ilow] .- mLCP.lvar[mLCP.ilow]
    itd.uvar_m_x .= @views mLCP.uvar[mLCP.iupp] .- pt0.x[mLCP.iupp]

    @assert all(pt0.x .> mLCP.lvar) && all(pt0.x .< mLCP.uvar)
    @assert @views all(pt0.s_l[mLCP.ilow] .> zero(T)) && all(pt0.s_u[mLCP.iupp] .> zero(T))

    itd.M11x = mul!(itd.M11x, mLCP.M11, pt0.x)
    itd.M12λ = mul!(itd.M12λ, mLCP.M12, pt0.λ)
    itd.M21x = mul!(itd.M21x, mLCP.M21, pt0.x)
    itd.M22λ = mul!(itd.M22λ, mLCP.M22, pt0.λ)
    itd.μ = @views compute_μ(itd.x_m_lvar, itd.uvar_m_x, pt0.s_l[mLCP.ilow], pt0.s_u[mLCP.iupp],
                             mLCP.n_low, mLCP.n_upp)

    return pt0, itd, Δ_xλ
end

function init_params_mLCP(mLCP :: mLCPModel{T}, ϵ :: tolerances_mLCP{T}) where {T<:Real}
    res = residuals_mLCP(zeros(T, mLCP.n_cols), zeros(T, mLCP.n_rows), zero(T), zero(T), zero(T))
    # init regularization values
    regu = regularization(T(sqrt(eps())*1e5), T(sqrt(eps())*1e5), 1e-5*sqrt(eps(T)), 1e0*sqrt(eps(T)), :classic)
    tmp_diag = ones(T, mLCP.n_rows+mLCP.n_cols)
    tmp_diag[1:mLCP.n_cols] .= .-regu.ρ
    tmp_diag[mLCP.n_cols+1:end] .= regu.δ
    J_augm = [-mLCP.M11     mLCP.M12
               mLCP.M21     mLCP.M22]
    diagind_J = diagind(J_augm)

    J_fact = lu(J_augm)
    J_P = J_fact.p
    itd = iter_data_mLCP(tmp_diag, # tmp diag
                         vcat(.-mLCP.M11[diagind(mLCP.M11)], mLCP.M22[diagind(mLCP.M22)]), #diag_M11&M22
                         J_augm, #J_augm
                         J_fact,
                         J_P,
                         diagind(J_augm), #diagind_J
                         zeros(T, mLCP.n_low), # x_m_lvar
                         zeros(T, mLCP.n_upp), # uvar_m_x
                         zeros(T, mLCP.n_cols), # init M11x
                         zeros(T, mLCP.n_cols), # init M12λ
                         zeros(T, mLCP.n_rows), # M21x
                         zeros(T, mLCP.n_rows), # M22λ
                         zero(T), #μ
                    )

    pad = preallocated_data(zeros(T, mLCP.n_cols+mLCP.n_rows+mLCP.n_low+mLCP.n_upp), # Δ_aff
                            zeros(T, mLCP.n_cols+mLCP.n_rows+mLCP.n_low+mLCP.n_upp), # Δ_cc
                            zeros(T, mLCP.n_cols+mLCP.n_rows+mLCP.n_low+mLCP.n_upp), # Δ
                            zeros(T, mLCP.n_cols+mLCP.n_rows), # Δ_xλ
                            zeros(T, mLCP.n_low), # x_m_l_αΔ_aff
                            zeros(T, mLCP.n_upp), # u_m_x_αΔ_aff
                            zeros(T, mLCP.n_low), # s_l_αΔ_aff
                            zeros(T, mLCP.n_upp), # s_u_αΔ_aff
                            zeros(T, mLCP.n_low), # rxs_l
                            zeros(T, mLCP.n_upp) #rxs_u
                            )

    pt, itd, pad.Δ_xλ = @views starting_points_mLCP(mLCP, itd, pad.Δ_xλ)

    res.r1 .= .-itd.M11x .+ itd.M12λ .+ pt.s_l .- pt.s_u .- mLCP.q1
    res.r2 .= itd.M21x .+ mLCP.q2 .- itd.M22λ
    res.r1Norm, res.r2Norm = norm(res.r1, Inf), norm(res.r2, Inf)
    ϵ.tol_r1, ϵ.tol_r2 = ϵ.r1*(one(T) + res.r1Norm), ϵ.r2*(one(T) + res.r2Norm)
    sc = stop_crit(res.r1Norm < ϵ.tol_r1 && res.r2Norm < ϵ.tol_r2, # optimal
                   false, # small_Δx
                   itd.μ < ϵ.μ, # small_μ
                   false # tired
                   )
   return regu, itd, ϵ, pad, pt, res, sc
end

function iter_mLCP!(pt :: point{T}, itd :: iter_data_mLCP{T}, mLCP :: mLCPModel{T},
                    res :: residuals_mLCP{T}, sc :: stop_crit, Δt :: Real, regu :: regularization{T},
                    pad :: preallocated_data{T}, max_iter :: Int, ϵ :: tolerances_mLCP{T}, start_time :: Real,
                    max_time :: Real, cnts :: counters, T0 :: DataType, display :: Bool) where {T<:Real}

    @inbounds while cnts.k<max_iter && !sc.optimal && !sc.tired # && !small_μ && !small_μ

        ###################### J update and factorization ######################
        itd.tmp_diag[1:mLCP.n_cols] .= -regu.ρ
        itd.tmp_diag[mLCP.n_cols+1: end] .= regu.δ
        itd.tmp_diag[mLCP.ilow] .-= @views pt.s_l[mLCP.ilow] ./ itd.x_m_lvar
        itd.tmp_diag[mLCP.iupp] .-= @views pt.s_u[mLCP.iupp] ./ itd.uvar_m_x
        itd.J_augm[itd.diagind_J] .= itd.diag_M .+ itd.tmp_diag
        itd.J_fact = try lu(itd.J_augm)#, itd.J_P)
        catch
            if T == Float32
                break
                break
            elseif T0 == Float128 && T == Float64
                break
                break
            end
            if cnts.c_pdd == 0 && cnts.c_catch == 0
                regu.δ *= T(1e2)
                regu.δ_min *= T(1e2)
                regu.ρ *= T(1e5)
                regu.ρ_min *= T(1e5)
            elseif cnts.c_pdd == 0 && cnts.c_catch != 0
                regu.δ *= T(1e1)
                regu.δ_min *= T(1e1)
                regu.ρ *= T(1e0)
                regu.ρ_min *= T(1e0)
            elseif cnts.c_pdd != 0 && cnts.c_catch==0
                regu.δ *= T(1e5)
                regu.δ_min *= T(1e5)
                regu.ρ *= T(1e5)
                regu.ρ_min *= T(1e5)
            else
                regu.δ *= T(1e1)
                regu.δ_min *= T(1e1)
                regu.ρ *= T(1e1)
                regu.ρ_min *= T(1e1)
            end
            cnts.c_catch += 1
            itd.tmp_diag[1:mLCP.n_cols] .= -regu.ρ
            itd.tmp_diag[mLCP.n_cols+1: end] .= regu.δ
            itd.tmp_diag[mLCP.ilow] .-= @views pt.s_l[mLCP.ilow] ./ itd.x_m_lvar
            itd.tmp_diag[mLCP.iupp] .-= @views pt.s_u[mLCP.iupp] ./ itd.uvar_m_x
            itd.J_augm[itd.diagind_J] .= itd.diag_M .+ itd.tmp_diag
            itd.J_fact = lu(itd.J_augm)#, itd.J_P)
        end

        if cnts.c_catch >= 4
            break
        end
        ########################################################################

        pad.Δ_aff = solve_augmented_system_aff!(itd.J_fact, pad.Δ_aff, pad.Δ_xλ, res.r1, res.r2,
                                                itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u,
                                                mLCP.ilow, mLCP.iupp, mLCP.n_cols, mLCP.n_rows,
                                                mLCP.n_low)
        α_aff_pri = @views compute_α_primal(pt.x, pad.Δ_aff[1:mLCP.n_cols], mLCP.lvar, mLCP.uvar)
        α_aff_dual_l = @views compute_α_dual(pt.s_l[mLCP.ilow],
                                             pad.Δ_aff[mLCP.n_rows+mLCP.n_cols+1:mLCP.n_rows+mLCP.n_cols+mLCP.n_low])
        α_aff_dual_u = @views compute_α_dual(pt.s_u[mLCP.iupp],
                                             pad.Δ_aff[mLCP.n_rows+mLCP.n_cols+mLCP.n_low+1:end])
        # alpha_aff_dual is the min of the 2 alpha_aff_dual
        α_aff_dual = min(α_aff_dual_l, α_aff_dual_u)
        pad.x_m_l_αΔ_aff .= @views itd.x_m_lvar .+ α_aff_pri .* pad.Δ_aff[1:mLCP.n_cols][mLCP.ilow]
        pad.u_m_x_αΔ_aff .= @views itd.uvar_m_x .- α_aff_pri .* pad.Δ_aff[1:mLCP.n_cols][mLCP.iupp]
        pad.s_l_αΔ_aff .= @views pt.s_l[mLCP.ilow] .+ α_aff_dual .*
                            pad.Δ_aff[mLCP.n_rows+mLCP.n_cols+1: mLCP.n_rows+mLCP.n_cols+mLCP.n_low]
        pad.s_u_αΔ_aff .= @views pt.s_u[mLCP.iupp] .+ α_aff_dual .*
                            pad.Δ_aff[mLCP.n_rows+mLCP.n_cols+mLCP.n_low+1: end]
        μ_aff = compute_μ(pad.x_m_l_αΔ_aff, pad.u_m_x_αΔ_aff, pad.s_l_αΔ_aff, pad.s_u_αΔ_aff,
                          mLCP.n_low, mLCP.n_upp)
        σ = (μ_aff / itd.μ)^3

        # corrector and centering step
        pad.Δ_cc = solve_augmented_system_cc!(itd.J_fact, pad.Δ_cc, pad.Δ_xλ , pad.Δ_aff, σ, itd.μ,
                                              itd.x_m_lvar, itd.uvar_m_x, pad.rxs_l, pad.rxs_u, pt.s_l, pt.s_u,
                                              mLCP.ilow, mLCP.iupp, mLCP.n_cols, mLCP.n_rows,
                                              mLCP.n_low)
        pad.Δ .= pad.Δ_aff .+ pad.Δ_cc # final direction
        α_pri = @views compute_α_primal(pt.x, pad.Δ[1:mLCP.n_cols], mLCP.lvar, mLCP.uvar)
        α_dual_l = @views compute_α_dual(pt.s_l[mLCP.ilow],
                                         pad.Δ[mLCP.n_rows+mLCP.n_cols+1:mLCP.n_rows+mLCP.n_cols+mLCP.n_low])
        α_dual_u = @views compute_α_dual(pt.s_u[mLCP.iupp], pad.Δ[mLCP.n_rows+mLCP.n_cols+mLCP.n_low+1: end])
        α_dual = min(α_dual_l, α_dual_u)
        # new parameters
        pt.x .= @views pt.x .+ α_pri .* pad.Δ[1:mLCP.n_cols]
        pt.λ .= @views pt.λ .+ α_dual .* pad.Δ[mLCP.n_cols+1: mLCP.n_rows+mLCP.n_cols]
        pt.s_l[mLCP.ilow] .= @views pt.s_l[mLCP.ilow] .+ α_dual .*
                                  pad.Δ[mLCP.n_rows+mLCP.n_cols+1: mLCP.n_rows+mLCP.n_cols+mLCP.n_low]
        pt.s_u[mLCP.iupp] .= @views pt.s_u[mLCP.iupp] .+ α_dual .*
                                  pad.Δ[mLCP.n_rows+mLCP.n_cols+mLCP.n_low+1: end]
        res.n_Δx = @views α_pri * norm(pad.Δ[1:mLCP.n_cols])
        itd.x_m_lvar .= @views pt.x[mLCP.ilow] .- mLCP.lvar[mLCP.ilow]
        itd.uvar_m_x .= @views mLCP.uvar[mLCP.iupp] .- pt.x[mLCP.iupp]

        if zero(T) in itd.x_m_lvar # "security" if x is too close from lvar ou uvar
            @inbounds @simd for i=1:mLCP.n_low
                if itd.x_m_lvar[i] == zero(T)
                    itd.x_m_lvar[i] = eps(T)^2
                end
            end
        end
        if zero(T) in itd.uvar_m_x
            @inbounds @simd for i=1:mLCP.n_upp
                if itd.uvar_m_x[i] == zero(T)
                    itd.uvar_m_x[i] = eps(T)^2
                end
            end
        end

        itd.μ = @views compute_μ(itd.x_m_lvar, itd.uvar_m_x, pt.s_l[mLCP.ilow], pt.s_u[mLCP.iupp],
                                 mLCP.n_low, mLCP.n_upp)

        itd.M11x = mul!(itd.M11x, mLCP.M11, pt.x)
        itd.M12λ = mul!(itd.M12λ, mLCP.M12, pt.λ)
        itd.M21x = mul!(itd.M21x, mLCP.M21, pt.x)
        itd.M22λ = mul!(itd.M22λ, mLCP.M22, pt.λ)
        res.r1 .= .-itd.M11x .+ itd.M12λ .+ pt.s_l .- pt.s_u .- mLCP.q1
        res.r2 .= itd.M21x .+ mLCP.q2 .- itd.M22λ

        # update stopping criterion values:
#         rcNorm, rbNorm = norm(rc), norm(rb)
#         xNorm = norm(x)
#         λNorm = norm(λ)
#         optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb * max(1, bNorm + ANorm * xNorm) &&
#                     rcNorm < ϵ_rc * max(1, cNorm + QNorm * xNorm + ANorm * λNorm)
        res.r1Norm, res.r2Norm = norm(res.r1, Inf), norm(res.r2, Inf)
        sc.optimal =  res.r1Norm < ϵ.tol_r1 && res.r2Norm < ϵ.tol_r2
        sc.small_Δx, sc.small_μ = res.n_Δx < ϵ.Δx, itd.μ < ϵ.μ

        cnts.k += 1
        if T == Float32
            cnts.km += 1
        elseif T == Float64
            cnts.km += 4
        else
            cnts.km += 16
        end

        if regu.δ >= regu.δ_min
            regu.δ /= 10
        end
        if regu.ρ >= regu.ρ_min
            regu.ρ /= 10
        end

        Δt = time() - start_time
        sc.tired = Δt > max_time

        if display == true
            @info log_row(Any[cnts.k, res.r1Norm, res.r2Norm, res.n_Δx, α_pri, α_dual, itd.μ, regu.ρ, regu.δ])
        end
    end

    return pt, res, itd, Δt, sc, cnts, regu
end
