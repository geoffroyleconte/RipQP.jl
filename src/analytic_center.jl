function update_analytic_center!(fd::QM_FloatData{T}, id::QM_IntData, pt::Point{T}, itd::IterData{T}, dda::DescentDirectionAllocs{T},
                                 pad::PreallocatedData{T}, res::AbstractResiduals{T}, cnts::Counters, 
                                 iconf::InputConfig) where {T <: Real}
  
  for i=1:10
    # only LP for now
    τ = itd.μ
    r, γ = T(0.999), T(0.05)
    # D = [s_l (x-lvar) + s_u (uvar-x)]
    compl = zeros(T, id.nvar)
    compl[id.ilow] .+= pt.s_l .* itd.x_m_lvar
    compl[id.iupp] .+= pt.s_u .* itd.uvar_m_x
    compl[id.ifree] .= one(T)
    ξ = minimum(compl) / itd.μ
    σ = γ * min((one(T) - r) * (one(T) - ξ) / ξ, T(2))^3
    # σ = one(T)

    pad.D .= .-1.0e-8
    pad.D[id.ilow] .-= one(T) ./ itd.x_m_lvar.^2 .+ pt.s_l ./ itd.x_m_lvar
    pad.D[id.iupp] .-= one(T) ./ itd.uvar_m_x.^2 .+ pt.s_u ./ itd.uvar_m_x
    pad.regu.δ = T(1.0e-6)
    pad.δv[1] = pad.regu.δ

    # rhs
    itd.Δxy[1: id.nvar] = .- itd.ATy
    itd.Δxy[id.ilow] .+= (-one(T) - σ * itd.μ) ./ itd.x_m_lvar
    itd.Δxy[id.iupp] .+= (σ * itd.μ + one(T)) ./ itd.uvar_m_x
    itd.Δxy[id.nvar+1: end] = fd.b .- itd.Ax
    res2 = .-itd.ATy
    res2[id.ilow] .+= .-pt.s_l .- one(T) ./ itd.x_m_lvar
    res2[id.iupp] .+= pt.s_u .+ one(T) ./ itd.uvar_m_x
    println("arc = ", norm(res2), "   arb = ", norm(itd.Δxy[id.nvar+1: end]))
    if typeof(pad) <: PreallocatedData_K2LDL
      pad.K.nzval[view(pad.diagind_K, 1:id.nvar)] = pad.D
      pad.K.nzval[view(pad.diagind_K, (id.nvar + 1):(id.ncon + id.nvar))] .= pad.regu.δ
      ldl_factorize!(Symmetric(pad.K, :U), pad.K_fact)
      ldiv!(pad.K_fact, itd.Δxy)
    else
      if typeof(pad) <: PreallocatedData_K2_5Krylov
        pad.sqrtX1X2 .= one(T)
        pad.sqrtX1X2[id.ilow] .*= itd.x_m_lvar
        pad.sqrtX1X2[id.iupp] .*= itd.uvar_m_x
        pad.D .*= pad.sqrtX1X2 .^2
      end
      if cnts.k != 0
        update_regu!(pad.regu)
      end
    
      if pad.atol > pad.atol_min
        pad.atol /= 10
      end
      if pad.rtol > pad.rtol_min
        pad.rtol /= 10
      end  
      out = solver!(itd.Δxy, pad, dda, pt, itd, fd, id, res, cnts, T, :IPF)
    end
    itd.Δs_l .= @views (σ * itd.μ .- pt.s_l .* itd.Δxy[id.ilow]) ./ itd.x_m_lvar .- pt.s_l
    itd.Δs_u .= @views (σ * itd.μ .+ pt.s_u .* itd.Δxy[id.iupp]) ./ itd.uvar_m_x .- pt.s_u

    # update point
    α_pri, α_dual = compute_αs(pt.x, pt.s_l, pt.s_u, fd.lvar, fd.uvar, itd.Δxy, itd.Δs_l, itd.Δs_u, id.nvar)
    pt.x .+= @views α_pri .* itd.Δxy[1:id.nvar]
    pt.y .+= @views α_dual .* itd.Δxy[id.nvar+1: end]
    itd.x_m_lvar .= @views pt.x[id.ilow] .- fd.lvar[id.ilow]
    itd.uvar_m_x .= @views fd.uvar[id.iupp] .- pt.x[id.iupp]
    pt.s_l .= pt.s_l .+ α_dual .* itd.Δs_l
    pt.s_u .= pt.s_u .+ α_dual .* itd.Δs_u
    
    update_IterData!(itd, pt, fd, id, true)
    res.rb .= itd.Ax .- fd.b
    res.rc .= itd.ATy .- itd.Qx .- fd.c
    res.rc[id.ilow] .+= pt.s_l
    res.rc[id.iupp] .-= pt.s_u
    res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)

    println("αx = ", α_pri, "   μ = ", itd.μ, "   ||rc|| = ", res.rcNorm, "   ||rb|| = ", res.rbNorm)
  end

  if iconf.analytic_center == 1
    pt.s_l .= one(T) * 1 ./ itd.x_m_lvar
    pt.s_u .= one(T) * 1 ./ itd.uvar_m_x
  elseif iconf.analytic_center == 2
    pt.s_l .= @views itd.Qx[id.ilow] - itd.ATy[id.ilow] .+ fd.c[id.ilow]
    for i = 1: id.nlow
      if pt.s_l[i] <= zero(T) 
        pt.s_l[i] = T(sqrt(eps(T)))
      end
    end
    pt.s_u .= @views .-itd.Qx[id.iupp] .+ itd.ATy[id.iupp] .- fd.c[id.iupp]
    for i = 1: id.nupp
      if pt.s_u[i] <= zero(T) 
        pt.s_u[i] = T(sqrt(eps(T)))
      end
    end
  end
  update_IterData!(itd, pt, fd, id, true)
  res.rb .= itd.Ax .- fd.b
  res.rc .= itd.ATy .- itd.Qx .- fd.c
  res.rc[id.ilow] .+= pt.s_l
  res.rc[id.iupp] .-= pt.s_u
  res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)

end