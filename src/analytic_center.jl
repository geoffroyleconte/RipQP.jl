function update_analytic_center!(fd::QM_FloatData{T}, id::QM_IntData, pt::Point{T}, itd::IterData{T}, dda::DescentDirectionAllocs{T},
                                 pad::PreallocatedData{T}, res::AbstractResiduals{T}, cnts::Counters) where {T <: Real}
  
  τ = 1.0e-2
  pad.D .= 0
  pad.D[id.ilow] .-= τ ./ itd.x_m_lvar.^2
  pad.D[id.iupp] .-= τ ./ itd.uvar_m_x.^2
  pad.δv[1] = T(1.0e-4)

  # rhs
  itd.Δxy[1: id.nvar] = itd.Qx .+ fd.c .- itd.ATy
  itd.Δxy[id.ilow] .-= τ ./ itd.x_m_lvar
  itd.Δxy[id.iupp] .-= τ ./ itd.uvar_m_x
  itd.Δxy[id.nvar+1: end] = fd.b .- itd.Ax
  @views println("arc = ", norm(itd.Δxy[1:id.nvar]), "   arb = ", norm(itd.Δxy[id.nvar+1: end]))
  out = solver!(pad, dda, pt, itd, fd, id, res, cnts, T, :IPF)

  # update point
  αx = @views compute_α_primal(pt.x, itd.Δxy[1:id.nvar], fd.lvar, fd.uvar)
  pt.x .+= @views αx .* itd.Δxy[1:id.nvar]
  pt.y .+= @views αx .* itd.Δxy[id.nvar+1: end]
  itd.x_m_lvar .= @views pt.x[id.ilow] .- fd.lvar[id.ilow]
  itd.uvar_m_x .= @views fd.uvar[id.iupp] .- pt.x[id.iupp]
  pt.s_l .= τ ./ itd.x_m_lvar
  pt.s_u .= τ ./ itd.uvar_m_x
  
  update_IterData!(itd, pt, fd, id, true)
  res.rb .= itd.Ax .- fd.b
  res.rc .= itd.ATy .- itd.Qx .- fd.c
  res.rc[id.ilow] .+= pt.s_l
  res.rc[id.iupp] .-= pt.s_u
  res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)

  println("αx = ", αx, "   μ = ", itd.μ, "   ||rc|| = ", res.rcNorm, "   ||rb|| = ", res.rbNorm)

end