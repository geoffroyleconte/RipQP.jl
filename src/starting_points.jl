function starting_points!(
  pt0::Point{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  itd::IterData{T},
  spd::StartingPointData{T},
  warm_start::Bool,
) where {T <: Real}
  mul!(itd.Qx, fd.Q, pt0.x)
  fd.uplo == :U ? mul!(itd.ATy, fd.A, pt0.y) : mul!(itd.ATy, fd.A', pt0.y)
  @. spd.dual_val = itd.Qx - itd.ATy + fd.c
  pt0.s_l = spd.dual_val[id.ilow]
  pt0.s_u = -spd.dual_val[id.iupp]

  # check distance to bounds δ for x, s_l and s_u
  @. itd.x_m_lvar = @views pt0.x[id.ilow] - fd.lvar[id.ilow]
  @. itd.uvar_m_x = @views fd.uvar[id.iupp] - pt0.x[id.iupp]
  warm_start && @assert all(itd.x_m_lvar .> zero(T)) && all(itd.uvar_m_x .> zero(T))

  # safety x
  if !warm_start
    δx_l1 = (id.nlow == 0) ? zero(T) : max(-T(1.5) * minimum(itd.x_m_lvar), T(1.e-2))
    δx_u1 = (id.nupp == 0) ? zero(T) : max(-T(1.5) * minimum(itd.uvar_m_x), T(1.e-2))
    itd.x_m_lvar .+= δx_l1
    itd.uvar_m_x .+= δx_u1
  end

  # safety s
  δs_l1 = (id.nlow == 0) ? zero(T) : max(-T(1.5) * minimum(pt0.s_l), T(1.e-4))
  δs_u1 = (id.nupp == 0) ? zero(T) : max(-T(1.5) * minimum(pt0.s_u), T(1.e-4))
  @. spd.s0_l1 = pt0.s_l + δs_l1
  @. spd.s0_u1 = pt0.s_u + δs_u1

  # correct components that to not respect the bounds
  xs_l1, xs_u1 = dot(spd.s0_l1, itd.x_m_lvar), dot(spd.s0_u1, itd.uvar_m_x)

  # 2nd safety x
  if !warm_start
    δx_l2 = (id.nlow == 0) ? zero(T) : δx_l1 + xs_l1 / sum(spd.s0_l1) / 2
    δx_u2 = (id.nupp == 0) ? zero(T) : δx_u1 + xs_u1 / sum(spd.s0_u1) / 2
    δx = max(δx_l2, δx_u2)
    pt0.x[id.ilow] .+= δx
    pt0.x[id.iupp] .-= δx
  end

  # 2nd safety s
  δs_l2 = (id.nlow == 0) ? zero(T) : δs_l1 + xs_l1 / sum(itd.x_m_lvar) / 2
  δs_u2 = (id.nupp == 0) ? zero(T) : δs_u1 + xs_u1 / sum(itd.uvar_m_x) / 2
  δs = max(δs_l2, δs_u2)
  @. pt0.s_l = pt0.s_l + δs
  @. pt0.s_u = pt0.s_u + δs

  # deal with the compensation phaenomenon in x if irng != []
  update_rngbounds!(pt0.x, id.irng, fd.lvar, fd.uvar, T(1e-4))

  # verify bounds
  @assert all(pt0.x .> fd.lvar) && all(pt0.x .< fd.uvar)
  id.nlow > 0 && @assert all(pt0.s_l .> zero(T))
  id.nupp > 0 && @assert all(pt0.s_u .> zero(T))

  # update itd
  update_IterData!(itd, pt0, fd, id, false)
end

function update_rngbounds!(x::Vector{T}, irng, lvar, uvar, ϵ) where {T <: Real}
  @inbounds @simd for i in irng
    if lvar[i] >= x[i]
      x[i] = lvar[i] + ϵ
    end
    if x[i] >= uvar[i]
      x[i] = uvar[i] - ϵ
    end
    if (lvar[i] < x[i] < uvar[i]) == false
      x[i] = (lvar[i] + uvar[i]) / 2
    end
  end
end
