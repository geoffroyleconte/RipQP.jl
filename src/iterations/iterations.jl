include("solve_method.jl")
include("centrality_corr.jl")
include("regularization.jl")
include("system_write.jl")
include("solvers/K2LDL.jl")
include("solvers/K2_5LDL.jl")
include("preconditioners/abstract-precond.jl")
include("solvers/Krylov_pad.jl")
include("solvers/K2Krylov.jl")
include("solvers/K2_5Krylov.jl")

function compute_α_dual(v, dir_v)
  n = length(v)
  T = eltype(v)
  if n == 0
    return one(T)
  end
  α = one(T)
  @inbounds for i = 1:n
    if dir_v[i] < zero(T)
      α_new = -v[i] * T(0.999) / dir_v[i]
      if α_new < α
        α = α_new
      end
    end
  end
  return α
end

function compute_α_primal(v, dir_v, lvar, uvar)
  n = length(v)
  T = eltype(v)
  α_l, α_u = one(T), one(T)
  @inbounds for i = 1:n
    if dir_v[i] > zero(T)
      α_u_new = (uvar[i] - v[i]) * T(0.999) / dir_v[i]
      if α_u_new < α_u
        α_u = α_u_new
      end
    elseif dir_v[i] < zero(T)
      α_l_new = (lvar[i] - v[i]) * T(0.999) / dir_v[i]
      if α_l_new < α_l
        α_l = α_l_new
      end
    end
  end
  return min(α_l, α_u)
end

@inline function compute_αs(x, s_l, s_u, lvar, uvar, Δxy, Δs_l, Δs_u, nvar)
  α_pri = @views compute_α_primal(x, Δxy[1:nvar], lvar, uvar)
  α_dual_l = compute_α_dual(s_l, Δs_l)
  α_dual_u = compute_α_dual(s_u, Δs_u)
  return α_pri, min(α_dual_l, α_dual_u)
end

@inline function compute_μ(x_m_lvar, uvar_m_x, s_l, s_u, nb_low, nb_upp)
  return (dot(s_l, x_m_lvar) + dot(s_u, uvar_m_x)) / (nb_low + nb_upp)
end

function update_pt!(x, y, s_l, s_u, α_pri, α_dual, Δxy, Δs_l, Δs_u, ncon, nvar)
  x .= @views x .+ α_pri .* Δxy[1:nvar]
  y .= @views y .+ α_dual .* Δxy[(nvar + 1):(ncon + nvar)]
  s_l .= s_l .+ α_dual .* Δs_l
  s_u .= s_u .+ α_dual .* Δs_u
end

function safe_boundary(v::T) where {T <: Real}
  if v == 0
    v = eps(T)^2
  end
  return v
end

# "security" if x is too close from lvar or uvar
function boundary_safety!(x_m_lvar, uvar_m_x)
  x_m_lvar .= safe_boundary.(x_m_lvar)
  uvar_m_x .= safe_boundary.(uvar_m_x)
end

function update_IterData!(itd, pt, fd, id, safety)
  T = eltype(itd.x_m_lvar)
  itd.x_m_lvar .= @views pt.x[id.ilow] .- fd.lvar[id.ilow]
  itd.uvar_m_x .= @views fd.uvar[id.iupp] .- pt.x[id.iupp]
  safety && boundary_safety!(itd.x_m_lvar, itd.uvar_m_x)

  itd.μ = compute_μ(itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, id.nlow, id.nupp)
  itd.Qx = mul!(itd.Qx, Symmetric(fd.Q, :U), pt.x)
  itd.xTQx_2 = dot(pt.x, itd.Qx) / 2
  itd.ATy = mul!(itd.ATy, fd.AT, pt.y)
  itd.Ax = mul!(itd.Ax, fd.AT', pt.x)
  itd.cTx = dot(fd.c, pt.x)
  itd.pri_obj = itd.xTQx_2 + itd.cTx + fd.c0
  if typeof(pt.x) <: Vector
    itd.dual_obj = @views dot(fd.b, pt.y) - itd.xTQx_2 + dot(pt.s_l, fd.lvar[id.ilow]) -
           dot(pt.s_u, fd.uvar[id.iupp]) + fd.c0
  else # views and dot not working with GPU arrays
    itd.dual_obj = dual_obj_gpu(
      fd.b,
      pt.y,
      itd.xTQx_2,
      pt.s_l,
      pt.s_u,
      fd.lvar,
      fd.uvar,
      fd.c0,
      id.ilow,
      id.iupp,
      itd.store_vdual_l,
      itd.store_vdual_u,
    )
  end
  itd.pdd = abs(itd.pri_obj - itd.dual_obj) / (one(T) + abs(itd.pri_obj))
end

function update_data!(
  pt::Point{T},
  α_pri::T,
  α_dual::T,
  itd::IterData{T},
  pad::PreallocatedData{T},
  res::Residuals{T},
  fd::QM_FloatData{T},
  id::QM_IntData,
) where {T <: Real}

  # (x, y, s_l, s_u) += α * Δ
  update_pt!(
    pt.x,
    pt.y,
    pt.s_l,
    pt.s_u,
    α_pri,
    α_dual,
    itd.Δxy,
    itd.Δs_l,
    itd.Δs_u,
    id.ncon,
    id.nvar,
  )
  update_IterData!(itd, pt, fd, id, true)

  #update Residuals
  res.rb .= itd.Ax .- fd.b
  res.rc .= itd.ATy .- itd.Qx .- fd.c
  res.rc[id.ilow] .+= pt.s_l
  res.rc[id.iupp] .-= pt.s_u
  # update stopping criterion values:
  #         rcNorm, rbNorm = norm(rc), norm(rb)
  #         xNorm = norm(x)
  #         yNorm = norm(y)
  #         optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb * max(1, bNorm + ANorm * xNorm) &&
  #                     rcNorm < ϵ_rc * max(1, cNorm + QNorm * xNorm + ANorm * yNorm)
  res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
  res.history == true && push_history_residuals!(res, itd.pdd, pad)
end

function iter!(
  pt::Point{T},
  itd::IterData{T},
  fd::Abstract_QM_FloatData{T},
  id::QM_IntData,
  res::Residuals{T},
  sc::StopCrit{Tc},
  dda::DescentDirectionAllocs{T},
  pad::PreallocatedData{T},
  ϵ::Tolerances{T},
  cnts::Counters,
  T0::DataType,
  display::Bool,
) where {T <: Real, Tc <: Real}
  @inbounds while cnts.k < sc.max_iter && !sc.optimal && !sc.tired
    time_fact = (cnts.kc == -1) ? time_ns() : UInt(0) # timer centrality_corr factorization
    out = update_pad!(pad, dda, pt, itd, fd, id, res, cnts, T0) # update data for the solver! function used
    time_fact = (cnts.kc == -1) ? time_ns() - time_fact : 0.0
    out == 1 && break

    time_solve = (cnts.kc == -1) ? time_ns() : 0.0 # timer centrality_corr solve

    # Solve system to find a direction of descent 
    out = update_dd!(dda, pt, itd, fd, id, res, pad, cnts, T0)
    out == 1 && break

    if typeof(pt.x) <: Vector
      α_pri, α_dual =
        compute_αs(pt.x, pt.s_l, pt.s_u, fd.lvar, fd.uvar, itd.Δxy, itd.Δs_l, itd.Δs_u, id.nvar)
    else
      α_pri, α_dual = compute_αs_gpu(
        pt.x,
        pt.s_l,
        pt.s_u,
        fd.lvar,
        fd.uvar,
        itd.Δxy,
        itd.Δs_l,
        itd.Δs_u,
        id.nvar,
        itd.store_vpri,
        itd.store_vdual_l,
        itd.store_vdual_u,
      )
    end

    if cnts.kc > 0   # centrality corrections
      α_pri, α_dual =
        multi_centrality_corr!(dda, pad, pt, α_pri, α_dual, itd, fd, id, cnts, res, T0)
      ## TODO replace by centrality_corr.jl, deal with α
    end

    update_data!(pt, α_pri, α_dual, itd, pad, res, fd, id) # update point, residuals, objectives...
    time_solve = (cnts.kc == -1) ? time_ns() - time_solve : UInt(0)
    (cnts.kc == -1) && nb_corrector_steps!(cnts, time_fact, time_solve)

    sc.optimal = itd.pdd < ϵ.pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc
    sc.small_μ = itd.μ < ϵ.μ

    cnts.k += 1
    if T == Float32
      cnts.km += 1
    elseif T == Float64
      cnts.km += 4
    else
      cnts.km += 16
    end

    sc.Δt = time() - sc.start_time
    sc.tired = sc.Δt > sc.max_time

    if display == true
      @info log_row(
        Any[
          cnts.k,
          itd.minimize ? itd.pri_obj : -itd.pri_obj,
          itd.pdd,
          res.rbNorm,
          res.rcNorm,
          α_pri,
          α_dual,
          itd.μ,
        ],
      )
    end
  end
end
