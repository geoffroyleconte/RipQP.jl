function check_bounds(x::T, lvar, uvar) where {T<:Real}
  ϵ = T(1.0e-4)
  if lvar >= x
    x = lvar + ϵ
  end
  if x >= uvar
    x = uvar - ϵ
  end
  if (lvar < x < uvar) == false
    x = (lvar + uvar) / 2
  end
  return x
end

# starting points
function update_rngbounds!(x, irng, lvar, uvar, ϵ) where {T <: Real} 
  @views broadcast!(check_bounds, x[irng], x[irng], lvar[irng], uvar[irng])
end

# α computation (in iterations.jl)
function ddir(dir_vi::T, vi::T) where {T<:Real}
  if dir_vi < zero(T)
    α_new = -vi * T(0.999) / dir_vi
    return α_new
  end
  return one(T)
end

function compute_α_dual(v, dir_v, store_v)
  map!(ddir, store_v, dir_v, v)
  return minimum(store_v)
end

function pdir_l(dir_vi::T, lvari::T, vi::T) where {T<:Real}
  if dir_vi < zero(T)
    α_new = (lvari - vi) * T(0.999) / dir_vi
    return α_new
  end
  return one(T)
end

function pdir_u(dir_vi::T, uvari::T, vi::T) where {T<:Real}
  if dir_vi > zero(T)
    α_new = (uvari - vi) * T(0.999) / dir_vi
    return α_new
  end
  return one(T)
end

function compute_α_primal(v, dir_v, lvar, uvar, store_v)
  map!(pdir_l, store_v, dir_v, lvar, v)
  α_l = minimum(store_v)
  map!(pdir_u, store_v, dir_v, uvar, v)
  α_u = minimum(store_v)
  return min(α_l, α_u)
end