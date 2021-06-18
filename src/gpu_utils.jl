using .CUDA

function sparse_transpose_dropzeros(rows, cols, vals::CuVector, nrows, ncols) 
  CPUvals = Vector(vals)
  MT = sparse(cols, rows, CPUvals, ncols, nrows)
  dropzeros!(MT)
  MTGPU = CUDA.CUSPARSE.CuSparseMatrixCSR(MT)
  return MTGPU
end

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

@inline function compute_αs(x::Vector, s_l::Vector, s_u::Vector, lvar::Vector, uvar::Vector, Δxy::Vector, 
                            Δs_l::Vector, Δs_u::Vector, nvar, 
                            store_vpri::CuVector, store_vdual_l::CuVector, store_vdual_u::CuVector)
  α_pri = @views compute_α_primal(x, Δxy[1:nvar], lvar, uvar, store_vpri)
  α_dual_l = compute_α_dual(s_l, Δs_l, store_vdual_l)
  α_dual_u = compute_α_dual(s_u, Δs_u, store_vdual_u)
  return α_pri, min(α_dual_l, α_dual_u)
end

# Infinity norm
# import LinearAlgebra.norm
function LinearAlgebra.norm(v::CUDA.CuVector{T}; p::Real=2) where {T <: Real}
  if p == 2
    return norm(v)
  elseif p == Inf
    return maximum(abs.(v))
  end
  return zero(T)
end

@inline function warp_reduce(val)
  val += shfl_down_sync(0xffffffff, val, 16, 32)
  val += shfl_down_sync(0xffffffff, val, 8, 32)
  val += shfl_down_sync(0xffffffff, val, 4, 32)
  val += shfl_down_sync(0xffffffff, val, 2, 32)
  val += shfl_down_sync(0xffffffff, val, 1, 32)
  return val
end

# preconditioner
function update_dp!(dp, AT::CUDA.CUSPARSE.CuSparseMatrixCSR{T}, D, δ, nvar, ncon) where T

  function kernel(dp, AT_rowptr, AT_colval, AT_nzval, D, δ, nrows, ncols)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    CUDA.assume(warpsize() == 32)
    warp_size = 32
    warp_id = thread_id ÷ warp_size
    i = warp_id + 1
    lane = (thread_id - 1) % warp_size
    sum = zero(T)
    if i <= nrows
      for k = (AT_rowptr[i] + lane): warp_size: (AT_colptr[i + 1] - 1)
        sum += AT_nzval[k]^2 / δ
      end
    end
    sum = warp_reduce(sum)
    if lane == 0 && i <= nrows
      dp[i] = sum
    end
    return nothing
  end
  threads = min(length(x), 256)
  blocks = ceil(Int, length(AT.colval)/threads)
  @cuda name="dp" threads=threads blocks=blocks kernel(dp, AT.rowPtr, AT.colVal, AT.nzVal, D, δ, size(AT, 1), size(AT, 2))
  return y2
end