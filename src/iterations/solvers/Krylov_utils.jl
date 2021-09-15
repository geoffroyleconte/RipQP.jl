function KSolver(s::Symbol)
  if s == :minres
    return :MinresSolver
  elseif s == :minres_qlp
    return :MinresQlpSolver
  elseif s == :cg
    return :CgSolver
  elseif s == :lsqr
    return :LsqrSolver
  elseif s == :lsmr
    return LsmrSolver
  elseif s == :qmr
    return :QmrSolver
  elseif s == :usymqr
    return :UsymqrSolver
  elseif s == :bicgstab
    return :BicgstabSolver
  end
end

ksolve!(
  KS::MinresSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = minres!(
  KS,
  K,
  rhs,
  M = M,
  verbose = verbose,
  atol = zero(T),
  rtol = zero(T),
  ratol = atol,
  rrtol = rtol,
)

ksolve!(
  KS::MinresQlpSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = minres_qlp!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::CgSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = cg!(KS, K, rhs, M = M, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::QmrSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = qmr!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::UsymqrSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = usymqr!(KS, K, rhs, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::BicgstabSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
) where {T, S} = bicgstab!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol)

ksolve!(
  KS::LsqrSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  λ::T = T(eps(T)^(1/4))
) where {T, S} = lsqr!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol, λ = λ)

ksolve!(
  KS::LsmrSolver{T, S},
  K,
  rhs::AbstractVector{T},
  M;
  verbose::Integer = 0,
  atol::T = T(sqrt(eps(T))),
  rtol::T = T(sqrt(eps(T))),
  λ::T = T(eps(T)^(1/4))
) where {T, S} = lsmr!(KS, K, rhs, verbose = verbose, atol = atol, rtol = rtol, λ = λ)

function kscale!(rhs::AbstractVector{T}) where {T <: Real}
  rhsNorm = norm(rhs)
  if rhsNorm != zero(T)
    rhs ./= rhsNorm
  end
  return rhsNorm
end

function kunscale!(sol::AbstractVector{T}, rhsNorm::T) where {T <: Real}
  if rhsNorm != zero(T)
    sol .*= rhsNorm
  end
end

function update_kresiduals_history!(
  res::AbstractResiduals{T},
  K,
  sol::AbstractVector{T},
  rhs::AbstractVector{T},
) where {T <: Real}
  if typeof(res) <: ResidualsHistory
    mul!(res.KΔxy, K, sol) # krylov residuals
    res.Kres = res.KΔxy .- rhs
  end
end

function update_kresiduals_history_LS!(
  res::AbstractResiduals{T},
  K,
  sol::AbstractVector{T},
  rhs::AbstractVector{T},
  vtmp::AbstractVector{T},
  δ::T,
) where {T <: Real}
  if typeof(res) <: ResidualsHistory
    mul!(vtmp, K', sol)
    mul!(res.KΔxy, K, vtmp) # krylov residuals
    res.KΔxy .+= δ .* sol
    mul!(res.Kres, K, rhs) 
    res.Kres .= res.KΔxy .- res.Kres
  end
end
