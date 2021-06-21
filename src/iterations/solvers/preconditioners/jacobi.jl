mutable struct JacobiData{T<:Real, S} <: PreconditionerDataK2{T, S}
  P::LinearOperator{T}
  invDiagK::S
end

function Jacobi(id :: QM_IntData, fd::QM_FloatData{T}, regu :: Regularization{T}, D :: AbstractVector{T}, K::AbstractMatrix{T}) where {T<:Real} 
  invDiagK = (one(T)/regu.δ) .* ones(T, id.nvar+id.ncon)
  invDiagK[1:id.nvar] .= .-one(T) ./ D
  P = opDiagonal(invDiagK)
  return JacobiData{T, typeof(fd.c)}(P, invDiagK)
end 

function update_preconditioner!(pdat :: JacobiData{T}, pad :: PreallocatedData{T}, itd :: IterData{T}, 
                                pt :: Point{T}, id :: QM_IntData, fd:: QM_FloatData{T}, cnts :: Counters) where {T<:Real}

  pad.pdat.invDiagK .= @views abs.(one(T) ./ pad.K.nzval[pad.diagind_K])
  pad.pdat.P = opDiagonal(pad.pdat.invDiagK)
end
