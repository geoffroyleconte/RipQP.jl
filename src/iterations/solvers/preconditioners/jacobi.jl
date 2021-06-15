mutable struct JacobiData{T<:Real} <: PreconditionerDataK2{T}
  P::LinearOperator{T}
  invDiagK :: Vector{T}
end

function Jacobi(id :: QM_IntData, fd::QM_FloatData{T}, regu :: Regularization{T}, D :: Vector{T}, K::SparseMatrixCSC{T, Int}) where {T<:Real} 
  invDiagK = (one(T)/regu.δ) .* ones(T, id.nvar+id.ncon)
  invDiagK[1:id.nvar] .= .-one(T) ./ D
  P = opDiagonal(invDiagK)
  return JacobiData{T}(P, invDiagK)
end 

function update_preconditioner!(pdat :: JacobiData{T}, pad :: PreallocatedData{T}, itd :: IterData{T}, 
                                pt :: Point{T}, id :: QM_IntData, fd:: QM_FloatData{T}, cnts :: Counters) where {T<:Real}

  pad.pdat.invDiagK .= @views abs.(one(T) ./ pad.K.nzval[pad.diagind_K])
  pad.pdat.P = opDiagonal(pad.pdat.invDiagK)
end
