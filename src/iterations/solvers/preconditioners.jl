abstract type PreconditionerDataK2{T<:Real} end

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

mutable struct SchurData{T<:Real} <: PreconditionerDataK2{T}
  P::LinearOperator{T}
  Up::SparseMatrixCSC{T,Int}
  dp::Vector{T}
  yop::Vector{T}
end

function update_dp!(dp, AT_colptr, AT_rowval, AT_nzval, D, δ, nvar, ncon)
  dp[1: nvar] .= .-D
  dp[nvar+1: end] .= δ
  for j=1:ncon
    for k=AT_colptr[j] : (AT_colptr[j+1] - 1)
      i = AT_rowval[k]
      dp[i] += AT_nzval[k]^2 / δ
    end
  end
end

function invschur!(yop, Up, dp, v)
  Upw = UnitUpperTriangular(Up)
  ldiv!(yop, Upw, v)
  ldiv!(Diagonal(dp), yop)
  ldiv!(Upw', yop)
end

function Schur(id :: QM_IntData, fd::QM_FloatData{T}, regu :: Regularization{T}, D :: Vector{T}, K::SparseMatrixCSC{T, Int}) where {T<:Real} 
  Up = [spzeros(T, id.nvar, id.nvar)                fd.AT;
        spzeros(T, id.ncon, id.nvar)  spzeros(T, id.ncon, id.ncon)]
  Up.nzval ./= regu.δ
  dp = similar(D, id.nvar + id.ncon)
  yop = similar(dp)
  update_dp!(dp, fd.AT.colptr, fd.AT.rowval, fd.AT.nzval, D, regu.δ, id.nvar, id.ncon)
  P = LinearOperator(T, id.nvar + id.ncon, id.nvar + id.ncon, true, true, v -> invschur!(yop, Up, dp, v))
  return SchurData{T}(P, Up, dp, yop)
end 

function update_preconditioner!(pdat :: SchurData{T}, pad :: PreallocatedData{T}, itd :: IterData{T}, 
                                pt :: Point{T}, id :: QM_IntData, fd:: QM_FloatData{T}, cnts :: Counters) where {T<:Real}

  pad.pdat.Up.nzval .= fd.AT.nzval ./ pad.regu.δ
  update_dp!(pad.pdat.dp, fd.AT.colptr, fd.AT.rowval, fd.AT.nzval, pad.D, pad.regu.δ, id.nvar, id.ncon)
  ldltest = ldl(Symmetric(pad.K, :U))
  ldltest.d .= abs.(ldltest.d)
  pad.pdat.P = LinearOperator(T, id.nvar + id.ncon, id.nvar + id.ncon, true, true, v -> ldiv!(pad.pdat.yop, ldltest, v))
  # pad.pdat.P = LinearOperator(T, id.nvar + id.ncon, id.nvar + id.ncon, true, true, v -> invschur!(pad.pdat.yop, pad.pdat.Up, pad.pdat.dp, v))
end