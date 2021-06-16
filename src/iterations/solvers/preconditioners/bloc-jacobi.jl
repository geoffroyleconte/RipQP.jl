using ExaPF
using KernelAbstractions

mutable struct BlockJacobiData{T<:Real, AT, GAT, VI, GVI, MT, GMT, MI, GMI, SMT} <: PreconditionerDataK2{T}
  P::LinearOperator{T}
  yop::Vector{T}
  BJP::ExaPF.LS.BlockJacobiPreconditioner{AT,GAT,VI,GVI,MT,GMT,MI,GMI,SMT}
end

function BlockJacobi(id :: QM_IntData, fd::QM_FloatData{T}, regu :: Regularization{T}, D :: Vector{T}, K::SparseMatrixCSC{T, Int}) where {T<:Real} 
  nblocks = 4
  BJP = ExaPF.LS.BlockJacobiPreconditioner(K, nblocks, CPU())
  ExaPF.LS.update(BJP, K, CPU())
  yop = similar(fd.c, id.nvar + id.ncon)
  P = LinearOperator(T, id.nvar + id.ncon, id.nvar + id.ncon, true, true, v -> mul!(yop, BJP.P, v)) 
  return BlockJacobiData(P, yop, BJP)
end 

function update_preconditioner!(pdat :: BlockJacobiData{T}, pad :: PreallocatedData{T}, itd :: IterData{T}, 
                                pt :: Point{T}, id :: QM_IntData, fd:: QM_FloatData{T}, cnts :: Counters) where {T<:Real}
                           
  ExaPF.LS.update(pad.pdat.BJP, pad.K, CPU())
  P = LinearOperator(T, id.nvar + id.ncon, id.nvar + id.ncon, true, true, v -> mul!(pad.pdat.yop, pad.pdat.BJP.P, v)) 
end