using ExaPF
using KernelAbstractions

mutable struct BlockJacobiData{T<:Real, AT, GAT, VI, GVI, MT, GMT, MI, GMI, SMT} <: PreconditionerDataK2{T}
  P::LinearOperator{T}
  yop::Vector{T}
  BJP::ExaPF.LS.BlockJacobiPreconditioner{AT,GAT,VI,GVI,MT,GMT,MI,GMI,SMT}
  Ks::SparseMatrixCSC{T, Int}
end

function BlockJacobi(id :: QM_IntData, fd::QM_FloatData{T}, regu :: Regularization{T}, D :: Vector{T}, K::SparseMatrixCSC{T, Int}) where {T<:Real} 
  nblocks = 4
  Ks = K .+ K' .- Diagonal(K)
  BJP = ExaPF.LS.BlockJacobiPreconditioner(Ks, nblocks, CPU())
  ExaPF.LS.update(BJP, Ks, CPU())
  yop = similar(fd.c, id.nvar + id.ncon)
  P = LinearOperator(T, id.nvar + id.ncon, id.nvar + id.ncon, true, true, v -> mul!(yop, BJP.P, v)) 
  display(BJP.P)
  return BlockJacobiData(P, yop, BJP, Ks)
end 

function update_preconditioner!(pdat :: BlockJacobiData{T}, pad :: PreallocatedData{T}, itd :: IterData{T}, 
                                pt :: Point{T}, id :: QM_IntData, fd:: QM_FloatData{T}, cnts :: Counters) where {T<:Real}

  pad.pdat.Ks .= pad.K .+ pad.K' .- Diagonal(pad.K)                             
  ExaPF.LS.update(pad.pdat.BJP, pad.pdat.Ks, CPU())
  P = LinearOperator(T, id.nvar + id.ncon, id.nvar + id.ncon, true, true, v -> mul!(pad.pdat.yop, pad.pdat.BJP.P, v)) 
end