using ExaPF
using KernelAbstractions

mutable struct BlockJacobiData{T<:Real, S, AT, GAT, VI, GVI, MT, GMT, MI, GMI, SMT} <: PreconditionerDataK2{T, S}
  P::LinearOperator{T}
  yop::S
  BJP::ExaPF.LS.BlockJacobiPreconditioner{AT,GAT,VI,GVI,MT,GMT,MI,GMI,SMT}
end

function BlockJacobi(id :: QM_IntData, fd::QM_FloatData{T}, regu :: Regularization{T}, D :: AbstractVector{T}, K::AbstractMatrix{T}) where {T<:Real} 
  nblocks = 4
  if typeof(fd.c) <: Vector
    BJP = ExaPF.LS.BlockJacobiPreconditioner(K, nblocks, CPU())
    ExaPF.LS.update(BJP, K, CPU())
  else
    BJP = ExaPF.LS.BlockJacobiPreconditioner(K, nblocks, CUDADevice())
    ExaPF.LS.update(BJP, K, CUDADevice())
  end
  yop = similar(fd.c, id.nvar + id.ncon)
  P = LinearOperator(T, id.nvar + id.ncon, id.nvar + id.ncon, true, true, v -> mul!(yop, BJP.P, v)) 
  return BlockJacobiData(P, yop, BJP)
end 

function update_preconditioner!(pdat :: BlockJacobiData{T}, pad :: PreallocatedData{T}, itd :: IterData{T}, 
                                pt :: Point{T}, id :: QM_IntData, fd:: QM_FloatData{T}, cnts :: Counters) where {T<:Real}
         
  if typeof(fd.c) <: Vector
    ExaPF.LS.update(pad.pdat.BJP, pad.K, CPU())
  else
    ExaPF.LS.update(pad.pdat.BJP, pad.K, CUDADevice())
  end
  P = LinearOperator(T, id.nvar + id.ncon, id.nvar + id.ncon, true, true, v -> mul!(pad.pdat.yop, pad.pdat.BJP.P, v)) 
end