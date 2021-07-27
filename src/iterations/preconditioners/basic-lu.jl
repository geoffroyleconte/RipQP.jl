mutable struct BasicLUData{T <: Real, S, F, Ftu, Faw} <: PreconditionerDataK2{T, S}
  P::LinearOperator{T, F, Ftu, Faw}
  diagQ::S
  invDiagK::S
end

function BasicLU(
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  D::AbstractVector{T},
  K::LinearOperator{T},
) where {T <: Real}
  # regularise problem
  basis, F = maxvolume_basis(A, volumetol=1.1, maxpass=2)
  rank_deficiency = count(>(m), basis)
  notbasis = setdiff(1:m+n, basis)
  B = Abis[:, basis]
  N = Abis[:, notbasis]
  

  return JacobiData(P, diagQ, invDiagK)
end

function update_preconditioner!(
  pdat::BasicLUData{T},
  pad::PreallocatedData{T},
  itd::IterData{T},
  pt::Point{T},
  id::QM_IntData,
  fd::QM_FloatData{T},
  cnts::Counters,
) where {T <: Real}
  if typeof(pad) <: PreallocatedData_K2_5minres
    pad.pdat.invDiagK[1:(id.nvar)] .=
      abs.(one(T) ./ (pad.D .- (pad.pdat.diagQ .* pad.sqrtX1X2 .^ 2)))
  else
    pad.pdat.invDiagK[1:(id.nvar)] .= abs.(one(T) ./ (pad.D .- pad.pdat.diagQ))
  end
  pad.pdat.invDiagK[(id.nvar + 1):end] .= one(T) / pad.regu.δ
end