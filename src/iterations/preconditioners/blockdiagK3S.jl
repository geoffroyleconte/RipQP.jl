export BlockDiagK3S

"""
    preconditioner = BlockDiagK3S()

Preconditioner modeling the inverse of K3S block diagonals.
Works with:
- [`K3SKrylovParams`](@ref)
"""
mutable struct BlockDiagK3S <: AbstractPreconditioner end

mutable struct BlockDiagK3SData{T <: Real, S, L} <: PreconditionerData{T, S}
  P::L
  K0::SparseMatrixCSC{T, Int}
  K0_fact::LDLFactorizations.LDLFactorization{T, Int, Int, Int}
  Dl::Diagonal{T, S}
  Du::Diagonal{T, S}
end

function PreconditionerData(
  sp::NewtonKrylovParams{BlockDiagK3S},
  id::QM_IntData,
  fd::QM_FloatData{T},
  regu::Regularization{T},
  K::Union{LinearOperator{T}, AbstractMatrix{T}},
) where {T <: Real}
  D = fill!(similar(fd.c), -regu.ρ_min)
  regu.ρ = regu.ρ_min / 10
  regu.δ = regu.δ_min / 10
  diag_Q = get_diag_Q(fd.Q.data.colptr, fd.Q.data.rowval, fd.Q.data.nzval, id.nvar)
  K0 = create_K2(id, D, fd.Q.data, fd.A, diag_Q, regu)
  K0_fact = ldl(Symmetric(K0, :U))
  Dl = Diagonal(fill!(similar(D, id.nlow), one(T)))
  Du = Diagonal(fill!(similar(D, id.nupp), one(T)))
  K0inv_op = LinearOperator(
    T,
    id.nvar + id.ncon,
    id.nvar + id.ncon,
    true,
    true,
    (res, v) -> ldiv!(res, K0_fact, v),
  )
  P = BlockDiagonalOperator(K0inv_op, Dl, Du)
    # K0_fact.d .= sqrt.(abs.(K0_fact.d))
  # M = BlockDiagonalOperator(
  #   LinearOperator(
  #     T,
  #     id.nvar + id.ncon,
  #     id.nvar + id.ncon,
  #     false,
  #     false,
  #     (res, v) -> ld_div!(res, K0_fact, v),
  #     ),
  #   LinearOperator(Dl),
  #   LinearOperator(Du),
  #   )
  # N = BlockDiagonalOperator(
  #   LinearOperator(
  #     T,
  #     id.nvar + id.ncon,
  #     id.nvar + id.ncon,
  #     false,
  #     false,
  #     (res, v) -> dlt_div!(res, K0_fact, v),
  #   ),
  #   LinearOperator(Dl),
  #   LinearOperator(Du),
  # )
  
  # P = LRPrecond(M, N)
  return BlockDiagK3SData(P, K0, K0_fact, Dl, Du)
end

function update_preconditioner!(
  pdat::BlockDiagK3SData{T},
  pad::PreallocatedDataNewtonKrylov{T},
  itd::IterData{T},
  pt::Point{T},
  id::QM_IntData,
  fd::QM_FloatData{T},
  cnts::Counters,
) where {T <: Real}
  pad.ρv[1] = pad.regu.ρ_min / 10
  pad.δv[1] = pad.regu.δ_min / 10
  # pdat.Dl.diag .= pt.s_l ./ itd.x_m_lvar
  # pdat.Du.diag .= pt.s_u ./ itd.uvar_m_x
  pad.pdat.Dl.diag .= one(T) ./ max.(one(T), itd.x_m_lvar ./ pt.s_l)
  pad.pdat.Du.diag .= one(T) ./ max.(one(T), itd.uvar_m_x ./ pt.s_u)
  # println(real.(eigvals(Matrix(pdat.P)* Matrix(pad.K))))
end

function update_preconditioner!(
  pdat::BlockDiagK3SData{T},
  pad::PreallocatedDataK3_5Krylov{T},
  itd::IterData{T},
  pt::Point{T},
  id::QM_IntData,
  fd::QM_FloatData{T},
  cnts::Counters,
) where {T <: Real}
pad.ρv[1] = pad.regu.ρ_min / 10
pad.δv[1] = pad.regu.δ_min / 10
  # pdat.Dl.diag .= sqrt.(pt.s_l ./ itd.x_m_lvar)
  # pdat.Du.diag .= sqrt.(pt.s_u ./ itd.uvar_m_x)
  pad.pdat.Dl.diag .= sqrt.(one(T) ./ max.(pt.s_l, itd.x_m_lvar))
  pad.pdat.Du.diag .= sqrt.(one(T) ./ max.(pt.s_u, itd.uvar_m_x))
end

function update_preconditioner!(
  pdat::BlockDiagK3SData{T},
  pad::PreallocatedDataK3Krylov{T},
  itd::IterData{T},
  pt::Point{T},
  id::QM_IntData,
  fd::QM_FloatData{T},
  cnts::Counters,
) where {T <: Real}
  pad.ρv[1] = pad.regu.ρ_min / 10
  pad.δv[1] = pad.regu.δ_min / 10
  # pdat.Dl.diag .= sqrt.(pt.s_l ./ itd.x_m_lvar)
  # pdat.Du.diag .= sqrt.(pt.s_u ./ itd.uvar_m_x)
  pad.pdat.Dl.diag .= one(T) ./ max.(pt.s_l, itd.x_m_lvar)
  pad.pdat.Du.diag .= one(T) ./ max.(pt.s_u, itd.uvar_m_x)
end