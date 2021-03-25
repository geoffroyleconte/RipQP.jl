using LinearOperators, LimitedLDLFactorizations

"Solve L|D|Lᵀ y = v"
function ldiv2!(y::AbstractVector, F::LimitedLDLFactorization, v::AbstractVector)
    D = F.D
    P = F.P
    Lp = F.L.colptr
    Li = F.L.rowval
    Lx = F.L.nzval
    y .= v
    n = length(v)
    x = view(y, P)
    @inbounds for j = 1:n
        xj = x[j]
        @inbounds for p = Lp[j] : (Lp[j+1] - 1)
        x[Li[p]] -= Lx[p] * xj
        end
    end
    @inbounds for j = 1:n
        x[j] /= abs(D[j])
    end
    @inbounds for j = n:-1:1
        xj = x[j]
        @inbounds for p = Lp[j] : (Lp[j+1] - 1)
            xj -= Lx[p] * x[Li[p]]
        end
        x[j] = xj
    end
    return y
end

"""
    P⁻¹ = ildl(K)

P⁻¹ is a linear operator that models (L|D|Lᵀ)⁻¹.

# Arguments
- `A::AbstractMatrix{T}`: matrix to factorize (its strict lower triangle and
                               diagonal will be extracted)
# Keyword arguments
- `memory::Int=0`: extra amount of memory to allocate for the incomplete factor `L`.
                   The total memory allocated is nnz(T) + n * `memory`, where
                   `T` is the strict lower triangle of A and `n` is the size of `A`.
- `α::T=zero(T)`: initial value of the shift in case the incomplete LDLᵀ
                 factorization of `A` is found to not exist. The shift will be
                 gradually increased from this initial value until success.
- `droptol::T=zero(T)`: to further sparsify `L`, all elements with magnitude smaller
                       than `droptol` are dropped.
"""
function ildl(K::AbstractMatrix{T}; kwargs...) where T
    F = lldl(K; kwargs...)
    y = similar(F.D)
    n = length(y)
    return LinearOperator(T, n, n, true, true, v -> ldiv2!(y, F, v))
end

function opilldl(F :: LimitedLDLFactorization{T,Ti}, y :: Vector{T}) where {T<:Real, Ti<:Integer}
    n = length(y)
    return LinearOperator(T, n, n, true, true, v -> ldiv2!(y, F, v))
end