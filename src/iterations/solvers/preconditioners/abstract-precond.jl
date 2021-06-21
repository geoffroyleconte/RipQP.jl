abstract type PreconditionerDataK2{T<:Real, S} end

include("bloc-jacobi.jl")
include("jacobi.jl")
include("schur-bloc-ldl.jl")