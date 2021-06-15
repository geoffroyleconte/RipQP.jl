abstract type PreconditionerDataK2{T<:Real} end

include("bloc-jacobi.jl")
include("jacobi.jl")
include("schur-bloc-ldl.jl")