abstract type PreallocatedDataNormal{T <: Real, S} <: PreallocatedData{T, S} end

abstract type PreallocatedDataNormalKrylov{T <: Real, S} <: PreallocatedDataNormal{T, S} end

include("K1Krylov.jl")