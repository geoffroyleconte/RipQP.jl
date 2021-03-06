export PreconditionerDataK2
"""
Abstract type that defines preconditioners for the K2 formulation.
The available preconditioners are:
- Identity
- Jacobi 
"""
abstract type PreconditionerDataK2{T <: Real, S} end

include("identity.jl")
include("jacobi.jl")
