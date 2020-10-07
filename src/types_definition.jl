mutable struct QM_FloatData{T}
    Qvals :: Vector{T}
    Avals :: Vector{T}
    b     :: Vector{T}
    c     :: Vector{T}
    c0    :: T
    lvar  :: Vector{T}
    uvar  :: Vector{T}
end

mutable struct QM_IntData
    Qrows  :: Vector{Int}
    Qcols  :: Vector{Int}
    Arows  :: Vector{Int}
    Acols  :: Vector{Int}
    ilow   :: Vector{Int}
    iupp   :: Vector{Int}
    irng   :: Vector{Int}
    n_rows :: Int
    n_cols :: Int
    n_low  :: Int
    n_upp  :: Int
end

mutable struct tolerances{T}
    pdd    :: T
    rb     :: T
    rc     :: T
    tol_rb :: T
    tol_rc :: T
    μ      :: T
    Δx     :: T
end

mutable struct point
    x    :: Vector
    λ    :: Vector
    s_l  :: Vector
    s_u  :: Vector
end

mutable struct residuals
    rb   :: Vector
    rc   :: Vector
    rbNorm
    rcNorm
    n_Δx
end

mutable struct regularization
    ρ
    δ
    ρ_min
    δ_min
end

mutable struct iter_data
    tmp_diag    :: Vector
    diag_Q      :: Vector
    J_augm      :: AbstractMatrix
    J_fact
    J_P
    diagind_J   :: Vector{Int}
    x_m_lvar    :: Vector
    uvar_m_x    :: Vector
    Qx          :: Vector
    ATλ         :: Vector
    Ax          :: Vector
    xTQx_2
    cTx
    pri_obj
    dual_obj
    μ
    pdd
    l_pdd       :: Vector
    mean_pdd
end

mutable struct preallocated_data
    Δ_aff            :: Vector
    Δ_cc             :: Vector
    Δ                :: Vector
    Δ_xλ             :: Vector
    x_m_l_αΔ_aff     :: Vector
    u_m_x_αΔ_aff     :: Vector
    s_l_αΔ_aff       :: Vector
    s_u_αΔ_aff       :: Vector
    rxs_l            :: Vector
    rxs_u            :: Vector
end

mutable struct stop_crit
    optimal   :: Bool
    small_Δx  :: Bool
    small_μ   :: Bool
    tired     :: Bool
end

mutable struct safety_compt
    c_catch  :: Int
    c_pdd    :: Int
end
