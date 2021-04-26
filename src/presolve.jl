function presolve(QM0 :: QuadraticModel)
    T = eltype(QM0.data.c)
    pb = MathOptPresolve.ProblemData{T}()

    # load ps with QM0
    A = sparse(QM0.data.Arows, QM0.data.Acols, QM0.data.Avals, QM0.meta.ncon, QM0.meta.nvar)
    MathOptPresolve.load_problem!(pb, "pb0", true, QM0.data.c, QM0.data.c0, 
                                  A, QM0.meta.lcon, QM0.meta.ucon, QM0.meta.lvar, QM0.meta.uvar)

    ps = MathOptPresolve.PresolveData(pb)
    status = MathOptPresolve.presolve!(ps)
    MathOptPresolve.extract_reduced_problem!(ps)
    QM = extract_qm_presolved(ps.pb_red, QM0.meta.name)

    return QM, ps
end

function extract_qm_presolved(pb_red :: MathOptPresolve.ProblemData{T}, QM_name :: String) where {T<:Real}
    m, n = pb_red.ncon, pb_red.nvar
    nzA = 0
    for i = 1:pb_red.ncon
        nzA += length(pb_red.arows[i].nzind)
    end
    aI = Vector{Int}(undef, nzA)
    aJ = Vector{Int}(undef, nzA)
    aV = Vector{T}(undef, nzA)
    nz_red = 0
    for (j, col) in enumerate(pb_red.acols)
        for (i, aij) in zip(col.nzind, col.nzval)
            nz_red += 1
            aI[nz_red] = i
            aJ[nz_red] = j
            aV[nz_red] = aij
        end
    end

    return QuadraticModel(pb_red.obj, Int[], Int[], T[],
                          Arows=aI, Acols=aJ, Avals=aV,
                          lcon=pb_red.lcon, ucon=pb_red.ucon, lvar=pb_red.lvar, uvar=pb_red.uvar,
                          c0=pb_red.obj0, x0 = zeros(T, pb_red.nvar), name=QM_name)
end

function postsolve(ps :: MathOptPresolve.PresolveData{T}, pt :: Point{T}, itd :: IterData{T}, QM0 :: QuadraticModel) where {T<:Real}

    sol_inner = MathOptPresolve.Solution{T}(ps.pb_red.ncon, ps.pb_red.nvar)
    sol_inner.x = pt.x[1:ps.pb_red.nvar]
    sol_inner.Ax = itd.Ax
    # sol_inner.y_lower = stats.multipliers

    # Post-solve
    sol_outer = MathOptPresolve.Solution{T}(ps.pb0.ncon, ps.pb0.nvar)
    MathOptPresolve.postsolve!(sol_outer, sol_inner, ps)
    x_opt = sol_outer.x[1:QM0.meta.nvar]
    itd.pri_obj = dot(QM0.data.c, x_opt) + QM0.data.c0 
    return x_opt
end