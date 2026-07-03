# Schema context so our apply_schema methods fire only for us
struct MatryoshkaCtx <: StatsModels.StatisticalModel end

struct GroupTerm <: StatsModels.AbstractTerm
    group::Symbol
end

function StatsModels.apply_schema(
        t::StatsModels.FunctionTerm{typeof(|)}, sch::StatsModels.Schema, Mod::Type{MatryoshkaCtx}
    )
    lhs, rhs = t.args   # StatsModels ≥ 0.7: FunctionTerm stores parsed args in .args
    lhs isa StatsModels.ConstantTerm && lhs.n == 1 ||
        throw(ArgumentError("only random intercepts `(1 | g)` are supported in v0, got `($(lhs) | $(rhs))`"))
    rhs isa StatsModels.Term ||
        throw(ArgumentError("grouping must be a single column, got `$(rhs)`"))
    return GroupTerm(rhs.sym)
end

function lower(lik::Likelihood, tbl)
    isempty(extra_formulas(lik)) || throw(
        ArgumentError(
            "formulas on family parameters (distributional regression) are not yet supported; " *
                "found $(length(extra_formulas(lik))) extra formula(s)"
        )
    )
    cols = Tables.columntable(tbl)
    f = response_formula(lik)
    for v in StatsModels.termvars(f)
        haskey(cols, v) || throw(
            ArgumentError(
                "variable :$v from the formula is not a column of the data; " *
                    "available columns: $(collect(keys(cols)))"
            )
        )
    end
    sch = StatsModels.schema(f, cols)
    fc = StatsModels.apply_schema(f, sch, MatryoshkaCtx)
    y = StatsModels.modelcols(fc.lhs, cols)
    n = length(y)

    comps = AbstractComponent[]
    fixed_terms = []
    # StatsModels ≥ 0.7: the concrete RHS after apply_schema is a MatrixTerm; its
    # constituent terms are exposed via `StatsModels.terms` (no `terms_tuple` accessor).
    for t in StatsModels.terms(fc.rhs)
        if t isa StatsModels.InterceptTerm{true}
            push!(comps, Intercept(n))
        elseif t isa GroupTerm
            col = Tables.getcolumn(cols, t.group)
            levels = unique(col)
            idx = [findfirst(==(v), levels) for v in col]
            push!(comps, RandomIntercept(t.group, idx, collect(levels)))
        elseif t isa StatsModels.InterceptTerm{false}
            # explicit 0: no intercept component
        else
            push!(fixed_terms, t)
        end
    end
    if !isempty(fixed_terms)
        mt = StatsModels.collect_matrix_terms(StatsModels.TupleTerm(fixed_terms))
        X = Matrix{Float64}(StatsModels.modelcols(mt, cols))
        names = Symbol.(StatsModels.coefnames(mt))
        names isa Vector || (names = [names])
        push!(comps, FixedEffects(X, names, mt))
    end
    isempty(comps) && throw(
        ArgumentError(
            "model has no predictor terms; formula `$(f)` has no intercept, fixed, or " *
                "random effect to build a model from"
        )
    )
    return Tuple(comps), collect(y), sch
end
