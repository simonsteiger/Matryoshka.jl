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

function check_columns(vars, cols)
    for v in vars
        haskey(cols, v) || throw(
            ArgumentError(
                "variable :$v from the formula is not a column of the data; " *
                    "available columns: $(collect(keys(cols)))"
            )
        )
    end
    return nothing
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
    check_columns(StatsModels.termvars(f), cols)
    # A bare column reference (`y`) parses to a plain `Term`; a transformed response
    # (`log(y)`) parses to a `FunctionTerm`, and a multivariate response (`y1 + y2`)
    # parses to a `Tuple` of `Term`s. Neither is supported in v0 — catch both here,
    # at model()-time, rather than silently building a wrong model (transformed: the
    # rebuild path re-collects the raw column, not the transform) or failing later at
    # sampling time (multivariate: `response` picks one column, wrong `n`).
    f.lhs isa StatsModels.Term || throw(
        ArgumentError(
            "left-hand side of the response formula must be a single, untransformed " *
                "column; transformed responses (e.g. `log(y) ~ x`) and multivariate " *
                "responses (e.g. `y1 + y2 ~ x`) are not supported in v0, got `$(f.lhs)`"
        )
    )
    sch = StatsModels.schema(f, cols)
    fc = StatsModels.apply_schema(f, sch, MatryoshkaCtx)
    y = StatsModels.modelcols(fc.lhs, cols)
    n = length(y)

    # Build `comps` as a genuine (growing) `Tuple` rather than pushing onto an
    # `AbstractComponent[]` vector: converting a `Vector{AbstractComponent}` to
    # a `Tuple` at the end yields an abstractly-eltyped `Tuple{Vararg{AbstractComponent}}`,
    # which makes every downstream consumer (notably the recursive `sum_contribs`
    # in model.jl, which recurses over `components` on every log-density
    # evaluation) type-unstable — JET flags this as a runtime-dispatch/failed-to-
    # optimize-due-to-recursion chain. Growing a real `Tuple` keeps each element's
    # concrete type in the tuple's type parameters (`Tuple{Intercept,FixedEffects}`
    # etc.), so `sum_contribs`'s `Base.tail` recursion shrinks a concrete tuple
    # type at each step instead of recursing on the same abstract type forever.
    comps = ()
    fixed_terms = ()
    # `fc.rhs` is the concrete `MatrixTerm` after `apply_schema`; its own `.terms`
    # tuple preserves `InteractionTerm`s intact. `StatsModels.terms` instead
    # *decomposes* `InteractionTerm`s into their constituent variables, which
    # silently dropped interactions from the design matrix (v0 bug).
    for t in fc.rhs.terms
        if t isa StatsModels.InterceptTerm{true}
            comps = (comps..., Intercept(n))
        elseif t isa GroupTerm
            col = Tables.getcolumn(cols, t.group)
            levels = unique(col)
            check_unique_labels(
                Symbol.(sanitize_level.(string.(levels))),
                string.(levels),
                "level (grouping variable :$(t.group))",
            )
            idx = [findfirst(==(v), levels) for v in col]
            comps = (comps..., RandomIntercept(t.group, idx, collect(levels)))
        elseif t isa StatsModels.InterceptTerm{false}
            # explicit 0: no intercept component
        else
            fixed_terms = (fixed_terms..., t)
        end
    end
    if !isempty(fixed_terms)
        mt = StatsModels.collect_matrix_terms(StatsModels.TupleTerm(fixed_terms))
        X = Matrix{Float64}(StatsModels.modelcols(mt, cols))
        raw = StatsModels.coefnames(mt)
        raw isa Vector || (raw = [raw])
        names = sanitize.(raw)
        check_unique_labels(names, raw, "coefficient")
        comps = (comps..., FixedEffects(X, names, mt))
    end
    isempty(comps) && throw(
        ArgumentError(
            "model has no predictor terms; formula `$(f)` has no intercept, fixed, or " *
                "random effect to build a model from"
        )
    )
    return comps, collect(y), sch
end
