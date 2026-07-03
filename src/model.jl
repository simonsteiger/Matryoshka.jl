struct Recipe{L <: Likelihood, P <: Priors, C <: Tuple, S}
    lik::L
    pri::P
    components::C
    schema::S
    response::Symbol
end

function resolve_priors(pri::Priors, components::Tuple, fam::Family)
    comp_priors = map(components) do c
        slots = priorslots(c)
        dists = map(slots) do (exact, class, default)
            d = lookup(pri, exact, class)
            something(d, default)
        end
        c isa FixedEffects ? collect(dists) : only(dists)
    end
    # `pairs(nt::NamedTuple)` is an `AbstractDict`, and `map` refuses to iterate
    # dictionaries (order is unspecified); `collect` it into a Vector{Pair} first.
    # The do-block also only yields the resolved distribution (not a Pair), so the
    # keys of `default_priors(fam)` are reattached explicitly via `NamedTuple{keys}`.
    dp = default_priors(fam)
    fam_vals = map(collect(pairs(dp))) do (p, default)
        d = lookup(pri, (p,), (p,))
        something(d, default)
    end
    fam_priors = NamedTuple{keys(dp)}(Tuple(fam_vals))
    # unknown-target detection: every user spec must have matched something
    valid = valid_targets(components, fam)
    unmatched = [s.path for s in pri.specs if !(s.path in valid)]
    return comp_priors, fam_priors, unmatched
end

function valid_targets(components::Tuple, fam::Family)
    v = Set{Tuple{Vararg{Symbol}}}()
    for c in components, (exact, class, _) in priorslots(c)
        push!(v, exact, class)
    end
    for p in parameters(fam)[2:end]
        push!(v, (p,))
    end
    return v
end

@model function sum_contribs(ms::Tuple)
    c ~ to_submodel(first(ms), false)
    if length(ms) == 1
        return c
    end
    rest ~ to_submodel(sum_contribs(Base.tail(ms)), false)
    return c .+ rest
end

@model function core_model(recipe, submodels, ilink, obs, fam_priors, y)
    η ~ to_submodel(sum_contribs(submodels), false)
    μ = ilink.(η)
    # obsmodels receive the link-scale linear predictor η and apply their own
    # inverse link internally (e.g. BernoulliLogit) — numerically safer than
    # constructing distributions from a mean that can saturate in Float64.
    v ~ to_submodel(obs(η, fam_priors, y), false)
    return (; η, μ)
end

maybe_prefix(m, ::Nothing) = m
maybe_prefix(m, s::Symbol) = DynamicPPL.prefix(m, s)   # spike Q1: bare Symbol, no Val() needed

function model(lik::Likelihood, pri::Priors, tbl)
    components, y, sch = lower(lik, tbl)
    comp_priors, fam_priors, unmatched = resolve_priors(pri, components, lik.family)
    if !isempty(unmatched)
        valid = sort!(collect(valid_targets(components, lik.family)); by = string)
        throw(
            ArgumentError(
                "unknown prior target(s): $(join(map(p -> join(p, "."), unmatched), ", ")). " *
                    "Valid targets for this model: $(join(map(p -> join(p, "."), valid), ", "))"
            )
        )
    end
    submodels = map((c, p) -> maybe_prefix(submodel(c, p), compprefix(c)), components, comp_priors)
    recipe = Recipe(lik, pri, components, sch, StatsModels.termvars(response_formula(lik).lhs)[1])
    return core_model(
        recipe, submodels, predictor_invlink(lik.family),
        obsmodel(lik.family), fam_priors, y,
    )
end

function model(m::DynamicPPL.Model, newdata)
    r = m.args.recipe
    cols = Tables.columntable(newdata)
    # only the predictor (RHS) variables are required; the response may be absent
    # (predict-mode: y becomes `missing` below) — unlike `lower`, which also
    # requires the response since it is building the training model.
    check_columns(StatsModels.termvars(response_formula(r.lik).rhs), cols)
    components = map(c -> rebuild(c, cols), r.components)
    comp_priors, fam_priors, _ = resolve_priors(r.pri, components, r.lik.family)
    submodels = map((c, p) -> maybe_prefix(submodel(c, p), compprefix(c)), components, comp_priors)
    y = haskey(cols, r.response) ? collect(Tables.getcolumn(cols, r.response)) : missing
    recipe = Recipe(r.lik, r.pri, components, r.schema, r.response)
    return core_model(
        recipe, submodels, predictor_invlink(r.lik.family),
        obsmodel(r.lik.family), fam_priors, y,
    )
end

function default_priors(lik::Likelihood, tbl)
    components, _, _ = lower(lik, tbl)
    rows = @NamedTuple{target::String, class::String, prior::Distribution}[]
    for c in components, (exact, class, default) in priorslots(c)
        push!(rows, (target = join(exact, "."), class = join(class, "."), prior = default))
    end
    for (p, d) in pairs(default_priors(lik.family))
        push!(rows, (target = String(p), class = String(p), prior = d))
    end
    return rows
end
