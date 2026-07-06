"""
    RandomIntercept(group::Symbol, idx::Vector{Int}, levels::Vector) <: AbstractComponent

Group-level (random) intercept component for one grouping variable — `(1 | group)`
in formula syntax. Uses a non-centered parameterisation: samples `sd` (from its
prior, default `Exponential(1)`) and `z ~ filldist(Normal(), nlevels)`,
returning `(sd .* z)[idx]`. `idx` maps each observation to its group's position
in `levels`; `levels` are the training-time factor levels, enforced (not
extended) on `rebuild` — a new level in `newdata` raises an `ArgumentError`.

# Example
```julia
using Matryoshka
using Matryoshka: RandomIntercept, compprefix

c = RandomIntercept(:g, [1, 1, 2], ["a", "b"])
compprefix(c) === :g
```
"""
struct RandomIntercept <: AbstractComponent
    group::Symbol
    idx::Vector{Int}
    levels::Vector{Any}
end

compprefix(c::RandomIntercept) = c.group
priorslots(c::RandomIntercept) = [((c.group, :sd), (:sd,), Exponential(1))]

# z draws are labeled DimVectors on a dim named after the grouping variable
# (`(1 | species)` → Dim{:species}), with sanitized level labels — matches
# ArviZ coord conventions (spec: 2026-07-05-labeled-parameters-design.md).
# Raw levels stay in the struct: rebuild matches raw data values; sanitized
# labels are cosmetic and derived here. Dim{c.group} is built from a runtime
# Symbol (type-unstable), but only at model-construction time — never on the
# log-density path.
@model function ri_submodel(idx, sd_prior, group_dim)
    sd ~ sd_prior
    z ~ withdims(filldist(Normal(), length(group_dim)), group_dim)
    # Gathering by `idx` changes the semantic axis from group-space to
    # observation-space (rows repeat/reorder groups), so the group `Dim`
    # label is no longer meaningful past this point — strip it with `collect`
    # before returning, or it silently leaks into `eta` downstream.
    return collect((sd .* z)[idx])
end
function submodel(c::RandomIntercept, sd_prior)
    labels = level_labels(c.levels)
    return ri_submodel(c.idx, sd_prior, Dim{c.group}(labels))
end

function rebuild(c::RandomIntercept, tbl)
    col = Tables.getcolumn(Tables.columntable(tbl), c.group)
    idx = map(col) do v
        i = findfirst(==(v), c.levels)
        i === nothing && throw(
            ArgumentError(
                "new level $(repr(v)) for grouping variable :$(c.group); " *
                    "training levels were $(c.levels). Prediction for new levels is not yet supported."
            )
        )
        i
    end
    return RandomIntercept(c.group, idx, c.levels)
end
