struct RandomIntercept <: AbstractComponent
    group::Symbol
    idx::Vector{Int}
    levels::Vector{Any}
end

compprefix(c::RandomIntercept) = c.group
priorslots(c::RandomIntercept) = [((c.group, :sd), (:sd,), Exponential(1))]

@model function ri_submodel(idx, nlevels, sd_prior)
    sd ~ sd_prior
    z ~ filldist(Normal(), nlevels)
    return (sd .* z)[idx]
end
submodel(c::RandomIntercept, sd_prior) = ri_submodel(c.idx, length(c.levels), sd_prior)

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
