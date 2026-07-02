struct FixedEffects{T} <: AbstractComponent
    X::Matrix{Float64}
    names::Vector{Symbol}
    term::T          # concrete StatsModels term for rebuild; `nothing` in unit tests
end

compprefix(::FixedEffects) = nothing
priorslots(c::FixedEffects) =
    [((:b, n), (:b,), Normal(0, 2.5)) for n in c.names]

# spike Q3 verdict (test/spike_notes.md): dict-style `b[n] ~ p` fails in DynamicPPL;
# arraydist fallback yields a single vector-valued VarName `b` (not `b[1]`/`b[2]`).
@model function fe_submodel(X, priors)
    b ~ arraydist(priors)
    return X * b
end
submodel(c::FixedEffects, priors::Vector) = fe_submodel(c.X, priors)

function rebuild(c::FixedEffects, tbl)
    X = StatsModels.modelcols(c.term, Tables.columntable(tbl))
    return FixedEffects(Matrix{Float64}(X), c.names, c.term)
end
