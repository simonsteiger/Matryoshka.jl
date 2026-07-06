"""
    FixedEffects(X::Matrix{Float64}, names::Vector{Symbol}, term) <: AbstractComponent

Population-level slope component for one or more predictor columns. Samples a
single vector-valued coefficient (`b ~ arraydist(priors)` — DynamicPPL does not
support per-name `VarName`s inside a `Dict`, so `X`'s columns map onto `b`
*positionally*, in `names` order; see `test/spike_notes.md` Q3) and returns
`X * b`. `term` is the concrete `StatsModels` term used to recompute `X` from
new data in `rebuild`; it may be `nothing` when `FixedEffects` is constructed
directly (e.g. in unit tests) rather than via `lower`.

# Example
```julia
using Matryoshka, Distributions
using Matryoshka: FixedEffects, submodel

X = [1.0 2.0; 3.0 4.0]
c = FixedEffects(X, [:x1, :x2], nothing)
m = submodel(c, [Normal(0, 1), Normal(0, 5)])
```
"""
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
# withdims labels each draw as a DimVector on the fixed :coef dim, so chains carry
# coefficient names end to end (spec: 2026-07-05-labeled-parameters-design.md).
@model function fe_submodel(X, priors, names)
    b ~ withdims(product_distribution(priors), Dim{:coef}(names))
    # `b` is a Dim{:coef}-labeled DimVector; `X * b` inherits that label via
    # DimensionalData's matrix-vector product. The coefficient axis has no
    # meaning once contracted against `X`, so strip the label with `collect`
    # before returning — same pattern as RandomIntercept's contribution.
    return collect(X * b)
end
submodel(c::FixedEffects, priors::Vector) = fe_submodel(c.X, priors, c.names)

function rebuild(c::FixedEffects, tbl)
    X = StatsModels.modelcols(c.term, Tables.columntable(tbl))
    return FixedEffects(Matrix{Float64}(X), c.names, c.term)
end
