"""
    Intercept(n::Int) <: AbstractComponent

Population-level intercept component. Contributes a single sampled
`intercept` parameter, broadcast across all `n` observations.

# Example
```julia
using Matryoshka
using Matryoshka: Intercept

c = Intercept(4)
c.n == 4
```
"""
struct Intercept <: AbstractComponent
    n::Int
end

compprefix(::Intercept) = nothing
priorslots(::Intercept) = [((:intercept,), (:intercept,), 2.5 * TDist(3))]
rebuild(::Intercept, tbl) = Intercept(length(Tables.rows(tbl)))

@model function intercept_submodel(prior, n)
    intercept ~ prior
    return fill(intercept, n)
end
submodel(c::Intercept, prior) = intercept_submodel(prior, c.n)
