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
