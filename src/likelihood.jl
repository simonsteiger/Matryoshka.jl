struct Likelihood{F <: Family, T <: Tuple}
    family::F
    formulas::T
end

response_formula(lik::Likelihood) = first(lik.formulas)
extra_formulas(lik::Likelihood) = Base.tail(lik.formulas)

function formula_line(ex)
    Meta.isexpr(ex, :call, 3) && ex.args[1] === :~ ||
        error("@likelihood: each line must be `lhs ~ rhs`, got `$(ex)`")
    # Delegate to StatsModels' own parser so (1 | g) etc. parse identically to @formula
    return :(StatsModels.@formula($(ex.args[2]) ~ $(ex.args[3])))
end

macro likelihood(family, ex)
    lines = Meta.isexpr(ex, :block) ? filter(x -> !(x isa LineNumberNode), ex.args) : [ex]
    isempty(lines) && error("@likelihood: needs at least a response formula")
    formulas = map(formula_line, lines)
    return :(Likelihood(to_family($(esc(family))), ($(map(esc, formulas)...),)))
end
