struct PriorSpec
    path::Tuple{Vararg{Symbol}}
    dist::Distribution
end

struct Priors
    specs::Vector{PriorSpec}
end

target_path(s::Symbol) = (s,)
function target_path(ex::Expr)
    Meta.isexpr(ex, :., 2) && ex.args[2] isa QuoteNode ||
        error("@priors: left-hand side must be a name or dotted path, got `$(ex)`")
    return (target_path(ex.args[1])..., ex.args[2].value)
end

function prior_line(ex)
    Meta.isexpr(ex, :call, 3) && ex.args[1] === :~ ||
        error("@priors: each line must be `target ~ Distribution`, got `$(ex)`")
    path = target_path(ex.args[2])
    return :(PriorSpec($(QuoteNode(path)), $(esc(ex.args[3]))))
end

macro priors(ex)
    lines = Meta.isexpr(ex, :block) ? filter(x -> !(x isa LineNumberNode), ex.args) : [ex]
    specs = map(prior_line, lines)
    return :(Priors(PriorSpec[$(specs...)]))
end

function lookup(pri::Priors, exact::Tuple, class::Tuple)
    # two-pass: exact beats class; within a pass, the later spec wins
    for s in reverse(pri.specs)
        s.path == exact && return s.dist
    end
    for s in reverse(pri.specs)
        s.path == class && return s.dist
    end
    return nothing
end
