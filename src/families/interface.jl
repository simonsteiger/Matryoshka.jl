abstract type Family end

function parameters end
function links end
function default_priors end
function obsmodel end

invlink(::Val{:identity}) = identity
invlink(::Val{:log}) = exp
invlink(::Val{:logit}) = logistic
logistic(x) = inv(one(x) + exp(-x))

to_family(f::Family) = f
predictor_param(f::Family) = first(parameters(f))
predictor_invlink(f::Family) = invlink(Val(links(f)[predictor_param(f)]))
