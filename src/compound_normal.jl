
struct CompoundNormal{T, E, F, S} <: Sampleable{F, S} where T <: Sampleable{F, S}
    μ::E
    Σ::E
end

AbstractTrees.children(dn::CompoundNormal) = (dn.μ, dn.Σ)
AbstractTrees.nodevalue(::CompoundNormal) = :CompoundNormal
Statistics.mean(dn::CompoundNormal) = dn.μ
Statistics.var(dn::CompoundNormal) = dn.Σ
Base.show(io::IO, dn::CompoundNormal) = print_tree(io, dn)

distribution(::CompoundNormal{T, E, F, S}) where {T, E, F, S} = return T

CompoundNormal(::Type{T}, μ::E, Σ::E) where {F, S, T <: Sampleable{F, S}, E} = CompoundNormal{T, E, F, S}(μ, Σ)

function get_deepest_sample(rng::AbstractRNG, dn::CompoundNormal{T, E, F, S}, n) where {T, E, F, S}
    iterate_col(col::Matrix) = col |> eachcol
    iterate_col(x) = x
    μ = get_deepest_sample(rng, dn.μ, n) |> iterate_col
    Σ = get_deepest_sample(rng, dn.Σ, n) |> iterate_col
    
    return distribution_from_mean_var.(T, μ, Σ)
end
get_deepest_sample(rng::AbstractRNG, dist, n) = rand(rng, dist, n)

function sample(rng::AbstractRNG, dn::CompoundNormal{T, E, F, S}, n, ns...) where {T, E, F, S}
    result = get_deepest_sample(rng, dn, n)
    return sample.(rng, result, ns...)
end
sample(rng::AbstractRNG, dist, n, ns...) = rand(rng, dist, n)
sample(_::AbstractRNG, dist) = dist

sample(args...) = sample(Random.default_rng(), args...)

function compound_fit_to_params(distribution_type, distributions)
    if eltype(distributions) <: distribution_type
        compound_μs = distributions .|> mean |> stack

        μ_dist = Distributions.fit(distribution_type, compound_μs)

        compound_vars = distributions .|> var |> stack

        var_dist = Distributions.fit(distribution_type, compound_vars)
        return CompoundNormal(distribution_type, μ_dist, var_dist)
    else
        return CompoundNormal(distribution_type, 
            compound_fit_to_params(distribution_type, mean.(distributions)), 
            compound_fit_to_params(distribution_type, var.(distributions)))
    end
end

function compound_fit(distribution_type, data)
    eltype(data) <: Real && return Distributions.fit(distribution_type, data)
    distributions = compound_fit.(distribution_type, data)
    return compound_fit_to_params(distribution_type, distributions)
end

distribution_from_mean_var(distribution_type::Type{T}, μ::E, Σ::E) where 
    {F, S, T <: Sampleable{F, S}, E<: Sampleable{F, S}} = CompoundNormal(distribution_type, μ, Σ)
distribution_from_mean_var(::Type{DiagNormal}, μ::T, Σ::T) where T <: AbstractVector = DiagNormal(μ, Distributions.PDMats.PDiagMat(abs.(Σ)))
distribution_from_mean_var(::Type{MvNormal}, μ::T, Σ::T) where T <: AbstractVector = MvNormal(μ, reshape(Σ, length(μ), :))
distribution_from_mean_var(::Type{Normal}, μ, Σ) = Normal(μ, √(abs(Σ)))
distribution_from_mean_var(distribution_type, μ, Σ) = distribution_type(μ, Σ)


function rand(rng::AbstractRNG, dn::CompoundNormal{T, E, F, S}) where {T, E, F, S}
    return rand(rng, distribution_from_mean_var(T, rand(rng, dn.μ), rand(rng, dn.Σ)))
end
Base.length(s::CompoundNormal) = length(s.μ) # return the length of each sample

function Distributions._rand!(rng::AbstractRNG, dn::CompoundNormal{T, E, F, S}, x::AbstractVector{T2}) where {T, E, F, S, T2<:Real}
    return Distributions._rand!(rng, distribution_from_mean_var(T, rand(rng, dn.μ), rand(rng, dn.Σ)), x)
end

