module SyntheticLogs

export KnotModel, KnotShape, RadiusShape, CenterShape, WeibullModel, get_all_distributions, generate_log

using DocStringExtensions

using Distributions
using Random
using Statistics
import Base.rand, Base.length, Base.vec
using LinearAlgebra
using AbstractTrees


using Roots
using Optim
using LeastSquaresOptim
using PolynomialRoots
using Rotations
using LsqFit
using StaticArrays
using Clustering
using Polynomials
using Interpolations
using ProgressMeter
using Requires
using ApproxFun
using Images
using OffsetArrays



using FFTW

include("compound_normal.jl")
include("centerline_generation.jl")
include("knot_model.jl")
include("knot_generation.jl")
include("surface_generation.jl")
include("fit_distributions.jl")
include("generation_functions.jl")

using SmoothingSplines, Statistics
using SparseArrays, LinearAlgebra, IterativeSolvers, FillArrays, LazyArrays, ImageFiltering, FFTW


include("centerline_estimation.jl")
# include("heightmap_generation.jl")
include("logcentric_coordinates.jl")
# include("point_cloud_filtering.jl")

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
function __init__()
    @require GLMakie="e9467ef8-e4e7-5192-8a1a-b1aee30e663a" include("../ext/glmakie_plotting.jl")
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/cuda_farthest_sampling.jl")
    @require GLMakie="e9467ef8-e4e7-5192-8a1a-b1aee30e663a" include("../point_cloud_filtering.jl")
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("../heightmap_generation.jl")
end
end

end
