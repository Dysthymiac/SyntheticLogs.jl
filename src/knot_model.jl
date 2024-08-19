
abstract type KnotShape end
abstract type CenterShape <: KnotShape end
abstract type RadiusShape <: KnotShape end


"""

"""
struct WeibullModel{Shape, T}
    X_max::T
    Y_max::T
    start_incline::T
    end_incline::T

    WeibullModel(shape::Type{<:KnotShape}, X_max::T1, Y_max::T2, start_inclide::T3, end_incline::T4) where {T1, T2, T3, T4} = new{shape, promote_type(T1, T2, T3, T4)}(X_max, Y_max, start_inclide, end_incline)
end


struct KnotModel{T1, T2, T3, T4, T5}
    θ0::T1
    θ_mult::T2
    l0::T3
    center::WeibullModel{CenterShape, T4}
    radius::WeibullModel{RadiusShape, T5}
end


get_lower_bounds() = [-360.0, 1,    -3000,  -500,    -500,       1e-4,           1e-4,       -500,        1,            1e-4,          1e-4]
get_upper_bounds() = [ 360.0, 2,     3000,   500,     500,     1-1e-2,           1-1e-2,      500,       50,            1-1e-2,     1-1e-2]

function get_lower_bounds(θ0, l0)
    lower_bounds = get_lower_bounds()
    lower_bounds[1] = θ0-50
    lower_bounds[3] = l0-50
    return lower_bounds
end
function get_upper_bounds(θ0, l0)
    upper_bounds = get_upper_bounds()
    upper_bounds[1] = θ0+50
    upper_bounds[3] = l0+50
    return upper_bounds
end

get_lower_bounds(model::KnotModel) = get_lower_bounds(model.θ0, model.l0)
get_upper_bounds(model::KnotModel) = get_upper_bounds(model.θ0, model.l0)

get_lower_bounds(x) = get_lower_bounds(x[1], x[3])
get_upper_bounds(x) = get_upper_bounds(x[1], x[3])

Base.vec(x::WeibullModel) = [x.X_max, x.Y_max, x.start_incline, x.end_incline] #getfield.(Ref(x), fieldnames(WeibullModel)) |> collect
WeibullModel(shape::Type{<:KnotShape}, params) = WeibullModel(shape, params...)

vec(x::KnotModel) = vcat([x.θ0, x.θ_mult, x.l0], vec(x.center), vec(x.radius))
optimvec(x::KnotModel) = vcat([x.θ0, x.θ_mult, x.l0], optimvec(x.center), optimvec(x.radius))
optimvec(x::WeibullModel) = [x.Y_max, x.start_incline, x.end_incline]
KnotModel(params) = @views KnotModel(params[1:3]..., WeibullModel(CenterShape, params[4:7]), WeibullModel(RadiusShape, params[8:11]))
function intersect_with_heightmap(fun, interp, θ, l)
    dist(x) = abs(x .- interp(abs(l+fun(x)), θ))
    x = optimize(dist, 0.1, maximum(interp))
    return Optim.minimizer(x)
end
var_x_weibull(shape::Type{<:KnotShape}, rest) = x -> (WeibullModel(shape, x, rest...))(x)
function KnotModel(params, heightmap_interp; use_real_distance=true)
    surface_intersection_x = intersect_with_heightmap(var_x_weibull(CenterShape, params[4:6]), heightmap_interp, params[1], params[3])
    if use_real_distance
        surface_intersection_rad = curve_length_local(surface_intersection_x, WeibullModel(CenterShape, surface_intersection_x, params[4:6]...))
    else
        surface_intersection_rad = surface_intersection_x
    end
    params = [params[1:3]..., surface_intersection_x, params[4:6]..., surface_intersection_rad, params[7:9]...]
    return KnotModel(params)
end

get_start_angle_bounds(::Type{CenterShape}) = (-(π/2 - π/32), π/2 - π/32)
get_end_angle_bounds(::Type{CenterShape}) = (-π/6, π/6)


get_start_angle_bounds(::Type{RadiusShape}) = (0, π/2 - π/32)
get_end_angle_bounds(::Type{RadiusShape}) = (0, π/8)

get_angles_bounds(type) = (get_start_angle_bounds(type), get_end_angle_bounds(type))

function to_general_model(X_max, Y_max, start_incline, end_incline; angles_bounds=get_angles_bounds(CenterShape))
    start_angle, end_angle = angles_from_ratios(X_max, Y_max, start_incline, end_incline; angles_bounds=angles_bounds)
    d0 = tan(start_angle)
    d1 = tan(end_angle)
    μ = d1
    α = (Y_max) - μ*X_max
    β = (d0-μ) * X_max / max(α, eps())
    return (α, β, μ, X_max)
end

function ratios_from_angles(X_max, Y_max, start_angle, end_angle; angles_bounds=get_angles_bounds(CenterShape)) # 0.4
    (min_start_angle, max_start_angle), (min_end_angle, max_end_angle) = angles_bounds
    max_end_angle = min(max_end_angle, atan((Y_max), X_max))
    min_start_angle = max(min_start_angle, atan(2(Y_max) - tan(end_angle) * X_max, X_max))

    end_incline = (end_angle - min(min_end_angle, max_end_angle))/(max_end_angle - min(min_end_angle, max_end_angle))
    start_incline = (start_angle - min_start_angle)/(max(min_start_angle, max_start_angle) - min_start_angle)

    return start_incline, end_incline
end

function angles_from_ratios(X_max, Y_max, start_ratio, end_ratio; angles_bounds=get_angles_bounds(CenterShape))
    (min_start_angle, max_start_angle), (min_end_angle, max_end_angle) = angles_bounds
   
    max_end_angle = atan((Y_max), X_max)
    end_angle = (1 - end_ratio) * min(min_end_angle, max_end_angle) + end_ratio * max_end_angle

    min_start_angle = max(min_start_angle, atan(2(Y_max) - tan(end_angle) * X_max, X_max))
    start_angle = (1 - start_ratio) * min_start_angle + start_ratio * max(min_start_angle, max_start_angle)
    return start_angle, end_angle
end

get_knot_center_angles(knot::KnotModel) = angles_from_ratios(vec(knot.center)...)
get_knot_radius_angles(knot::KnotModel) = angles_from_ratios(vec(knot.radius)...; angles_bounds=get_angles_bounds(RadiusShape))

to_general_model(w::WeibullModel{shape}; angles_bounds=get_angles_bounds(shape)) where shape<:KnotShape= to_general_model(vec(w)...; angles_bounds=angles_bounds)

weibull_exp(x, α, β, μ, R) = α * (1-exp(-β*x/(R-x))) + μ * x
weibull_linear(x, α, _, μ, _) = α + μ * x
weibull(x, α, β, μ, R) = x < R ? weibull_exp(x, α, β, μ, R) : weibull_linear(x, α, β, μ, R)
(w::WeibullModel)(x) = weibull(x, to_general_model(w)...)


weibull_exp_dx(x, α, β, μ, R) = μ + α * β * R * exp(-x*β / (R-x)) / (R-x)^2
weibull_linear_dx(x, α, β, μ, R) = μ
weibull_dx(x, α, β, μ, R) = x < R ? weibull_exp_dx(x, α, β, μ, R) : weibull_linear_dx(x, α, β, μ, R)
differentiate(x, w::WeibullModel) = weibull_dx(x, to_general_model(w)...)


closest_point_exp(x, xp, yp, α, β, μ, R) = -x + xp + (yp - x*μ - (1 - exp((-x*β) / (R - x)))*α)*(μ - ((-β) / (R - x) + ((-x*β) / ((R - x)^2)))*exp((-x*β) / (R - x))*α)
closest_point_linear(x, xp, yp, α, β, μ, R) = -x + xp + (yp - x*μ - α)*μ
closest_point(x, xp, yp, α, β, μ, R) = x < R ? closest_point_exp(x, xp, yp, α, β, μ, R) : closest_point_linear(x, xp, yp, α, β, μ, R)
closest_point(x, xp, yp, w::WeibullModel) = closest_point(x, xp, yp, to_general_model(w)...)

only_reals(x; ϵ=1e-8) = [real(x) for x ∈ x if abs(imag(x)) < ϵ]

function find_closest_point(p, model::WeibullModel{CenterShape})
    xp = p[2]
    yp = p[1]
    α, β, μ, R = to_general_model(model)
    dddx(x) = closest_point_exp(x, xp, yp, α, β, μ, R) 

    distance(x) = (x - xp)^2 + (model(x) - yp)^2
    

    linear_root = (xp + μ * (yp - α))/(1+μ^2)
    roots = find_zeros(dddx, 0, model.X_max)
    roots = [0; roots; linear_root]
    root = roots[argmin(distance.(roots))]
    
    return model(root), root
end
find_closest_point(p, model::KnotModel) = find_closest_point(p, model.center)

function find_center_points(X, model; use_real_distance=true)
    X = to_local_coords(X, model)
    if use_real_distance
        curve = find_center_points_local(X, model)
    else
        curve_l = model.center.(X[:, 3])
        curve = stack((zeros(length(curve_l)), curve_l, X[:, 3]), dims=1)
    end
    curve = from_local_coords(curve, model)
    return curve
end


function find_center_points_with_lengths(X, model; use_real_distance=true)
    X = to_local_coords(X, model)
    if use_real_distance
        curve = find_center_points_local(X, model)
        lengths = curve_length_local.(curve[3, :], Ref(model))
    else
        curve_l = model.center.(X[:, 3])
        curve = stack((zeros(length(curve_l)), curve_l, X[:, 3]), dims=1)
        lengths = curve[3, :]
    end
    curve = from_local_coords(curve, model)
    return curve, lengths
end

find_center_points_local(X, model::WeibullModel{CenterShape}) = @views [0; find_closest_point.(eachcol(X[2:3, :]), Ref(model)) |> stack]
find_center_points_local(X, model::KnotModel) = find_center_points_local(X, model.center)

function trapezoid_rule_integral_length(a, b, model; steps=100)
    a==b && return zero(a)
    xs = range(a, b, length=steps)
    vals = .√(1 .+ differentiate.(xs, Ref(model)).^2)
    dx = xs[2] - xs[1]
    return 0.5dx * (vals[1] + vals[end] + 2sum(vals[2:end-1]))
end

curve_length_local(ρ, model) = 
    @views trapezoid_rule_integral_length(zero(ρ), ρ, model)

curve_length_local(ρ, model::KnotModel) = curve_length_local(ρ, model.center)

function curve_lengths(X, model)
    X = to_local_coords(X, model)
    return curve_length_local.(X[3, :], Ref(model))
end

to_local_coords(X, model) = ((X .- @SVector([model.θ0, model.l0, 0])))  .* @SVector [model.θ_mult, 1, 1]
from_local_coords(X, model) = ((X ./ @SVector([model.θ_mult, 1, 1]))) .+ @SVector [model.θ0, model.l0, 0] 

LeakyReLU(x, α=0.1) = x ≥ 0 ? x : α * x

function robust_filter(x; mul=3)
    centered = abs.(x .- median(x))
    mad = median(centered)
    return centered .< mul*mad
end

function model_distances_radii(X, model; use_real_distance=true)
    
    X_local = to_local_coords(X, model)
    if use_real_distance
        curve = find_center_points_local(X_local, model)
        dists = norm.(eachcol(curve .- X_local))
        radii = (model.radius).(curve_length_local.(curve[3, :], Ref(model))) .|> abs
    else
        dists = abs.(X_local[2, :] .- model.center.(X_local[3, :]))
        radii = (model.radius).(curve_length_local.(X_local[3, :], Ref(model))) .|> abs
    end
    return dists, radii
end


model_objective(heightmap_interp; use_real_distance=true, end_incline_mult=0.01) = (X, params) -> model_objective(X, params, heightmap_interp; end_incline_mult=end_incline_mult, use_real_distance=use_real_distance)

function model_objective(X, model::KnotModel; use_real_distance=true, end_incline_mult=0.01)
    
    dists, radii = model_distances_radii(X, model; use_real_distance=use_real_distance)
    radii_dists = abs.(dists .- radii) .+ end_incline_mult * (2 - model.center.end_incline - model.radius.end_incline)

    return radii_dists
end

function model_objective(X, params, heightmap_interp; use_real_distance=true, end_incline_mult=0.01)
    model = KnotModel(params, heightmap_interp; use_real_distance=use_real_distance)
    return model_objective(X, model; use_real_distance=use_real_distance, end_incline_mult=end_incline_mult)
end

function model_distances(X, model; use_real_distance=true)
    dists, radii = model_distances_radii(X, model; use_real_distance=use_real_distance)
    
    radii_dists = (dists .- radii)
    return radii_dists
end

function find_max_length(X, model)
    curve_points = find_center_points(X, model)
    lengths = curve_lengths(curve_points, model)
    max_len, max_ind = findmax(lengths)
    return max_len, curve_points[:, max_ind]
end

interleave(x, y; n=size(x,2)) = [x y][:, vec([(1:n)'; (n+1:2n)'])]

function generate_model_mesh(model, obj=nothing; n_segments=100, circle_points=36, max_ρ=nothing, use_real_distance=true)
    dfdx(x) = differentiate(x, model.center)
    if isnothing(max_ρ)
        max_ρ = model.center.X_max
        if !isnothing(obj)
            _, max_point = find_max_length(obj, model)
            max_ρ = max(max_ρ, max_point[3])
        end
    end

    xs = range(0.01, max_ρ, n_segments)
    diffs = dfdx.(xs)
    if use_real_distance
        lengths = curve_length_local.(xs, Ref(model))
        rads = (model.radius).(lengths)
    else
        rads = (model.radius).(xs)
    end
    angles = range(0, 2π, circle_points+1) |> collect
    angles = angles[1:end-1]
    cosines = cos.(angles)
    sines = sin.(angles)
    lin = LinearIndices((1:circle_points,1:n_segments))
    points = []
    faces = []
    cycle_coords(i, max_val) = 1 + (i - 1) % max_val
    get_indices(i, j) = lin[cycle_coords(j, circle_points), i]
    
    for (i, x, diff, rad) ∈ zip(1:n_segments, xs, diffs, rads)
        if use_real_distance
            tangent = normalize(@SVector [0, diff, 1])
            oz = normalize(tangent × @SVector [0, 1, 0])
            oy = normalize(oz × tangent)
        else
            tangent = @SVector [0, 0, 1]
            oz = @SVector [0, 1, 0]
            oy = normalize(oz × tangent)
        end
        wTr = hcat(tangent, oy, oz)
        point = [0; model.center(x); x] .+ wTr * [zeros(circle_points) rad.*cosines rad.*sines]'
        push!(points, from_local_coords(point, model))
        if i < n_segments
            push!(faces, ([get_indices(i, j) get_indices(i+1, j+1) get_indices(i, j+1);
                            get_indices(i, j) get_indices(i+1, j+1) get_indices(i+1, j)]' for j ∈ 1:circle_points)...)
        end
    end
    points = reduce(hcat, points)
    faces = reduce(hcat, faces)
    return points, faces
end

function fit_knot_model(obj, init_model, heightmap_interp; use_real_distance=true, max_iters=100, show_trace=false)
    
    objective = model_objective(heightmap_interp; use_real_distance=use_real_distance)
    lower = get_lower_bounds(init_model)[[1:3; 5:7; 9:end]]
    upper = get_upper_bounds(init_model)[[1:3; 5:7; 9:end]]
    ys = zeros(size(obj, 2))
    res = curve_fit(
        objective, 
        obj, 
        ys, 
        clamp.(init_model |> optimvec, lower, upper);
        lower=lower, upper=upper,
        maxIter=max_iters,
        autodiff=:forwarddiff,
        show_trace=show_trace,
        )
    
    model = KnotModel(res |> coef, heightmap_interp; use_real_distance=use_real_distance)
    return model, res
end


function initial_weibull_fit(obj, heightmap_interp; use_real_distance=true)
    θ, l, ρ = @views (obj[1, :], obj[2, :], obj[3, :])
    med = median(ρ)
    
    start_line = l[ρ[:] .< med]' / [ones(1, sum(ρ[:] .< med)); ρ[ρ[:] .< med]']
    end_line = l[ρ[:] .≥ med]' / [ones(1, sum(ρ[:] .≥ med)); ρ[ρ[:] .≥ med]']


    θ0 = median(θ) 
    l0 = start_line[1]

    X0 = maximum(ρ)
    Y0 = end_line[1] + end_line[2] * X0 - l0

    
    start_angle = abs(atan(start_line[2]))
    end_angle = abs(atan(end_line[2]))
    start_incline, end_incline = ratios_from_angles(Y0, X0, start_angle, end_angle)
    init_center = WeibullModel(CenterShape, X0, Y0, 
        clamp(start_incline, 1e-4, 0.9), 
        clamp(end_incline, 1e-4, 0.9))

    init_radius = WeibullModel(RadiusShape, 1, 1, 1, 1)
    
    init_model = KnotModel(θ0, 1, l0, init_center, init_radius)
    centers = find_center_points(obj, init_model; use_real_distance=use_real_distance)
    diffs = centers .- obj
    
    ratio = mean(abs.(diffs[1, :])) / mean(abs.(diffs[2, :]))
    dists = norm.(eachcol([1, ratio, 1] .* diffs))
    lengths = curve_lengths(centers, init_model)
    med_len = median(lengths)


    start_radius_line = dists[lengths .< med_len]' / lengths[lengths .< med_len]'
    end_radius_line = dists[lengths .≥ med_len]' / [ones(1, sum(lengths[:] .≥ med_len)); lengths[lengths[:] .≥ med_len]']

    if use_real_distance
        radius_X0 = curve_length_local(X0, init_center)
    else
        radius_X0 = X0
    end
    radius_Y0 = end_radius_line[1] + radius_X0 * end_radius_line[2]
    radius_start_angle = atan(start_radius_line)
    radius_end_angle = atan(end_radius_line[2])
    start_incline, end_incline = ratios_from_angles(radius_Y0, radius_X0, radius_start_angle, radius_end_angle)
    
    init_radius = WeibullModel(RadiusShape, radius_X0, radius_Y0, 
        clamp(start_incline, 1e-4, 0.9), 
        clamp(end_incline, 1e-4, 0.9))

    init_model = KnotModel(θ0, ratio, l0, 
        init_center, init_radius)
    
    return init_model
end

function fit_knot_model(obj, surface_interp; use_real_distance=true, max_iters=100, show_trace=false)
    
    init_model = initial_weibull_fit(obj, surface_interp, use_real_distance=use_real_distance)
    
    init_model = KnotModel(clamp.(init_model |> vec, get_lower_bounds(), get_upper_bounds()))

    return fit_knot_model(obj, init_model, surface_interp; use_real_distance=use_real_distance, max_iters=max_iters, show_trace=show_trace)
end

function length_colors(points, model;use_real_distance=true) 
    curve = find_center_points(points, model;use_real_distance=use_real_distance)
    return curve_lengths(curve, model)
end

function apply_bounds(points, (l_bounds, ρ_bounds))
    return points
end

function plot_fitted_model!(ax, model, obj=nothing; coordinate_transform=identity,
                             n_segments=10, circle_points=36, max_ρ=nothing,
                             plot_centerline=true, connect_to_centerline=true, bounds=nothing)
    points, faces = generate_model_mesh(model, obj; 
        n_segments=n_segments, circle_points=circle_points, max_ρ=max_ρ)

    if !isnothing(bounds)
        points = apply_bounds(points, bounds)
    end
    !isnothing(obj) && meshscatter!(ax, coordinate_transform(obj), color=:blue, markersize=2, transparency=false)
    
    mesh!(ax, coordinate_transform(points)', faces', alpha=0.9, color=length_colors(points, model))
    
    if !isnothing(obj)
        if plot_centerline
            curve = find_center_points(obj, model)
            scatter!(ax, coordinate_transform(curve), color=:red)
        end
        if connect_to_centerline
            pairs = interleave(curve, obj)
            linesegments!(ax, coordinate_transform(pairs), color=:magenta)
        end
    end
    return ax
end


farthest_sample_object_inds(obj, fraction=4) = farthest_sampling(size(obj, 2) ÷ fraction, obj |> gpu) |> cpu
farthest_sample_object(obj, fraction=4) = obj[:, farthest_sample_object_inds(obj, fraction)]

function subsample_knot_inds(obj; fraction=4, q=0.9)

    inds = farthest_sample_object_inds(obj, fraction)
    ρs = obj[3, inds]
    quantile_inds = ρs .< quantile(ρs, q)
    return inds[quantile_inds]
end
subsample_knot(obj; fraction=4, q=0.95) = obj[:, subsample_knot_inds(obj; fraction=fraction, q=q)]

