
function create_surface_interpolant(surface, height_extremas)
    l_axis = range(height_extremas..., size(surface,1))
    θ_axis = range(1, 360, size(surface, 2))
    interp = interpolate(surface, (OnCell() |> Interpolations.Flat |> Quadratic |> BSpline,
                                    OnCell() |> Periodic |> Quadratic |> BSpline))
    interp = Interpolations.scale(interp, l_axis, θ_axis)
    interp = Interpolations.extrapolate(interp, (Interpolations.Flat(), Periodic()))
    return interp
end

function reverse_transform(obj, shift)
    obj = copy(obj)
    obj[1, :] .= rad2deg.((obj[1, :] .- shift)./max.(obj[3,:], 1)) .+ shift

    obj = from_cylindric(obj)
    
    return obj
end
reverse_transform(shift) = obj -> reverse_transform(obj, shift)

heightmap_transform(l_axis) = function heightmap_transform(obj)
    real_height = l_axis[end] - l_axis[1]
    scale_coord(l) = length(l_axis) * (l - l_axis[1])/real_height
    obj = copy(obj)
    obj[2, :] .= scale_coord.(obj[2, :])
    return obj
end

function fit_all_knots(
    all_obj_points, 
    all_obj_labels, 
    surface_interp,
    real_height; 
    all_params_init = nothing,
    interactive_finetune=false, 
    use_real_distance=true, knots_map=surface_interp.itp.itp,
    min_number_points=300)
    
    sign_multiplier = 1
    sign_sums = 0
    for i ∈ sort(unique(all_obj_labels))
        local obj = all_obj_points[:, all_obj_labels .== i]
        if size(obj, 2) < min_number_points
            continue
        end
        minind = argmin(@view obj[3, :])
        maxind = argmax(@view obj[3, :])
        sign_sums += sign(obj[2, maxind] - obj[2, minind])
    end

    if sign_sums < 0
        sign_multiplier = -1
    end

    all_dists = []

    all_params = []
    f = Figure(fontsize=24, figure_padding = 50)
    ax = Axis3(f[1,1], aspect=:data)
    counter = 0
    @showprogress desc="Fitting knots..." for i ∈ sort(unique(all_obj_labels))

        local obj = all_obj_points[:, all_obj_labels .== i] |> copy
        if sign_multiplier < 0
            obj[2, :] .= real_height .- obj[2, :]
        end 

        if val_range(obj[1, :]) > 180
            obj[1, obj[1, :] .> 180] .-= 359
        end
        shift = mean(obj[1, :])
        
        if size(obj, 2) < min_number_points
            continue
        end
        counter += 1
        if isnothing(all_params_init)
            model_params, res1 = fit_knot_model(subsample_knot(obj), surface_interp, show_trace=false, use_real_distance=use_real_distance)
        else
            model_params = all_params_init[counter]
        end

        if interactive_finetune
            model_params = interactive_fit(obj, model_params, surface_interp; use_real_distance=use_real_distance)
        end

        plot_fitted_model!(ax, model_params, obj; 
            max_ρ=model_params.center.X_max,
            coordinate_transform=reverse_transform(shift),
            plot_centerline=false, connect_to_centerline=false)

        scatter!(ax, reverse_transform(shift)(obj), color=:blue, transparency=true)
        
        push!(all_dists, model_distances(obj, model_params; use_real_distance=use_real_distance))
        push!(all_params, model_params)
    end
    
    display(f)

    return all_params, all_dists, sign_multiplier
end


function fit_knots(log_data, knot_results=nothing; knot_gt=nothing, interactive=false)
    knot_points = log_data.knot_points
    knot_labels = log_data.knot_labels
    if !isnothing(knot_results) 
        knot_results = first(knot_results)
    end
    # multiplier = mean(heightmap)
    shifted_range = 1 .+ log_data.height_extremas .- log_data.height_extremas[1]
    if knot_gt |> isnothing
        knot_gt = log_data.knots
    end
    interp = create_surface_interpolant(log_data.heightmap .+ 15 .* knot_gt, shifted_range)
    
    all_params, all_dists, sign_multiplier = fit_all_knots(
        knot_points, 
        knot_labels, 
        interp,
        1 + log_data.height_extremas[2] - log_data.height_extremas[1]; 
        all_params_init = knot_results, knots_map = knot_gt,
        interactive_finetune=interactive)
    heightmap = log_data.heightmap
    if sign_multiplier < 0
        heightmap = reverse(log_data.heightmap, dims=1)
    end
    return (knots=all_params, dists=all_dists, direction=sign_multiplier, heightmap=heightmap)
end



function process_cluster_θs(cluster_mat)
    degθs = mod.(rad2deg.(cluster_mat[1, :]), 360)

    sort_i = sortperm(degθs)

    sorted_degθs = degθs[sort_i]

    n = length(degθs)
    equal_part = 360 / n
    equal_parts = sorted_degθs[1] .+ (0:n-1) .* equal_part

    radial_distance(x1, x2) = let d = abs(x1 - x2); d > 180 ? 180 - d : d; end
    res = radial_distance.(equal_parts, sorted_degθs)[invperm(sort_i)]
    return (res .- mean(res)) .|> deg2rad
end

function update_cluster_θs!(cluster_mat)
    cluster_mat[1, :] .= process_cluster_θs(cluster_mat)
    return cluster_mat
end

function knot_params_to_arrays(log_params; cluster_radius=80)
    params_mat = log_params .|> vec |> stack

    cluster_objs = find_knot_clusters(log_params; radius=cluster_radius)
    
    gaps = map(cluster_objs) do objs
        mean(getfield.(objs, :l0))
    end |> sort |> diff .|> abs

    params_mats = map(cluster_objs) do objs
        objs .|> vec |> stack
    end

    for params_mat ∈ params_mats
        params_mat[3, :] .-= mean(params_mat[3, :])
    end
    number_of_knots = size.(params_mats, 2)

    return params_mats, gaps, number_of_knots
end



function get_all_generation_parameters(knot_results)
    function get_params(knot_params)
        params_mats, gaps, number_of_knots = knot_params_to_arrays(knot_params.knots)
        update_cluster_θs!.(params_mats)
        params_mats = map(x->x[[1:3; 5:7; 9:11], :], params_mats)
        return params_mats, gaps, number_of_knots
    end
    
    params = get_params.(knot_results)

    params_mats, gaps, number_of_knots = zip(params...) |> collect

    all_knots_param_mats = reduce(hcat, reduce.(hcat, params_mats))
    min_knot_size = minimum(all_knots_param_mats[end-2, :])
    
    normcor = cor(all_knots_param_mats; dims=2)

    distribution = compound_fit(DiagNormal, params_mats)
    distribution_n_knots = compound_fit(Normal, number_of_knots)
    
    distribution_gaps = Distributions.fit(Normal, reduce(vcat, gaps))

    return (distribution, distribution_n_knots, distribution_gaps), normcor, min_knot_size
end

cycle_clamp(x, min_val, max_val) = min_val + mod(x - min_val, max_val - min_val)

function ensure_noninterlapping_knots()
    rad_angles_both = angles_from_ratios.(params[end-3, :], 
        params[end-2, :], 
        params[end-1, :], 
        params[end, :]; angles_bounds=get_angles_bounds(RadiusShape))

    first_angles = rad_angles_both .|> first
    rad_angles = deg2rad.(first_angles ./ params[2, :])
    while sum(rad_angles) + (2π/180)*length(rad_angles) > π
        ind = argmax(rad_angles)
        deleteat!(rad_angles, ind)
        params = params[:, [1:ind-1; ind+1:end]]
        θs_shifts = @view params[1, :]
    end
    n = length(rad_angles)
    min_diffs = (rad_angles .+ rad_angles[[2:end; 1]]) .+ 2π/180

    equal_part = 2π / n
    diffs = max.(min_diffs, θs_shifts .+ equal_part)

    ratio = 2π / sum(diffs)
    if ratio < 1
        diffs *= ratio
    end
    
    θs = cumsum(diffs)

    θs = cycle_clamp.((θs .+ 2π * rand()), 0, 2π)
end


absmin(a, b) = sign(a) * min(abs(a), abs(b))

function draw_sector!(ax, angle, radius)
    point0 = Point2f(0, 0)
    point1 = Point2f(sind(-angle-radius), cosd(-angle-radius))
    point2 = Point2f(sind(-angle+radius), cosd(-angle+radius))
    poly!(ax, [point0, point1, point2],)
end

function generate_cluster(
    surface_interp,
    l,
    distribution, 
    distribution_n_knots,
    corr_matrix, min_knot_size)
    
    n = clamp(round(Integer, rand(distribution_n_knots)), 3, 7)

    final_dist = MvNormal(mean(distribution), .√(var(distribution)*var(distribution)') .* corr_matrix)

    params = rand(final_dist, n)
    equal_part = 360 / n
    θs_shifts = absmin.(rad2deg.(params[1, :]), equal_part/4)
    diffs = θs_shifts .+ equal_part
    θs = cumsum(diffs)
    θs = cycle_clamp.((θs .+ 360 * rand()), 0, 360)
    params[1, :] .= θs
    params[end-2, :] .= max.(min_knot_size, params[end-2, :])

    params = KnotModel.(eachcol(params), Ref(surface_interp)) 
    multipliers = differentiate.(0, getfield.(params, :center))
    params = params.|> vec |> stack
    params = clamp.(params, get_lower_bounds(), get_upper_bounds())

    rad_angles_both = angles_from_ratios.(params[end-3, :], 
        params[end-2, :], 
        params[end-1, :], 
        params[end, :]; angles_bounds=get_angles_bounds(RadiusShape))
    first_angles = rad_angles_both .|> first 
    rad_angles = multipliers .* (first_angles ./ params[2, :])
    while sum(rad_angles) + (2π/180)*length(rad_angles) > π
        ind = argmax(rad_angles)
        deleteat!(rad_angles, ind)
        deleteat!(θs_shifts, ind)
        params = params[:, [1:ind-1; ind+1:end]]
    end

    n = size(params, 2)
    equal_part = 360 / n
    i = 1
    while i ≤ length(θs_shifts) && n > 1
        rad_angles_sum = rad2deg.(rad_angles[i] .+ rad_angles[[2:end; 1]][i])
        if equal_part .- rad_angles_sum .- θs_shifts[i] .+ θs_shifts[[2:end; 1]][i] < 0
            deleteat!(rad_angles, i)
            deleteat!(θs_shifts, i)
            params = params[:, [1:i-1; i+1:end]]
            n = size(params, 2)
            equal_part = 360 / n
        else
            i += 1
        end
    end
    rad_angles_sums = rad2deg.(rad_angles .+ rad_angles[[2:end; 1]])
    diffs = θs_shifts .+ equal_part
    
    θs = cumsum(diffs)

    θs = cycle_clamp.((θs .+ 360 * rand()), 0, 360)


    params[1, :] .= θs

    params[3, :] .+= l

    params = clamp.(params, get_lower_bounds(), get_upper_bounds())

    return KnotModel.(eachcol(params[[1:3; 5:7; 9:11], :]), Ref(surface_interp))
end

function generate_clusters(
    surface_interp,
    n,
    start_l,
    distribution_knots,
    distribution_n_knots, 
    distribution_gaps,
    normcov, min_knot_size)

    generate_l(last_l, _) = last_l + rand(distribution_gaps)
    ls = []

    while length(ls) < 1
        gaps = rand(distribution_gaps, n)
        ls = cumsum(gaps) .+ start_l
        lrange, θrange = surface_interp.itp.ranges
        max_l = last(lrange)
        ls = ls[ls .< max_l]
    end
    dist = sample(distribution_knots, 1)[1]
    return [generate_cluster(surface_interp, l, 
                            sample(dist, 1)[1],
                            distribution_n_knots, 
                            normcov, min_knot_size) for l ∈ ls]
end

