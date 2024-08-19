
isdefined(Base, :get_extension) ? (using GLMakie) : (using ..GLMakie)
isdefined(Base, :get_extension) ? (using GLMakie.Observables: @lift) : (using ..GLMakie.Observables: @lift)

sliders_from_vals(ax, labels, vals, lower, upper) = SliderGrid(ax, 
    ((label = string(label), range = range(l, u, length=720), startvalue = val) 
        for (label, val, l, u) ∈ zip(labels, vals, lower, upper))...,
    tellwidth=false)

sliders_from_weibull(ax, w, lower, upper) = sliders_from_vals(ax, fieldnames(typeof(w))[2:end], vec(w)[2:end], lower, upper)


function interactive_fit(obj, params, surface_interp; use_real_distance=true)

    f = Figure()

    ax = Axis3(f[1,1:3], aspect=:data)

    best_params = Observable(params)
    params_obs = GLMakie.Observables.async_latest(Observable(params))
    pointsfaces = @lift generate_model_mesh($params_obs, obj; use_real_distance=use_real_distance)
    points, faces = ((@lift first($pointsfaces)'), (@lift last($pointsfaces)'))

    obj_original = copy(obj)
    inds = subsample_knot_inds(obj)
    non_inds = trues(size(obj, 2))
    non_inds[inds] .= false
    obj = obj[:, inds]    

    curve_lengths = @lift find_center_points_with_lengths(obj, $params_obs; use_real_distance=use_real_distance)
    curve, lengths = ((@lift first($curve_lengths)), (@lift last($curve_lengths)))

    dists = @lift norm.(eachcol(to_local_coords($curve, $params_obs) .- to_local_coords(obj, $params_obs)))
    rads = @lift ($params_obs).radius.($lengths)

    rad_points = @lift begin
        local_curve = to_local_coords($curve, $params_obs)
        local_obj = to_local_coords(obj, $params_obs)
        lrads = $rads
        
        dirs = local_obj .- local_curve
        lens = norm.(eachcol(dirs))
        dirs = dirs ./ lens'

        return from_local_coords(local_curve .+ lrads' .* dirs, $params_obs)
    end
    rads_dists = @lift abs2.($dists .- $rads)
    
    scatter!(ax, obj, color=rads_dists, colormap=:afmhot)
    linesegments!(ax, @lift(interleave($rad_points, obj)), color=:magenta)

    mesh!(ax, points, faces, alpha=0.9, color=@lift(length_colors(($points)', $params_obs)))

    scatter!(ax, curve, color=lengths, colormap=:magma)
    
    g = GridLayout(f[2, 1:3], tellwidth=false)

    distance_sum_squared(x) = sum(abs2, model_distances(obj, x; use_real_distance=use_real_distance))
    distance_rmse(x) = .√mean(abs2, model_distances(obj, x; use_real_distance=use_real_distance))

    distances = @lift distance_rmse($params_obs)
    best_distances = @lift distance_rmse($best_params)
    dist_label = Label(g[1, 1], @lift("Best residual: $($best_distances),\nCurrent residuals: $($distances)"), justification=:right)
    update_button = Button(g[1, 2], label="Update best params")
    reset_best_button = Button(g[1, 3], label="Reset to best params")
    reset_button = Button(g[1, 4], label="Reset all params")

    fit_button = Button(g[1, 5], label="Auto fit")

    on(update_button.clicks) do _
        best_params[] = params_obs[]
    end

    on(reset_button.clicks) do _
        best_params[] = params
        params_obs[] = params
        update_sliders!(params_obs[])
    end

    on(reset_best_button.clicks) do _
        params_obs[] = best_params[]
        update_sliders!(params_obs[])
    end

    on(fit_button.clicks) do _
        model, _ = fit_knot_model(obj, params_obs[], surface_interp; max_iters=50, show_trace=true)
        params_obs[] = model
        update_sliders!(params_obs[])
    end

    lower_bounds = get_lower_bounds(params)
    upper_bounds = get_upper_bounds(params)

    pos_grid = sliders_from_vals(f[3, 1], 
        ["θ_0", "θ_mult", "l_0"], 
        [params.θ0, params.θ_mult, params.l0], 
        lower_bounds[1:3], 
        upper_bounds[1:3])
    
    center_grid = sliders_from_weibull(f[3, 2], params.center, lower_bounds[5:7], upper_bounds[5:7])
    radius_grid = sliders_from_weibull(f[3, 3], params.radius, lower_bounds[9:11], upper_bounds[9:11])
    
    pos_vals = map(tuple, (slider.value for slider ∈ pos_grid.sliders)...)
    center_vals = map(tuple, (slider.value for slider ∈ center_grid.sliders)...)
    radius_vals = map(tuple, (slider.value for slider ∈ radius_grid.sliders)...)
    
    params_from_sliders = @lift KnotModel([($pos_vals)..., ($center_vals)..., ($radius_vals)...], 
                                            surface_interp; use_real_distance=use_real_distance)
    connection = GLMakie.Observables.connect!(params_obs, params_from_sliders)
    function update_sliders!(params)
        off(connection)
        set_close_to!.(pos_grid.sliders, [params.θ0, params.θ_mult, params.l0])
        set_close_to!.(center_grid.sliders, vec(params.center)[2:end])
        set_close_to!.(radius_grid.sliders, vec(params.radius)[2:end])
        on(connection.f, connection.observable; weak=connection.weak)
    end

    wait(display(f))

    return best_params[]
end