
function generate_log(cluster_num, real_height, 
    knot_generation_params, knot_cor, min_knot_size, incline_distrib, bumps_distrib, minvals,
    surface_dist, centerline_dist, all_surface_knots, all_surface_knots_distrib)
        
    incline_val = rand(incline_distrib)
    bumps = max.(rand(bumps_distrib, cluster_num), minvals)
    surface = get_random_surface(surface_dist)
    # surface .-= range(-3, 0, length=size(surface, 1))
    height_extremas = (0, real_height)

    rec_noise = add_noise_to_rec(surface, create_fractal_noise(size(surface)))

    rec_height = restore_heightmap(2rec_noise, height_extremas, incline_val)

    clusters = generate_clusters(create_surface_interpolant(rec_height, height_extremas),
        cluster_num,
        0,
        knot_generation_params...,
        knot_cor, min_knot_size)

    knots = reduce(vcat, clusters)

    full_rec = restore_heightmap(rec_height, height_extremas, (0, 0), bumps, clusters)
    surface_knots = generate_surface_knots(all_surface_knots_distrib, knots, full_rec, height_extremas)
    full_rec = add_surface_knots(full_rec, surface_knots...)

    # full_rec = add_surface_knots(full_rec, surface_knots...)
    
    centerline = evaluate_centerline(rand(centerline_dist), 100, real_height)
    gt_image = add_surface_ground_truth(zeros(size(full_rec)), surface_knots[2])

    points, ρs, labels = heightmap_from_logcentric_points(full_rec, centerline, gt_image; n_points=200000, height_extremas=(0, real_height))

    return knots, centerline, full_rec, gt_image, points, ρs, labels
end