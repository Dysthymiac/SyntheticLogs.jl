
function get_incline_distribs(surface_fits)
    inclines_bumps = getfield.(surface_fits, :incline)

    inclines = stack(first.(inclines_bumps))
    bumps = reduce(hcat, last.(inclines_bumps))
    bumps = bumps[:, abs.(bumps[2, :]) .> 0]
    bumps[3, :] .= abs.(bumps[3, :])

    incline_distrib = Distributions.fit(MvNormal, inclines)
    bumps_distrib = Distributions.fit(MvNormal, bumps)
    minvals = minimum(bumps, dims=2)
    return incline_distrib, bumps_distrib, minvals
end

function get_surface_distrib(surface_fits)
    all_coeffs = getfield.(surface_fits, :coeffs)
    all_coeffs = all_coeffs .|> vec |> stack
    return Distributions.fit(DiagNormal, all_coeffs)
end

function get_random_surface(surface_dist, sz=(450, 360); y_degrees=10, x_degrees=10)
    return reconstruct_from_chebyshev(reshape(rand(surface_dist), y_degrees, x_degrees), sz)
end

function get_all_distributions(all_data, knot_results, surface_fits)
    knot_generation_params, knot_cor, min_knot_size = get_all_generation_parameters(knot_results)
    incline_distrib, bumps_distrib, minvals = get_incline_distribs(surface_fits)
    surface_dist = get_surface_distrib(surface_fits)
    centerline_dist = fit_centerline_dist(all_data)
    all_surface_knots = reduce(hcat, stack.(getindex.(surface_fits, :knot_results)))[3:end, :]
    all_surface_knots_distrib = Distributions.fit(MvNormal, all_surface_knots)

    return (knot_generation_params, knot_cor, min_knot_size, incline_distrib, bumps_distrib, minvals,
        surface_dist, centerline_dist, all_surface_knots, all_surface_knots_distrib)
end

