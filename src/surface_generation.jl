

standardize_range(img; dims=:) = (img .- mean(img,dims=dims)) ./ std(img,dims=dims)

to_fft_image(heightmap) = heightmap |> fft |> fftshift .|> abs .|> x->x+1 .|> log 

to_image(xs...; kws...) = colorview(Gray, mosaicview(xs .|> normalize_range; nrow=1, kws...) |> collect)

function clip_quantiles!(img, low=0.01, high=1-low)
    hqval = quantile(vec(img), high)
    img[img .> hqval] .= hqval
    lqval = quantile(vec(img), low)
    img[img .< lqval] .= lqval
    return img
end

function clip_quantiles(img, low=0.01, high=1-low)
    result = copy(img)
    return clip_quantiles!(result, low, high)
end

is_between_cartesian(l, x, h) = all(Tuple(l) .≤ Tuple(x) .≤ Tuple(h))

function create_mask(σ, sz)
    mask = parent(Kernel.gaussian((σ, σ), sz .+ 1 .- (sz .% 2)))[1:sz[1], 1:sz[2]]
    mask = mask ./ maximum(mask)
    return mask
end

apply_mask(x, mask) = real.(ifft(fft(x) .* ifftshift(mask)))

function separate_fft(heightmap, σ=5)
    mask = create_mask(σ, size(heightmap))
    return apply_mask(heightmap, mask), apply_mask(heightmap, maximum(mask) .- mask)
end

function separate_fft(heightmap, σ1, σ2)
    mask1 = create_mask(σ1, size(heightmap))
    mask2 = create_mask(σ2, size(heightmap))
    mask = abs.(mask2 .- mask1)
    return apply_mask(heightmap, mask), apply_mask(heightmap, maximum(mask) .- mask)
end

## Approximations

function create_fourier_coeffs(x; degrees=1:5)
    S = PeriodicSegment(0..2π)
    V = Array{Float64}(undef, length(degrees), length(x));
    for (i, k) ∈ enumerate(degrees)
        V[i, :] = Fun(S, [zeros(k-1); 1]).(x)
    end
    return V
end

function create_chebyshev_coeffs(x; degrees=1:5)
    S = Chebyshev(0..1)
    V = Array{Float64}(undef, length(degrees), length(x));
    for (i, k) ∈ enumerate(degrees)
        V[i, :] = Fun(S, [zeros(k-1); 1]).(x)
    end
    return V
end

fit_looped_curve(x, y; degrees=1:5) = create_fourier_coeffs(x; degrees=degrees)' \ y
evaluate_curve(coeffs, x; degrees=1:5) = create_fourier_coeffs(x; degrees=degrees)' * coeffs
fit_chebyshev_curve(x, y; degrees=1:5) = create_chebyshev_coeffs(x; degrees=degrees)' \ y
evaluate_chebyshev_curve(coeffs, x; degrees=1:5) = create_chebyshev_coeffs(x; degrees=degrees)' * coeffs


function chebyshev_coeffs_from_image(xs, ys, img; x_degrees = 1:10, y_degrees = 1:10)
    all_coeffs = map(i->fit_looped_curve(xs, img[i, :]; degrees=x_degrees), axes(img, 1)) |> stack
    chebyshev_coeffs = fit_chebyshev_curve.(Ref(ys), eachrow(all_coeffs); degrees=y_degrees) |> stack
    return chebyshev_coeffs
end
chebyshev_coeffs_from_image(img; x_degrees = 1:10, y_degrees = 1:10) = 
    chebyshev_coeffs_from_image(range(0, 2π, length=size(img, 2)),
                                range(0, 1, length=size(img, 1)),
                                img,
                                x_degrees = x_degrees,
                                y_degrees = y_degrees)

function reconstruct_from_chebyshev(coeffs, xs, ys; 
    y_degrees=1:size(coeffs, 1), x_degrees=1:size(coeffs, 2))
    all_coeffs = stack(evaluate_chebyshev_curve.(eachcol(coeffs), Ref(ys); degrees=y_degrees), dims=1)
    return curve_image_from_coeffs(eachcol(all_coeffs), axes(all_coeffs, 2), (length(ys), length(xs)); degrees=x_degrees)
end

reconstruct_from_chebyshev(coeffs, sz; y_degrees=1:size(coeffs, 1), x_degrees=1:size(coeffs, 2)) =
    reconstruct_from_chebyshev(coeffs, range(0, 2π, sz[2]), range(0, 1, sz[1]); 
        y_degrees=y_degrees, x_degrees=x_degrees)

function curve_image_from_coeffs(all_coeffs, inds, sz; degrees=1:5)
    result = zeros(sz)
    xs = range(0, 2π, length=sz[2])
    
    result[inds[1], :] .= evaluate_curve(all_coeffs[1], xs; degrees=degrees)

    for (i0, i1, coeff0, coeff1) ∈ zip(inds, inds[2:end], all_coeffs, all_coeffs[2:end])
        result[i1, :] .= evaluate_curve(coeff1, xs; degrees=degrees)

        j_range = i1 - i0
        j_vals = i0+1:i1-1
        j_ratios = (j_vals .- i0) ./ j_range
        for (j, j_ratio) ∈ zip(j_vals, j_ratios)
            result[j, :] .= evaluate_curve((1-j_ratio) .* coeff0 .+ j_ratio .* coeff1, xs; degrees=degrees)
        end
    end
    return result
end

function create_curve_image(img, inds=axes(img,1); degrees=1:5)
    xs = range(0, 2π, length=size(img, 2))
    all_coeffs = map(i->fit_looped_curve(xs, img[i, :]; degrees=degrees), inds)
    return curve_image_from_coeffs(all_coeffs, inds, size(img); degrees=degrees)
end

## Surface knots
using Optim

function get_dog_σs(rs, σ_mults, val_mult)
    m = 1 + val_mult
    σs = .√(rs.^2 .* ((1 ./ max.(eps(), σ_mults.^2)) .- 1)./max.(eps(), 2log(m)))

    σs1 = σs .* σ_mults
    return σs, σs1, m
end

function create_dog_kernel(axes, center, σs, σs1, m)
    ys = -center[1] .+ axes[1]
    xs = -center[2] .+ axes[2]

    gaussian(x, σ) = exp(-x^2 / 2σ^2)
    gaussian2d(ys, xs, σs) = gaussian.(ys, σs[1]) * gaussian.(xs, σs[2])'

    g1 = gaussian2d(ys, xs, σs)
    g2 = gaussian2d(ys, xs, σs1)
    
    g = -g1 .+ m .* g2
    ming, maxg = extrema(g)
    if (maxg - ming) > eps()
        g ./= maxg - ming
    end
    
    return g
end

function create_dog_knot_kernel(axes, center, rs, σ_mults, val_mult)
    return create_dog_kernel(axes, center, get_dog_σs(rs, σ_mults, val_mult)...)
end

function create_dog_knot_kernel(rs, σ_mults, val_mult)
    σs, σs1, m = get_dog_σs(rs, σ_mults, val_mult)
    radii = ceil.(Integer, 4 .* max.(σs, σs1))
    ys = -radii[1]:radii[1]
    xs = -radii[2]:radii[2]
    g = create_dog_kernel((ys, xs), (0, 0), σs, σs1, m)
    return centered(g)
end

function put_kernel_on_image!(img, kernel, position)
    center = CartesianIndex(round.(Integer, position))
    inds = CartesianIndices(kernel) .+ center
    R = CartesianIndices(img)
    Ifirst, Ilast = first(R), last(R)
    sizex = size(img, 2)
    # @show Ifirst, Ilast
    for (ind, val) ∈ zip(inds, vec(kernel))
        # if is_between_cartesian(Ifirst, ind, Ilast)
        if (Ifirst[1] ≤ ind[1] ≤ Ilast[1]) && (1-sizex ≤ ind[2])
            ind1 = CartesianIndex(ind[1], 1+(ind[2]+sizex-1)%sizex)
            img[ind1] += val #* img[ind1]
        end
    end
    return img
end

function create_noise(sz, σ=2, θ=0, λ=5, γ=0.3, ψ=0; base=randn(sz))
    mask = Kernel.gabor(sz..., σ, θ, λ, γ, ψ)[1][1:sz[1], 1:sz[2]]
    mask = fftshift(fft(mask))
    result = ifft(fft(base) .* ifftshift(mask)) .|> real
    return result
end


function upscale_fft(img, sz)
    f = fft(img) |> fftshift |> centered
    result = zeros(eltype(f), sz) |> centered
    result[axes(f)...] .= f
    return ifft((result)) .|> real
end

function create_fractal_noise(sz, 
    σ=2, θ=0, λ=5, γ=0.3, ψ=0; 
    octaves=[2, 4, 8, 16], 
    base=randn)

    noise = create_noise(sz, σ, θ, λ, γ, ψ; base=base(sz))
    for i ∈ octaves
        noise .+= upscale_fft(
            create_noise(sz.÷i, σ, θ, λ, γ, ψ; base=base(sz.÷i)), 
            sz) .* i^2
    end
    return noise
end

function create_fractal_gabor_noise(sz, σ=1, θ=0, λ=3, γ=0.3, ψ=0; steps=0:5, base=randn(sz))
    # mask = Kernel.gabor(sz..., σ, θ, λ, γ, ψ)[1][1:sz[1], 1:sz[2]]
    mask = reduce((acc, i)->acc .+ Kernel.gabor(sz..., σ+i, θ, λ+i, γ, ψ)[1][1:sz[1], 1:sz[2]], steps)
    mask = fftshift(fft(mask))
    result = ifft(fft(base) .* ifftshift(mask)) .|> real
    return result
end

function add_knot_to_patch(rec_img, (centroid, knot_sz), knot_params)
    mry, mrx, σy, σx, m, m1 = knot_params
    kernel = m1 .* create_dog_knot_kernel(axes(rec_img), centroid, knot_sz .* (mry, mrx), (σy, σx), m)
    # display([to_image(kernel) to_image(rec_img .+ OffsetArray(kernel, OffsetArrays.Origin(rec_img)))])
    return rec_img .+ OffsetArray(kernel, OffsetArrays.Origin(rec_img))
end

function create_knot_objective_function(centroid_sz, lower, upper)
    function objective(rec_img, params)
        params = clamp.(params, lower, upper)
        rec1 = add_knot_to_patch(rec_img, centroid_sz, params)
        return rec1 |> vec #(knot_patch .- rec1) |> vec
    end
    return objective
end

get_surface_knot_lower_bounds() = [0.5, 0.5, 0.2, 0.2, 0.1, 7]
get_surface_knot_upper_bounds() = [2, 2, 0.9, 0.9, 2, 20]

function fit_knot_kernel(knot_patch, rec_knot_patch, knot_center_sz)

    lower = get_surface_knot_lower_bounds()
    upper = get_surface_knot_upper_bounds()
    res = curve_fit(
        create_knot_objective_function(knot_center_sz, lower, upper), 
        rec_knot_patch, 
        vec(knot_patch),
        [1, 1, 0.6, 0.9, 0.2, 7];
        lower=lower,
        upper=upper,
        autodiff=:forwarddiff,
        show_trace=false,
        )
    return res |> coef
end


extend_image(img, overlap=size(img,2)) = OffsetArray(
    [img[:, 1+end-overlap:end] img img[:, 1:overlap]], 
    CartesianIndices(
        (1:size(img,1), 
        1-overlap:size(img, 2) + overlap)
        )
    )

function extract_extended_bounding_box(img, bb, extension)
    extended_img = extend_image(img)
    
    minI, maxI = extrema(CartesianIndices(img))
    minbb, maxbb = extrema(bb)
    minY = max(minI[1], minbb[1] - extension[1])
    maxY = min(maxI[1], maxbb[1] + extension[2])
    
    minX = minbb[2] - extension[2]
    maxX = maxbb[2] + extension[2]
    new_bb = CartesianIndices((minY:maxY, minX:maxX))
    
    return OffsetArray(extended_img[new_bb], new_bb)
end

function finer_fit_ellipse(sz, center, points)
    function objective(points, params)
        ry, rx, y0, x0 = params
        centered = points .- [y0, x0]
        r = ry
        centered[2, :] .*= ry/rx
        return norm.(eachcol(centered)) .- r
    end
    res = curve_fit(
        objective, 
        points, 
        zeros(size(points, 2)), 
        [sz..., center...];
        autodiff=:forwarddiff,
        show_trace=false,
        )
    ry, rx, y0, x0 = coef(res)
    return ry, rx, y0, x0
end

function find_centroid_size(knot, center)
    labels = label_components(knot .>0 )
    label = labels[CartesianIndex(round.(Integer, center))]
    knot = labels .== label
    perimeter = knot .- erode(knot, create_ball_strel(1))
    perimeter[begin, :] .= perimeter[end, :] .= false
    perimeter[:, begin] .= perimeter[:, end] .=false
    points = stack(findall(perimeter .> 0) .|> Tuple, dims=1)
    vandermonde = stack([points[:, 1].^2, points[:, 2].^2, points[:, 1], points[:, 2]])
    coeffs = vandermonde \ ones(size(points, 1))
    coeffs = [coeffs; -1]
    coeffs .*= sign(coeffs[1])
    A, C, D, E, F = coeffs
    AC = A * C
    if AC > 0
        ry = √(max(0, 2(A * E^2 + C * D^2 - 4AC*F)*(A+C+abs(A-C))))/4AC
        rx = √(max(0, 2(A * E^2 + C * D^2 - 4AC*F)*(A+C-abs(A-C))))/4AC
        y0 = -D/2A
        x0 = -E/2C
    else
        (ry, rx), (y0, x0) = val_range(points; dims=1)./2, center
    end
    ry, rx, y0, x0 = finer_fit_ellipse((ry, rx), (y0, x0), points')

    return (y0, x0), (ry, rx)
end

add_surface_ground_truth(img, knot_centers_sizes) = add_surface_knots(
    img, [(1, 1, 0.9, 0.9, 1, 1) for _ ∈ 1:length(knot_centers_sizes)], knot_centers_sizes)

function get_surface_knots_centers_sizes(extremas, img_length, knots)
    real_height = extremas[2] - extremas[1]
    # scale_coord(l, θ) = (size(img, 1) * (l - extremas[1])/real_height, θ)
    scale_coord(l, θ) = (img_length * l / real_height, θ)
    scale_size(l, θ) = (img_length * l / real_height, θ)
    centers = [scale_coord((k.l0 + k.center.Y_max), k.θ0) for k ∈ knots]
    radii = [scale_size(k.radius.Y_max/cos(get_knot_center_angles(k)[1]), (k.radius.Y_max / k.θ_mult)) for k ∈ knots]
    return centers, radii
end

function fit_surface_knots_from_knot_models(knots, img, rec_img, extremas; extension=(20,20))

    real_height = extremas[2] - extremas[1]
    centers, radii = get_surface_knots_centers_sizes(extremas, size(img, 1), knots)

    to_bbox(center, radius) = CartesianIndices((
        round(Integer, center[1] - radius[2]):round(Integer, center[1] + radius[2]),
        round(Integer, center[2] - radius[1]):round(Integer, center[2] + radius[2])
    ))
    bbox = to_bbox.(centers, radii)
    sizes = radii

    rec_knot_patches = [extract_extended_bounding_box(rec_img, box, extension) for box ∈ bbox]
    knot_patches = [extract_extended_bounding_box(img, box, extension) for box ∈ bbox]
    filtered_inds = findall(length.(rec_knot_patches) .> 0)
    rec_knot_patches = rec_knot_patches[filtered_inds]
    knot_patches = knot_patches[filtered_inds]

    knot_centers_sizes = collect(zip(centers, sizes))[filtered_inds]

    function create_gt_patch(patch, center, radii)
        return create_dog_knot_kernel(axes(patch), center, radii, (0.9, 0.9), 1)
    end
    gt_patches = create_gt_patch.(knot_patches, centers[filtered_inds], sizes[filtered_inds])

    res = [fit_knot_kernel(x...) for x ∈ zip(knot_patches, rec_knot_patches, knot_centers_sizes)]

    function copy_patch(patch)
        result = copy(patch)
        result .= 0
        return result
    end
    res_patches = [add_knot_to_patch(x...) for x ∈ zip(rec_knot_patches, knot_centers_sizes, res)]

    fitted_knot_patches = [add_knot_to_patch(copy_patch(x[1]), x[2:end]...) for x ∈ zip(rec_knot_patches, knot_centers_sizes, res)]
    gt_image = add_surface_ground_truth(zeros(size(img)), knot_centers_sizes)

    return res, knot_centers_sizes, (rec_patches=rec_knot_patches, res_patches=res_patches, knot_patches=knot_patches, gt_patches=gt_patches, gt_image=gt_image, fitted_knot_patches=fitted_knot_patches)
end


function generate_surface_knots(surface_distrib, knots, img, height_extremas)
    real_height = height_extremas[2] - height_extremas[1]

    centers, radii = get_surface_knots_centers_sizes(height_extremas, size(img, 1), knots)

    sizes = radii .|> reverse

    knot_centers_sizes = zip(centers, sizes)
    n = length(centers)
    knots_params = [ones(2, n); rand(surface_distrib, n)] |> eachcol

    lower = get_surface_knot_lower_bounds()

    upper = get_surface_knot_upper_bounds()
    knots_params = [clamp.(knots_param, lower, upper) for knots_param ∈ knots_params]

    return knots_params, knot_centers_sizes
end

add_noise_to_rec!(rec, noise; multiplier=0.4) = rec .+= multiplier .* noise ./ std(noise)

add_noise_to_rec(rec, noise; multiplier=0.4) = add_noise_to_rec!(copy(rec), noise; multiplier=multiplier)


function add_surface_knots!(rec, knots_params, knot_centers_sizes)

    for (knot_params, center_sz) ∈ zip(knots_params, knot_centers_sizes)
        mry, mrx, σy, σx, m, m1 = knot_params
        center, knot_sz = center_sz
        knot_img = m1 .* create_dog_knot_kernel(knot_sz .* (mry, mrx), (σy, σx), m)
        center = round.(Integer, center)
        put_kernel_on_image!(rec, knot_img, center)
    end
    return rec
end

add_surface_knots(rec, knots_params, knot_centers_sizes) = add_surface_knots!(copy(rec), knots_params, knot_centers_sizes)


normal_bump(x, α, μ, σ) = α * exp(-(x-μ)^2/2max(0.1, σ^2))

restore_incline(x, (slope, base)) = base .+ slope .* x
function restore_cluster_bumps!(ys, x, bumps, bump_means)
    for ((α, μ, σ), bump_mean) ∈ zip(eachcol(bumps), bump_means)
        ys .+= normal_bump.(x, α, μ + bump_mean, σ)
    end
    return ys
end
restore_cluster_bumps(x, bumps, bump_means) = restore_cluster_bumps!(zeros(length(x)), x, bumps, bump_means)

restore_incline_bumps(x, incline, bumps, bump_means) = restore_cluster_bumps!(restore_incline(x, incline), x, bumps, bump_means)

function thickness_objective(x, params, bump_means)
    clusters = reshape(params[3:end], 3, :)
    ys = restore_incline_bumps(x, params[1:2], clusters, bump_means)
    return ys
end
thickness_objective(bump_means) = (x, params) -> thickness_objective(x, params, bump_means)

function remove_height_from_heightmap(heightmap, clusters, extremas)
    heights = mean(heightmap, dims=2)

    xs = range(1, extremas[2] - extremas[1], length(heights)) |> collect
    b = stack((xs, ones(length(xs))))
    ab = b \ heights
    heights = vec(heights)
    cluster_means = [mean([(k.l0 + k.center.Y_max) for k ∈ knots]) for knots ∈ clusters]
    # @show ab
    start_params = vec([ab; repeat([1, 0, 1], length(cluster_means))])
    lower = vec([-10;   0; repeat([ 0.1, -100, 0.1], length(cluster_means))])
    upper = vec([ 10; 500;  repeat([20.0,  100, 100], length(cluster_means))])

    res = curve_fit(
        thickness_objective(cluster_means), 
        xs, 
        heights, 
        start_params;
        autodiff=:forwarddiff,
        show_trace=false,
        )

    best_params = res |> coef
    ab = best_params[1:2]
    cluster_bumps = reshape(best_params[3:end], 3, :)


    base = thickness_objective(xs, best_params, cluster_means)

    flat_heightmap = heightmap .- base

    return (ab, cluster_bumps, flat_heightmap)
end
function restore_heightmap(heightmap, extremas, ab=(0, 0), cluster_bumps=[], clusters=[])
    xs = range(1, extremas[2] - extremas[1], size(heightmap, 1)) |> collect
    cluster_means = [mean([ (k.l0 + k.center.Y_max) for k ∈ knots]) for knots ∈ clusters]

    heightmap = heightmap .+ restore_incline_bumps(xs, ab, cluster_bumps, cluster_means)
    return heightmap
end

function fit_surface(heightmap, knots, extremas)
    knot_clusters = find_knot_clusters(knots)
    
    ab, cluster_bumps, heightmap = remove_height_from_heightmap(heightmap, knot_clusters, extremas)

    coeffs = chebyshev_coeffs_from_image(heightmap; x_degrees = 1:10, y_degrees = 1:10)

    base_rec = reconstruct_from_chebyshev(coeffs, size(heightmap))

    res, knot_centers_sizes, patches = fit_surface_knots_from_knot_models(knots, heightmap, base_rec, extremas)

    noise = create_fractal_noise(size(heightmap))
    
    noise_rec = add_noise_to_rec(base_rec, noise)
    full_rec = add_surface_knots(noise_rec, res, knot_centers_sizes)
    full_rec = restore_heightmap(full_rec, extremas, ab, cluster_bumps, knot_clusters)

    return (full_rec=full_rec, 
            noise_rec=noise_rec, 
            base_rec=base_rec, 
            standardized_heightmap=heightmap,
            coeffs=coeffs,
            knot_results=res,
            patches=patches, 
            incline=(ab, cluster_bumps))
end

function heightmap_to_cartesian_points(heightmap; n_points=10000, height_extremas=(0, 5size(heightmap, 1)))
    interp = create_surface_interpolant(heightmap, height_extremas)
    l_axis, θ_axis = interp.itp.ranges
    
    rnd_lθ = rand(2, n_points) .* [maximum(l_axis), maximum(θ_axis)]
    rnd_ρ = [interp(l, θ) for (l, θ) ∈ eachcol(rnd_lθ)]
    
    points = @views from_cylindric(rnd_lθ[1, :], rnd_lθ[2, :], rnd_ρ)

    return points, rnd_ρ
end

function heightmap_from_logcentric_points(heightmap, centerline, ground_truth; n_points=10000, height_extremas=(0, 5size(heightmap, 1)))
    interp = create_surface_interpolant(heightmap, height_extremas)
    gtinterp = create_surface_interpolant(ground_truth, height_extremas)
    l_axis, θ_axis = interp.itp.ranges

    rnd_lθ = rand(2, n_points) .* [maximum(l_axis), maximum(θ_axis)]
    rnd_ρ = [interp(l, θ) for (l, θ) ∈ eachcol(rnd_lθ)]
    rnd_label = [gtinterp(l, θ) for (l, θ) ∈ eachcol(rnd_lθ)]

    points = @views convert_from_logcentric(stack((rnd_lθ[2, :], rnd_lθ[1, :], rnd_ρ))', centerline)

    return points, rnd_ρ, rnd_label
end