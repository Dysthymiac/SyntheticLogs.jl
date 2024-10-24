
export filter_point_cloud

using LinearAlgebra, Statistics, Random, BandedMatrices, FillArrays, OffsetArrays
# using GLMakie

if isinteractive() 
    println("Using GLMakie....")
    using GLMakie
else
    using Observables
end
using ColorSchemes

function filter_point_cloud(data; snake_size=180, 
                            min_radius=75, 
                            max_radius=200, 
                            max_resiudal=4, 
                            points_threshold=300, 
                            filtered_threshold=200,
                            max_distance=10, max_rad_diff=20,
                            snake_coverage=0.8,
                            snake_iterations=10,
                            max_skipped_layers=5,
                            debug_plots=false)
    first_layer = round(Integer, data[1, 1])
    last_layer = round(Integer, data[end, 1])

    points_filter = falses(size(data, 1))
    circles = Matrix{Float64}(undef, last_layer-first_layer+1, 5)
    circle_ind = 1

    circle = Vector{Float64}(undef, 3)
    snake_points = Matrix{Float64}(undef, snake_size, 2)
    snake_params = init_snake_params(snake_size)
    skipped_layers = -1

    if debug_plots
        i, end_action, exit, debug_vars... = init_debug_plots(first_layer,last_layer)
    else
        i = Observable(first_layer)
        end_action = ()-> (i[] += 1; true)
    end
    while i[] <= last_layer || (debug_plots && !exit[])
        inds = searchsorted(data[:, 1], i[])
        points = @view data[inds, :]
        if skipped_layers ≥ 0
            skipped_layers += 1
        end
        skipped_layers > max_skipped_layers && end_action() && break
        size(points, 1) < points_threshold && end_action() && continue
        
        p = @view points[:, 2:3]
        
        res, _ = init_circle_snake!(circle, snake_points, p)
        # println(circle[3])
        (circle[3] < min_radius || circle[3] > max_radius || mean(res) > max_resiudal ) && end_action() && continue

        all_snakes = optimize_snake!(snake_points, p, snake_params...; iterations=snake_iterations, save_steps=debug_plots)

        all_dists = distance_to_snake(p, snake_points)
        snake_dists = vec(minimum(all_dists, dims=1))
        sum(snake_dists .< max_distance) < snake_size * snake_coverage && end_action() && continue
        dists = vec(minimum(all_dists, dims=2))

        filtered = dists .< max_distance
        
        sum(filtered) < filtered_threshold && end_action() && continue
        skipped_layers = 0
        points_filter[inds] .= filtered
        
        
        circles[circle_ind, 1] = i[]
        circles[circle_ind, 2] = points[1, 4]
        # circle_fit_hyper_fast!(circle, snake_points)
        circle_fit_snake!(circle, snake_points)
        circles[circle_ind, 3:5] .= vec(circle)
        circle_ind += 1
        if debug_plots
            do_debug_plots(i, exit, debug_vars..., res, dists, circle, points, snake_points,all_snakes, filtered, max_distance) && break
        else
            end_action()
        end
    end
    all_rads = @view circles[1:circle_ind-1, end]
    diff_inds = [1, findall(abs.(diff(all_rads)) .> max_rad_diff)..., length(all_rads)]
    if length(diff_inds) > 1
        large_diffs = diff(diff_inds)
        largest_gap = argmax(large_diffs)
        circle_filt = diff_inds[largest_gap]:diff_inds[largest_gap+1]-1
        for j ∈ [(1:diff_inds[largest_gap]-1)..., (diff_inds[largest_gap+1]:length(all_rads))...]
            points_filter[searchsorted(data[:, 1], circles[j, 1])] .= false
        end
    else
        circle_filt = 1:circle_ind-1
    end
    return points_filter, circles[circle_filt, :]
end

function init_debug_plots(first_layer, last_layer)
    f = Figure()
    ready = Observable(true)
    exit = Observable(false)
    ax = f[1, 1] = Axis(f, aspect=DataAspect())
    grid = f[2, 1] = GridLayout(tellwidth = false)
    next_button = grid[1,3] = Button(f, label="Next")
    slider = grid[1,2] = Slider(f, range = first_layer:last_layer, startvalue = first_layer)
    prev_button = grid[1,1] = Button(f, label="Prev")
    
    save_button = grid[1,4] = Button(f, label="Save")
    on(slider.value) do _
        ready[] = true
    end
    on(next_button.clicks) do _
        set_close_to!(slider, slider.value[] + 1)
        ready[] = true
    end
    on(prev_button.clicks) do _
        set_close_to!(slider, slider.value[] - 1)
        ready[] = true
    end
    on(events(f.scene).window_open) do event
        if !event
            ready[] = true
            exit[] = true
        end
    end
    on(save_button.clicks) do _
        # CairoMakie.activate!()
        save("figure.png", f)
    end
    display(f)
    function end_action()
        val = slider.value[] + 1 > last_layer ? first_layer : slider.value[] + 1
        set_close_to!(slider, val)
        true
    end
    return slider.value, end_action, exit, ready, ax
end

function circle_fit_snake!(circle, snake_points)
    inds = vcat(2:size(snake_points, 1), [1])
    diffs = snake_points[:, 1] .* snake_points[inds, 2] .- snake_points[:, 2] .* snake_points[inds, 1]
    A = 0.5sum(diffs)
    circle[1:2] = sum((snake_points .+ snake_points[inds, :]).*diffs; dims=1)./6A
    circle[3] = mean(sqrt.(sum((snake_points .- transpose(circle[1:2])).^2, dims=2)))
end

function shift_snake(snake, shift)
    center = mean(snake, dims=1)
    diffs = snake .- center
    diffs = diffs ./ sqrt.(sum(diffs.^2, dims=2))
    return snake .+ diffs .* shift
end

function do_debug_plots(i, exit, ready, ax, res, dists, circle, points, snake_points, all_snakes, filtered, max_distance)
    println("Layer ", i[])
    println("Residual ", mean(res))
    minD, maxD = extrema(dists)
    szs = 4 .+ (1 .- (dists.-minD)./(maxD.-minD)).^2 .* 2
    empty!(ax)
    GLMakie.lines!(ax, get_circle_points(circle...)..., color=:magenta)
    GLMakie.scatter!(ax, [circle[1]], [circle[2]], markersize=1, color=:magenta)
    # lines!(ax, get_circle_points((circle.+[0,0,max_distance])...)..., color=:magenta, linestyle = :dash)
    # lines!(ax, get_circle_points((circle.+[0,0,-max_distance])...)..., color=:magenta, linestyle = :dash)

    # circle_filtered = (abs.(.√(sum((points[:, 2:3] .- transpose(circle[1:2])).^2, dims=2)) .- circle[3]) .< 10) |> vec
    # println(sum((points[:, 2:3] .- transpose(circle[1:2])).^2, dims=2))
    GLMakie.scatter!(ax, points[:, 2:3] |> transpose, markersize=szs, color=:blue)
    # scatter!(ax, points[circle_filtered, 2:3] |> transpose, markersize=szs[circle_filtered].*0.5, color=:white)
    GLMakie.scatter!(ax, points[filtered, 2:3] |> transpose, markersize=szs[filtered].*0.5, color=:white)
    
    
    GLMakie.lines!(ax, eachcol(snake_points[vcat(1:end, 1), :])..., color=:red)
    GLMakie.lines!(ax, eachcol(shift_snake(snake_points, max_distance)[vcat(1:end, 1), :])..., color=:red, linestyle = :dash)
    GLMakie.lines!(ax, eachcol(shift_snake(snake_points, -max_distance)[vcat(1:end, 1), :])..., color=:red, linestyle = :dash)


    all_dists = distance_to_snake(points[:, 2:3], snake_points)
    snake_dists = vec(minimum(all_dists, dims=1))
    # println(sum(snake_dists .< max_distance))
    GLMakie.scatter!(ax, eachcol(snake_points)..., markersize=10, color=:orange)
    GLMakie.scatter!(ax, eachcol(snake_points[snake_dists .< max_distance, :])..., markersize=8, color=:white)


    # t = range(0, 1; length=size(all_snakes, 1))
    # for i ∈ 1:size(all_snakes, 1)
    #     lines!(ax, eachcol(all_snakes[i, :, :])..., color = get(colorschemes[:magma], t[i], :extrema))
    # end
    ready[] = false
    while !ready[]
        sleep(0.001)
    end
    return exit[]
end

function init_circle_snake!(circle, snake_points, points)
    _, res, inliers = fit_circle!(circle, points)
    get_circle_points!(eachcol(snake_points)..., circle...)
    return res, inliers
end

vectornorm(x; dims) = .√sum(x.^2; dims=dims)

get_dist_γ(radius; α=1, t=0.5) = let tt = t^(1/α); (1 - tt)/(tt*radius^2) end

function init_snake_params(n, 
                            α=0.01, 
                            β=1, 
                            γ=50, 
                            σ=20,
                            dist_α=1)
    a = γ*(2α+6β)+1
    b = γ*(-α-4β)
    c = γ*β
    P = inv(Symmetric(BandedMatrix(0=>FillArrays.Fill(a, n), 
            1=>FillArrays.Fill(b, n-1), 
            2=>FillArrays.Fill(c, n-2),
            n-2=>FillArrays.Fill(c, 2),
            n-1=>FillArrays.Fill(b, 1))))
    γ_dist = get_dist_γ(σ, α=dist_α) #2/σ^2
    Exy = zeros(n, 2)
    return Exy, P, γ, γ_dist, dist_α
end

function optimize_snake!(snake, points, Exy, P, γ, γ_dist, distance_α=1; 
                        iterations=10,
                        end_mult=0.1, save_steps=true)
#=
∂Xₜ/∂t = AXₜ + fₓ(Xₜ, Yₜ)
Xₜ = (I - γA)⁻¹ { Xₜ₋₁ + γ fₓ(Xₜ₋₁, Yₜ₋₁)}
=#
    params = init_external_energy_grad(size(snake, 1), size(points, 1))
    all_snakes = save_steps ? zeros(iterations, size(snake)...) : nothing
    mults = range(1, end_mult; length=iterations)
    for i ∈ 1:iterations
        external_energy_grad!(Exy, snake, points, distance_α, γ_dist, params...)
        snake .= P*(snake .+ γ.*Exy .* mults[i])
        save_steps && (all_snakes[i, :, :] .= snake)
    end
    return all_snakes
end

function unsqueeze(A, dim)
    s = [size(A)...]
    insert!(s,dim,1)
    return reshape(A, s...)
end

function distance_to_snake(points, snake_points)
    d = diff(snake_points[vcat(end, 1:end), :]; dims=1)
    dd = sum(d .* d; dims=2);
    q = unsqueeze(points, 2) .- unsqueeze(snake_points, 1)
    qd = sum(q.*unsqueeze(d, 1); dims=3)
    t = clamp.(qd ./ unsqueeze(dd, 1), 0, 1)
    rej = q .- t .* unsqueeze(d, 1)
    # dist = minimum(.√sum(rej.*rej; dims=3); dims=2)
    return .√sum(rej.*rej; dims=3)
    # return vec(dist)
end

function init_external_energy_grad(snake_size, points_num)
    dx = zeros(snake_size, points_num)
    dy = zeros(snake_size, points_num)
    d = zeros(snake_size, points_num)
    E = zeros(snake_size, points_num)
    return dx, dy, d, E
end

function external_energy_grad!(Exy, snake, points, α, γ, dx, dy, d, E)
    dx .= snake[:, 1] .- transpose(points[:, 1])
    dy .= snake[:, 2] .- transpose(points[:, 2])
    d .= dx.^2 .+ dy.^2
    
    E .= 1 ./(γ .* d .+ 1).^(α+1)
    Exy[:, 1] .= vec(sum(-2*γ.*dx.*E; dims=2))
    Exy[:, 2] .= vec(sum(-2*γ.*dy.*E; dims=2))
    return Exy
end

function get_circle_points(center_x, center_y, radius; num_points=360)
    x, y = (Vector{Float64}(undef, num_points) for _ ∈ 1:2)
    return get_circle_points!(x, y, center_x, center_y, radius)
end

function get_circle_points!(x, y, center_x, center_y, radius)
    t = range(0, 2π; length=length(x))
    x .= cos.(t).*radius.+center_x
    y .= sin.(t).*radius.+center_y
    return x, y
end

function get_circle_points3(center_x, center_y, center_z, radius; num_points=360)
    x, y, z = (Vector{Float64}(undef, num_points) for _ ∈ 1:3)
    return get_circle_points3!(x, y, z, center_x, center_y, center_z, radius)
end

function get_circle_points3!(x, y, z, center_x, center_y, center_z, radius)
    t = range(0, 2π; length=length(x))
    x .= cos.(t).*radius.+center_x
    y .= sin.(t).*radius.+center_y
    z .= center_z
    return x, y, z
end


function circle_fit_hyper_fast!(par, XY; show_warnings=false)
    X = @view XY[:,1]
    Y = @view XY[:,2]
    Z = X.*X .+ Y.*Y
    ZXY1 = [Z X Y ones(length(Z),1)]
    M = transpose(ZXY1)*ZXY1
    S = mean(ZXY1; dims=1)
    N = [8S[1] 4S[2] 4S[3] 2; 4S[2] 1 0 0; 4S[3] 0 1 0; 2 0 0 0]
    NM = N\M
    F = eigen(NM)
    E = F.vectors
    D = F.values
    ID = sortperm(D)
    Dsort = D[ID]
    show_warnings && Dsort[1]>0 && @warn "Error in Hyper: the smallest e-value is positive..."
    
    show_warnings && Dsort[2]<0 && @warn "Error in Hyper: the second smallest e-value is negative..."
    
    A = E[:,ID[2]]

    par[1:2] .= vec(-transpose(A[2:3])/A[1]/2)
    par[3] = sqrt(A[2]*A[2]+A[3]*A[3]-4*A[1]*A[4])/abs(A[1])/2
    return par
end

squared_distance_to_circle(points, circle) = vec((.√(sum((points .- transpose(circle[1:2])).^2; dims=2)) .- circle[3]).^2);

function model_fit_repeated_lts!(par, points; minimum_sample_size=4, 
                                        trimming_size=round(Integer, size(points, 1)*0.5),
                                        iter_outlier_ratio=0.5,
                                        iter_prob=0.99,
                                        iterations=log(1-iter_prob)/log(1-(1-iter_outlier_ratio)^minimum_sample_size),
                                        use_wls=false,
                                        fit_func=circle_fit_hyper_fast!,
                                        rng_seed=0,
                                        squared_residual_func=squared_distance_to_circle)
    
    n = size(points, 1);
    # rng = isnothing(rng_seed) ? GLOBAL_RNG : MersenneTwister(rng_seed)
    # @something(rng_seed, Random.seed!(rng_seed))
    if !isnothing(rng_seed)
        Random.seed!(rng_seed)
    end
    SSe = 1e20
    best_par = Vector{Float64}(undef, 3)
    n = size(points, 1)
    residual = 0
    inliers = falses(n)
    if iterations > 0
        for _ ∈ 1:iterations
            p0 = random_subsample(points, minimum_sample_size)
            fit_func(par, p0)
            inds = sortperm(squared_residual_func(points, par))
            p1 = points[inds[1:trimming_size], :]
            fit_func(par, p1)
            res = squared_residual_func(points, par)
            SSe1, res, inds = use_wls ? get_wls(res) : get_lts(res, trimming_size)
            if SSe1 < SSe
                best_par .= par
                SSe = SSe1
                residual = res
                inliers .= inds
            end
        end
    else
        fit_func(best_par, points)
        residual = squared_residual_func(points, best_par)
        inliers .= true
    end
    par .= best_par
    return par, residual, inliers
end

fit_circle!(args...) = model_fit_repeated_lts!(args...)

function get_wls(e)
    m = median(abs.(e))
    e_star = e ./ 6m
    w = zeros(size(e_star))
    filt = abs.(e_star) .< 1
    w[filt] .= (1 .- e_star[filt].^2).^2
    res = w .* e.^2
    SSe = sum(res)
    inds = w .> 0
    return SSe, res, inds
end

function get_lts(res, h)
    inds = falses(length(res))
    inds[sortperm(res)[1:h]] .= true
    res = res[inds]
    SSe = sum(res)
    return SSe, res, inds
end

function random_subsample(points, h0)
    perm = randperm(size(points, 1));
    p0 = points[perm[1:h0], :];
    for i ∈ h0:size(points, 1)
        if rank(p0) >= 2
            break;
        end
        p0 = points[perm[1:i], :];
    end
    return p0
end

