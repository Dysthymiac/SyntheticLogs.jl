"""
    fit_centerline_coeffs(centerline; degrees=2:3)

Fit Chebyshev coefficients to the `centerline`, which is passed as a `3 × n` matrix of points. 
Returns `2 × length(degrees)` matrix with coefficients for `y` and `z` coordinates of centerline.

!!! note

    It is assumed that the centerline is aligned along the `x` axis. If it is not the case, 
    the centerline can be rotated with the help of PCA to make sure that the first principle component corresponds to `x` axis
"""
function fit_centerline_coeffs(centerline; degrees=2:3)
    xs = range(0, 1, length=size(centerline, 2))
    ys = centerline[2, :]
    zs = centerline[3, :]
    y_coeffs = fit_chebyshev_curve(xs, ys; degrees=degrees)
    z_coeffs = fit_chebyshev_curve(xs, zs; degrees=degrees)
    return [y_coeffs; z_coeffs]
end


function process_centerline(centerline)
    centerline = centerline ./ val_range(centerline[1, :])
    centerline[1, :] .-= minimum(centerline)
    centerline[2:3, :] .-= mean(centerline[2:3, :], dims=2)
    return centerline
end


function fit_centerline_dist(all_data)
    centerlines = getfield.(all_data, :centerline_raw)
    centerlines_coeffs = fit_centerline_coeffs.(process_centerline.(centerlines)) |> stack

    return Distributions.fit(MvNormal, centerlines_coeffs)
end

"""
    evaluate_centerline(coeffs, n=100, len=1; degrees=2:3)

Evaluate `n` centerline points using Chebyshev coefficients `coeffs` that have `x` coordinates of `range(0, len, n)`. 
"""
function evaluate_centerline(coeffs, n=100, len=1; degrees=2:3)
    xs = range(0, 1, n)
    ys = evaluate_chebyshev_curve(coeffs[1:length(degrees)], xs; degrees=degrees)
    zs = evaluate_chebyshev_curve(coeffs[length(degrees)+1:end], xs; degrees=degrees)
    return stack((xs .* len, ys, zs); dims=1)
end