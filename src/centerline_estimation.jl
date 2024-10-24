
export fit_predict_3D_centerline, fit_3D_centerline, predict_3D_centerline



function fit_3D_centerline(X, Y, Z; segments=30, λ=250.0)
    
    minX, maxX = extrema(X)
    Xs = range(minX, maxX, length=segments+1)

    splX = fit(SmoothingSpline, X, Y, λ) 
    splY = fit(SmoothingSpline, X, Z, λ) 

    return Xs, splX, splY
end

function predict_3D_centerline(Xs, splX, splY)
    Y_pred = predict(splX, Xs) # fitted vector
    Z_pred = predict(splY, Xs) # fitted vector
    return Xs, Y_pred, Z_pred
end

function fit_predict_3D_centerline(X, Y, Z; segments=30, λ=250.0)
    return predict_3D_centerline(fit_3D_centerline(X, Y, Z; segments=segments, λ=λ)...)
end

function fit_predict_3D_centerline(points; segments=30, λ=250.0)
    res = fit_predict_3D_centerline(points[1, :], points[2, :], points[3, :]; segments=segments, λ=λ)
    return stack(res, dims=1)
end