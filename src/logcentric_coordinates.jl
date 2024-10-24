export convert_to_logcentric, convert_to_logcentric!, rescale_points!, rescale_points, convert_from_logcentric, convert_from_logcentric!

using LinearAlgebra, Statistics, StaticArrays


scale_range(val, oldMin, oldRange, newMin, newRange) = newMin + newRange * (val - oldMin) / oldRange

function rescale_points!(points, width, height, minY=nothing, maxY=nothing)
    if isnothing(minY)
        minY = minimum(points[2, :])
    end
    if isnothing(maxY)
        maxY = maximum(points[2, :])
    end
    oldLRange = maxY - minY
    points[1, :] .= scale_range.(@view(points[1, :]), -π, 2*π, 1, width-1)
    points[2, :] .= scale_range.(@view(points[2, :]), minY, oldLRange, 1, height-1)
    return points
end
rescale_points(points, width, height, minY, maxY) = rescale_points!(copy(points), width, height, minY, maxY)

function convert_to_logcentric!(points, centerline)
    converter = PointConverter(centerline)
    converted_points = apply_converter!(converter, points)
    return converted_points
end
convert_to_logcentric(points, centerline) = convert_to_logcentric!(copy(points), centerline)

function convert_to_logcentric!(points, centerline, width, height, minY=nothing, maxY=nothing)
    convert_to_logcentric!(points, centerline)
    rescale_points!(points, width, height, minY, maxY)
    return points
end
convert_to_logcentric(points, centerline, width, height, minY=nothing, maxY=nothing) = convert_to_logcentric!(copy(points), centerline, width, height, minY, maxY)

function convert_from_logcentric!(points, centerline)
    converter = PointConverter(centerline)
    converted_points = apply_converter_from_logcentric!(converter, points)
    return converted_points
end
convert_from_logcentric(points, centerline) = convert_from_logcentric!(copy(points), centerline)


find_closest_centerline_points(points, centerline) = find_closest_centerline_points(points, PointConverter(centerline))

function convert_points!(points, 
    OX, 
    OY, 
    OZ, 
    normals, 
    cumlens, 
    centerline)

    n = size(points, 2)
    filt0 = trues(n)
    filt1 = falses(n)
    filt = falses(n)
    unused = trues(n)

    centerline_points = vcat(eachcol(centerline) .|> SVector{3}, [@SVector [0, 0, 0]])
    
    for (Ox, Oy, Oz, normal, cumlen, origin, next_origin) ∈ zip(OX, OY, OZ, normals, cumlens, centerline_points[1:end-1], centerline_points[2:end])
        filt1 .= all(next_origin .== 0) .|| ((points' * normal) .≤ (next_origin ⋅ normal))
        
        filt .= filt0 .& filt1
        p = @view points[:, filt .& unused]
        unused[filt] .= false
        
        filt0 .= .!filt1
        

        T = [Ox Oy Oz]'

        convert_points_subset!(p, T, origin, cumlen)
    end
    return points
end

function convert_points(points, 
    OX, 
    OY, 
    OZ, 
    normals, 
    cumlens, 
    centerline)

    result = copy(points)
    convert_points!(result, OX, OY, OZ, normals, cumlens, centerline)
    return result
end

function convert_points_from_logcentric!(points, 
    OX, 
    OY, 
    OZ, 
    normals, 
    cumlens, 
    centerline)

    n = size(points, 2)
    unused = trues(n)
    filt1 = falses(n)
    filt = falses(n)
    centerline_points = vcat(eachcol(centerline) .|> SVector{3}, [@SVector [0, 0, 0]])
    prev_cumlen = -Inf
    
    for (Ox, Oy, Oz, cumlen, origin, next_origin) ∈ zip(OX, OY, OZ, cumlens, centerline_points[1:end-1], centerline_points[2:end])
        filt1 .= all(next_origin .== 0) .|| (points[2, :] .≤ cumlen)
        filt .= (points[2, :] .> prev_cumlen) .& filt1 .& unused
        unused[filt] .= false
        !any(filt) && continue
        p = @view points[:, filt]
        
        prev_cumlen = cumlen
        # @show cumlen
        T = [Ox Oy Oz]
        convert_points_subset_from_logcentric!(p, T, origin, cumlen)
    end
    # @show sum(unused)
    return points
end

function convert_points_from_logcentric(points, 
    OX, 
    OY, 
    OZ, 
    normals, 
    cumlens, 
    centerline)

    result = copy(points)
    convert_points_from_logcentric!(result, OX, OY, OZ, normals, cumlens, centerline)
    return result
end

struct PointConverter{T1, T2, T3, T4}
    OX::T1
    OY::T1
    OZ::T1
    normals::T2
    cumulative_lengths::T3
    centerline::T4

    # PointConverter(centerline) = new(get_planes(centerline)..., centerline[1:end-1, :])
    # PointConverter{T}(OX, OY, OZ, normals, cumulative_lengths, centerline) where {T<:Real} = new(OX, OY, OZ, normals, cumulative_lengths, centerline)
end

PointConverter(centerline) = PointConverter(get_planes(centerline)..., centerline[:, 1:end-1])

apply_converter(converter::PointConverter, points) = convert_points(points, 
                                                                            converter.OX, 
                                                                            converter.OY, 
                                                                            converter.OZ,
                                                                            converter.normals, 
                                                                            converter.cumulative_lengths,
                                                                            converter.centerline)

apply_converter!(converter::PointConverter, points) = convert_points!(points, 
                                                                            converter.OX, 
                                                                            converter.OY, 
                                                                            converter.OZ,
                                                                            converter.normals, 
                                                                            converter.cumulative_lengths,
                                                                            converter.centerline)

apply_converter_from_logcentric(converter::PointConverter, points) = convert_points_from_logcentric(points, 
                                                                            converter.OX, 
                                                                            converter.OY, 
                                                                            converter.OZ,
                                                                            converter.normals, 
                                                                            converter.cumulative_lengths,
                                                                            converter.centerline)

apply_converter_from_logcentric!(converter::PointConverter, points) = convert_points_from_logcentric!(points, 
                                                                            converter.OX, 
                                                                            converter.OY, 
                                                                            converter.OZ,
                                                                            converter.normals, 
                                                                            converter.cumulative_lengths,
                                                                            converter.centerline)

(converter::PointConverter)(points) = apply_converter(converter, points)

function get_planes(centerline)
    centerline_vectors = diff(centerline; dims=2)
    centerline_lengths = norm.(eachcol(centerline_vectors))
    cumlens = [0; cumsum(centerline_lengths)]

    normalized_vectors = (centerline_vectors ./ centerline_lengths') |> eachcol .|> SVector{3}
    n0 = normalized_vectors |> diff .|> normalize
    n1 = n0 .× normalized_vectors[2:end]
    normals = [n1 .× n0; [@SVector zeros(3)] ]
    
    baseOY = @SVector [0, 1, 0]
    OZ = normalized_vectors
    OX = Ref(baseOY) .× OZ
    OY = OZ .× OX
    
    return (OX, OY, OZ, normals, cumlens)
end

function convert_points_subset!(p, 
                                T, 
                                origin, 
                                cumlen)
    p .= T * (p .- origin)
    p .= [atan.(p[2, :], p[1, :])  p[3, :] .+ cumlen hypot.(p[1, :], p[2, :]) ] |> transpose
end

function convert_points_subset_from_logcentric!(p, 
    T, 
    origin, 
    cumlen)
    
    p .= [p[3, :] .* cosd.(p[1, :]) p[3, :] .* sind.(p[1, :]) p[2, :] .- cumlen] |> transpose
    p .= T * p .+ origin
    # wait(display(scatter(p)))
end

function find_closest_centerline_points(points, 
                                        OX, 
                                        OY, 
                                        OZ, 
                                        normals,  
                                        centerline)
    n = size(points, 2)
    closest_points = zeros(3, n)
    find_closest_centerline_points!(closest_points, points, OX, OY, OZ, normals, centerline)
    return closest_points
end

function find_closest_centerline_points!(closest_points, 
    points, 
    OX, 
    OY, 
    OZ, 
    normals,  
    centerline)

    n = size(points, 2)
    filt0 = trues(n)
    filt1 = falses(n)
    filt = falses(n)
    unused = trues(n)
    
    for (Ox, Oy, Oz, normal, origin) in zip(OX, OY, OZ, normals, eachcol(centerline))

        filt1 .= (points' * normal) .≤ (origin ⋅ normal)
        filt .= filt0 .& filt1
        p = @view points[:, filt .& unused]

        T = [Ox Oy Oz] #|> transpose

        # p0 = (p .- transpose(origin)) * T
        p0 = T * (p .- origin)
        
        t = p0[3, :]
        closest_points[:, filt .& unused] = (origin .+ transpose(t) .* Oz)


        unused[filt] .= false
        filt0 .= .!filt1
    end
    return closest_points
end


find_closest_centerline_points!(closest_points, points, converter::PointConverter) = find_closest_centerline_points!(closest_points, 
                                                                                            points, 
                                                                                            converter.OX, 
                                                                                            converter.OY, 
                                                                                            converter.OZ,
                                                                                            converter.normals, 
                                                                                            converter.centerline)

find_closest_centerline_points(points, converter::PointConverter) = find_closest_centerline_points(points, 
                                                                            converter.OX, 
                                                                            converter.OY, 
                                                                            converter.OZ,
                                                                            converter.normals, 
                                                                            converter.centerline)
