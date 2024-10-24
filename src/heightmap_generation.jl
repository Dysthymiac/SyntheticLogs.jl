export calculate_heightmap, calculate_heightmap!, segment_branches

using CUDA, CUDA.CUSPARSE

function calculate_heightmap(converted_points, α, width, height, realHeight; use_cuda=true)
    heightmap = zeros(height, width)
    calculate_heightmap!(heightmap, converted_points, α, realHeight; use_cuda=use_cuda)
    return heightmap
end

function do_cg!(b, R; use_cuda=true)
    if use_cuda
        cub = CuArray(b)
        cuR = sparse_to_cuda(R)
        cg!(cub, cuR, cub)
        copyto!(b, cub)
    else
        cg!(b, R, b)
    end
end

function calculate_heightmap!(heightmap, converted_points, α, realHeight; use_cuda=true)
    width = size(heightmap, 2)
    height = size(heightmap, 1)
    b = vec(heightmap)
    invΔθ = 360 / width
    invΔL = realHeight / height
    reg = construct_regularizer(width, height, invΔθ, invΔL) #|> cu

    R = constructAb!(b, converted_points, width, height)# |> cu
    R .+= α*reg
    do_cg!(b, R; use_cuda=use_cuda)
end

make_odd(x) = x + 1 - x % 2

normalize_from_extrema(x, min, max) = ((x - min) / (max - min))
normalize_from_extrema(x) = normalize_from_extrema.(x, extrema(x)...)

function gaussian_second_diff(σ, sz, dim)
    k = Kernel.gaussian((σ, σ), make_odd.(sz))
    x = axes(k)[dim]
    res = (-(-1 .+ x.^2 ./ σ^2) .* k ./ (2π * σ^4)) |> parent
    return res[1:sz[1], 1:sz[2]] 
end
gaussian_second_diff_y(σ, sz) = gaussian_second_diff(σ, sz, 1)
gaussian_second_diff_x(σ, sz) = gaussian_second_diff(σ, sz, 1)

gaussian_second_diff_weighted_sum(σy, σx, sz) = σy .* gaussian_second_diff_y(σy, sz) .+ σx .* gaussian_second_diff_x(σx, sz) 

function segment_branches(I, branch_radius, real_size)
    unpad(I) = I[size(I, 1)÷2+1:end, :]
    pad(I) = vcat(I[end:-1:1, :], I)
    σ = branch_radius .* size(I) ./ real_size
    Ip = pad(I)
    kernel = gaussian_second_diff_weighted_sum(σ..., size(Ip))
    return fft(Ip) .* fft(kernel) |> ifft |> ifftshift |> real |> unpad |> x->max.(x, 0) |> normalize_from_extrema
end


sparse_to_cuda(x::SparseMatrixCSC{Tv, Ti}) where {Tv,Ti<:Integer} = CuSparseMatrixCSC{Tv, Ti}(CuArray(x.colptr), CuArray(x.rowval), CuArray(x.nzval), (x.m, x.n))

function construct_regularizer(width, height, invΔθ, invΔL)
    szN = width * height
    szL = szN-width
    szθ = szN-height

    valsL = fill(invΔL, 2szL)
    valsL[1:2:end] .= -invΔL
    
    valsθ = fill(invΔθ, 2szN)
    valsθ[1:2:2height] .= -invΔθ
    valsθ[2height+2:2:end] .= -invΔθ

    rowsL = Vector{Int32}(undef, 2szL)
    rowsL[1:2:end] .= 1:szL
    rowsL[2:2:end] .= 1:szL
    
    rowsθ = Vector{Int32}(undef, 2szN)
    rowsθ[1:2:2height] .= 1:height
    rowsθ[2:2:2height] .= szθ+1:szN
    rowsθ[2height+1:2:2szN] .= 1:szθ
    rowsθ[2height+2:2:2szN] .= height+1:szN
    
    colsL = Vector{Int32}(undef, szN+1)
    colsL[1] = 1
    colsL[2:szL+1] .= 2:2:2szL
    colsL[szL+2:end] .= 2szL+1

    colsθ = Vector{Int32}(undef, szN+1)
    colsθ[1:szN] .= 1:2:2szN
    colsθ[end] = 2szN+1

    @inbounds begin
        DL = SparseMatrixCSC(szN, szN, colsL, rowsL, valsL)
        Dθ = SparseMatrixCSC(szN, szN, colsθ, rowsθ, valsθ)
    end
    return transpose(Dθ) * Dθ .+ transpose(DL) * DL
end


function constructAb!(b, converted_points::AbstractMatrix{T}, width, height) where T
    N = size(converted_points, 2)
    M = height * width
    colptr = Vector{Int32}(undef, N+1)
    rowvec = Vector{Int32}(undef, 4N)
    nzvec = Vector{T}(undef, 4N)

    @inbounds begin
        x1 = @~ floor.(Int32, @view converted_points[1, :])
        y1 = @~ floor.(Int32, @view converted_points[2, :])
        
        x2 = @~ min.(x1 .+ 1, width)
        y2 = @~ min.(y1 .+ 1, height)
        wx = @~ converted_points[1, :] .- x1
        wy = @~ converted_points[2, :] .- y1

        colptr[1:N] .= 1:4:4N
        colptr[end] = 4N+1

        rowvec[1:4:end] .= y1.+(x1.-1).*height
        rowvec[2:4:end] .= y2.+(x1.-1).*height
        rowvec[3:4:end] .= y1.+(x2.-1).*height
        rowvec[4:4:end] .= y2.+(x2.-1).*height

        nzvec[1:4:end] .= wx.*wy
        nzvec[2:4:end] .= wx.*(1 .-wy)
        nzvec[3:4:end] .= (1 .-wx).*wy
        nzvec[4:4:end] .= (1 .-wx).*(1 .-wy)
        Aᵗ = SparseMatrixCSC(M, N, colptr, rowvec, nzvec)
    end
    dropzeros!(Aᵗ)
    AᵗA = Aᵗ * transpose(Aᵗ)
    @views mul!(b, Aᵗ, converted_points[3, :])
    
    return AᵗA
end

