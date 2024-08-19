
# using ChainRules

@generated function farthest_sampling_cuda_kernel(xyz_all, tmp_all, idx_all, batches, ::Val{block_size}) where {block_size} 
    max_power = ceil(Int32, log2(block_size))
    powers = SVector{max_power+1, Int32}(2 .^p for p ∈ max_power:-1:0)
quote
    @inbounds begin 
        tid = threadIdx().x 
        bid = blockIdx().x

        start_i = batches[1, bid]
        end_i = batches[2, bid]
        start_m_i = batches[3, bid]
        end_m_i = batches[4, bid] 

        xyz = @view xyz_all[:, start_i:end_i]
        tmp = @view tmp_all[start_i:end_i]
        idx = @view idx_all[start_m_i:end_m_i]

        n = size(xyz, 2)
        dists = CuStaticSharedArray(Float32, block_size)
        dists_i = CuStaticSharedArray(Int32, block_size)
        old = 1

        stride = block_size
        if tid == 1 
            idx[1] = start_i
        end
        sync_threads()
        for j ∈ 1:length(idx)
            besti = 1
            best = Float32(-1)
            x1 = xyz[1, old]
            y1 = xyz[2, old]
            z1 = xyz[3, old]
            for k ∈ tid:stride:n
                x2 = xyz[1, k]
                y2 = xyz[2, k]
                z2 = xyz[3, k]
                d = (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2
                d2 = min(d, tmp[k])
                tmp[k] = d2
                if d2 > best
                    besti = k
                    best = d2
                end
            end
            dists[tid] = best
            dists_i[tid] = besti
            sync_threads()
            for i ∈ 1:length($powers)-1
                if tid-1 < $powers[i+1] 
                    tid2 = tid+$powers[i+1]
                    if dists[tid2] > dists[tid]
                        dists[tid] = dists[tid2]
                        dists_i[tid] = dists_i[tid2]
                    end
                end
                sync_threads()
            end
            old = dists_i[1]
            if tid == 1
                idx[j] = old + start_i - 1
            end
        end
        return
    end
end
end

function batch_indices(b, n)
    starts = 1:b:n
    ends = b:b:n
    if ends[end] ≠ n
        ends = vcat(ends, n)
    end
    return vcat(transpose(starts), transpose(ends))
end

function farthest_sampling(m, xyz::T; batch_size=1024*5, TOTAL_THREADS=1024) where T <: CuArray
    n = size(xyz, 2)
    idx = CuArray{Int32}(undef, m)
    n_threads = opt_n_threads(n; TOTAL_THREADS=TOTAL_THREADS)

    batch_size = min(batch_size, n)
    m_batch_size = round(Int32, batch_size * m/n)
    
    batches = CuArray(vcat(batch_indices(batch_size, n), batch_indices(m_batch_size, m)))
    batches_n = size(batches, 2)

    tmp = CUDA.fill(typemax(Float32), n)
    @CUDA.sync @cuda blocks=batches_n threads=n_threads farthest_sampling_cuda_kernel(xyz, tmp, idx, batches, Val(n_threads))

    return idx
end

function farthest_sampling_cpu(m, xyz)
    
    idx = Array{Int32}(undef, m)
    dists = fill(typemax(Float64), m)

    idx[1] = xyz[:, 1]

    points_left = 2:size(xyz, 2)

    for i ∈ 2:m
        last_added = idx[i-1]
        
        dist_to_last_added_point = 
            sum((xyz[:, last_added] .- xyz[:, points_left]).^2, dims=1)

        dists[points_left] .= min.(dist_to_last_added_point, dists[points_left])

        best_ind = np.argmax(dists[points_left])
        idx[i] = points_left[best_ind]

        deleteat!(points_left, best_ind)
    end

    return idx
end

function farthest_sampling(m, xyz; use_cuda=false, kws...)
    if use_cuda
        return farthest_sampling(m, cu(xyz); kws...)
    else
        return farthest_sampling_cpu(m, xyz)
    end
end

# ChainRules.@non_differentiable farthest_sampling(m, xyz)

