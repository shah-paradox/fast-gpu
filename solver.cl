// =========================================================================
// KERNEL 1: Sparse Matrix-Vector Multiplication (Ap = A * p)
// =========================================================================
__kernel void kernel_Ap(
    __global const float* p,     // Search direction
    __global const char* mask,   // Mask (0 or 1)
    __global float* Ap,          // Output: Result of A * p
    const int N, const int M) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int c = get_global_id(2); // Channel (0, 1, 2)

    if (x < 1 || x >= M - 1 || y < 1 || y >= N - 1) return;

    int id = y * M + x;
    int ch_off = c * N * M;
    int gid = ch_off + id;

    if (mask[id]) {
        // Dirichlet Boundary: If neighbor is outside mask, its p value is 0
        float p_c = p[gid];
        float p_t = mask[id - M] ? p[gid - M] : 0.0f;
        float p_b = mask[id + M] ? p[gid + M] : 0.0f;
        float p_l = mask[id - 1] ? p[gid - 1] : 0.0f;
        float p_r = mask[id + 1] ? p[gid + 1] : 0.0f;

        // Poisson Stencil: 4*center - neighbors
        Ap[gid] = 4.0f * p_c - (p_t + p_b + p_l + p_r);
    } else {
        Ap[gid] = 0.0f;
    }
}

// =========================================================================
// KERNEL 2: Vector Math (Update Solution and Residual)
// =========================================================================
__kernel void kernel_update_x_r(
    __global float* x_vec,       // Solution (tgt)
    __global float* r,           // Residual
    __global const float* p,     // Search direction
    __global const float* Ap,    // A * p
    __global const char* mask,   // Mask
    const float alpha,           // Scalar step size
    const int N, const int M, const int c) 
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    
    if (px >= M || py >= N) return;
    
    int id = py * M + px;
    int gid = (c * N * M) + id;

    if (mask[id]) {
        x_vec[gid] += alpha * p[gid];
        r[gid]     -= alpha * Ap[gid];
    }
}

// =========================================================================
// KERNEL 3: Parallel Reduction (Dot Product)
// =========================================================================
// Computes local sums of (a * b) and stores them in partial_sums array.
// The CPU will do the final tiny sum of partials.
__kernel void kernel_dot_product(
    __global const float* a,
    __global const float* b,
    __global const char* mask,
    __global float* partial_sums,
    __local float* local_cache,
    const int N, const int M, const int c)
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    int local_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int group_id = get_group_id(1) * get_num_groups(0) + get_group_id(0);

    float val = 0.0f;
    if (px < M && py < N) {
        int id = py * M + px;
        int gid = (c * N * M) + id;
        if (mask[id]) {
            val = a[gid] * b[gid];
        }
    }

    local_cache[local_id] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree-based reduction in local memory
    for (int s = (get_local_size(0) * get_local_size(1)) / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            local_cache[local_id] += local_cache[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        partial_sums[group_id] = local_cache[0];
    }
}

// =========================================================================
// KERNEL 4: Update Search Direction (p = r + beta * p)
// =========================================================================
__kernel void kernel_update_p(
    __global float* p,
    __global const float* r,
    __global const char* mask,
    const float beta,
    const int N, const int M, const int c)
{
    int px = get_global_id(0);
    int py = get_global_id(1);

    if (px >= M || py >= N) return;
    
    int id = py * M + px;
    int gid = (c * N * M) + id;

    if (mask[id]) {
        p[gid] = r[gid] + beta * p[gid];
    } else {
        p[gid] = 0.0f;
    }
}
