import os
import time

import cv2
import numpy as np
import pyopencl as cl

class SolverCGv1:
    def __init__(self):
        # Initialize OpenCL context and queue
        platforms = cl.get_platforms()
        # Just grab the first platform/device for simplicity
        self.ctx = cl.Context(dev_type=cl.device_type.ALL,
                              properties=[(cl.context_properties.PLATFORM, platforms[0])])
        self.queue = cl.CommandQueue(self.ctx)
        
        # Load kernel
        with open("solver.cl", "r") as f:
            kernel_src = f.read()
        self.prg = cl.Program(self.ctx, kernel_src).build()

    def reset(self, N, M, mask, tgt, grad):
        self.N = N
        self.M = M
        self.m3 = M * 3
        
        # OpenCL needs planar layout for these kernels: c * N * M + id
        # The input numpy arrays are shape (N, M, 3). We need (3, N, M)
        self.mask_host = np.ascontiguousarray(mask.astype(np.int8))
        self.tgt_host = np.ascontiguousarray(tgt.transpose(2, 0, 1).astype(np.float32))
        self.grad_host = np.ascontiguousarray(grad.transpose(2, 0, 1).astype(np.float32))

        # We need buffers for p, r, Ap, partial_sums
        self.r_host = np.zeros_like(self.tgt_host)
        self.p_host = np.zeros_like(self.tgt_host)
        self.Ap_host = np.zeros_like(self.tgt_host)

        mf = cl.mem_flags
        self.mask_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.mask_host)
        self.tgt_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.tgt_host)
        self.grad_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.grad_host)
        self.r_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.r_host)
        self.p_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.p_host)
        self.Ap_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.Ap_host)

        # For partial sums: block size will be 16x16 = 256. 
        # Number of groups = ceil(M/16) * ceil(N/16)
        self.wg_x = 16
        self.wg_y = 16
        self.gx = (M + self.wg_x - 1) // self.wg_x
        self.gy = (N + self.wg_y - 1) // self.wg_y
        self.num_groups = self.gx * self.gy
        
        self.partial_sums_host = np.zeros(self.num_groups, dtype=np.float32)
        self.partial_sums_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.partial_sums_host.nbytes)
        
        # Step 0: Initialize Residual (r_0 = b - A x_0)
        # This is done on CPU for simplicity as it's just once
        for c in range(3):
            for y in range(1, N - 1):
                for x in range(1, M - 1):
                    id = y * M + x
                    if mask[y, x]:
                        # A*x
                        val = 4.0 * self.tgt_host[c, y, x]
                        val -= self.tgt_host[c, y-1, x] if mask[y-1, x] else 0.0
                        val -= self.tgt_host[c, y+1, x] if mask[y+1, x] else 0.0
                        val -= self.tgt_host[c, y, x-1] if mask[y, x-1] else 0.0
                        val -= self.tgt_host[c, y, x+1] if mask[y, x+1] else 0.0
                        
                        r_val = self.grad_host[c, y, x] - val
                        self.r_host[c, y, x] = r_val
                        self.p_host[c, y, x] = r_val

        cl.enqueue_copy(self.queue, self.r_buf, self.r_host)
        cl.enqueue_copy(self.queue, self.p_buf, self.p_host)
        
        # Compute initial r_sq
        self.r_sq = np.zeros(3, dtype=np.float32)
        for c in range(3):
            self.r_sq[c] = np.sum((self.r_host[c][mask == 1])**2)

    def step(self, iterations):
        global_work_size = (self.gx * self.wg_x, self.gy * self.wg_y)
        local_work_size = (self.wg_x, self.wg_y)
        global_work_size_3d = (self.gx * self.wg_x, self.gy * self.wg_y, 3)
        local_work_size_3d = (self.wg_x, self.wg_y, 1)

        err = np.zeros(3, dtype=np.float32)
        
        for it in range(iterations):
            # 1. Ap = A * p
            self.prg.kernel_Ap(self.queue, global_work_size_3d, local_work_size_3d,
                               self.p_buf, self.mask_buf, self.Ap_buf,
                               np.int32(self.N), np.int32(self.M))

            pAp = np.zeros(3, dtype=np.float32)
            r_sq_new = np.zeros(3, dtype=np.float32)

            for c in range(3):
                # 2. Dot Product pAp = p * Ap
                self.prg.kernel_dot_product(self.queue, global_work_size, local_work_size,
                                            self.p_buf, self.Ap_buf, self.mask_buf,
                                            self.partial_sums_buf,
                                            cl.LocalMemory(self.wg_x * self.wg_y * 4),
                                            np.int32(self.N), np.int32(self.M), np.int32(c))
                cl.enqueue_copy(self.queue, self.partial_sums_host, self.partial_sums_buf)
                pAp[c] = np.sum(self.partial_sums_host)
                
                # 3. Alpha and Update x, r
                alpha = np.float32(self.r_sq[c] / pAp[c]) if pAp[c] > 1e-12 else np.float32(0.0)
                
                self.prg.kernel_update_x_r(self.queue, global_work_size, local_work_size,
                                           self.tgt_buf, self.r_buf, self.p_buf, self.Ap_buf, self.mask_buf,
                                           alpha, np.int32(self.N), np.int32(self.M), np.int32(c))

                # 4. Dot Product r_sq_new = r * r
                self.prg.kernel_dot_product(self.queue, global_work_size, local_work_size,
                                            self.r_buf, self.r_buf, self.mask_buf,
                                            self.partial_sums_buf,
                                            cl.LocalMemory(self.wg_x * self.wg_y * 4),
                                            np.int32(self.N), np.int32(self.M), np.int32(c))
                cl.enqueue_copy(self.queue, self.partial_sums_host, self.partial_sums_buf)
                r_sq_new[c] = np.sum(self.partial_sums_host)
                err[c] = np.sqrt(r_sq_new[c])

                # 5. Beta and Update p
                beta = np.float32(r_sq_new[c] / self.r_sq[c]) if self.r_sq[c] > 1e-12 else np.float32(0.0)
                
                self.prg.kernel_update_p(self.queue, global_work_size, local_work_size,
                                         self.p_buf, self.r_buf, self.mask_buf,
                                         beta, np.int32(self.N), np.int32(self.M), np.int32(c))
            
            self.r_sq = r_sq_new
            
            # Stop early if converged
            if np.max(err) < 15.0:
                break

        # Read back final tgt
        cl.enqueue_copy(self.queue, self.tgt_host, self.tgt_buf)
        # Convert back to (N, M, 3)
        final_img = self.tgt_host.transpose(1, 2, 0)
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)
        
        return final_img, err

class GridProcessorGPU:
    def __init__(self, n_cpu=8, grid_x=8, grid_y=8):
        self.core = SolverCGv1()

    def reset(self, src, mask, tgt, offset_src, offset_tgt):
        N, M = tgt.shape[:2]
        
        # We need to construct the large arrays padded with zeros
        # This matches the C++ process logic
        padded_mask = np.zeros((N, M), dtype=np.int32)
        padded_grad = np.zeros((N, M, 3), dtype=np.float32)
        
        src_h, src_w = mask.shape[:2]
        for y in range(src_h):
            for x in range(src_w):
                if np.any(mask[y, x] > 0):
                    ty = y + offset_tgt[0]
                    tx = x + offset_tgt[1]
                    if 0 <= ty < N and 0 <= tx < M:
                        padded_mask[ty, tx] = 1
                        sy = y + offset_src[0]
                        sx = x + offset_src[1]
                        
                        # grad is 4*src_c - neighbors
                        g = 4.0 * src[sy, sx]
                        g -= src[sy-1, sx] if sy > 0 else 0
                        g -= src[sy+1, sx] if sy < src.shape[0]-1 else 0
                        g -= src[sy, sx-1] if sx > 0 else 0
                        g -= src[sy, sx+1] if sx < src.shape[1]-1 else 0
                        padded_grad[ty, tx] = g

        tgt_float = tgt.astype(np.float32)
        self.core.reset(N, M, padded_mask, tgt_float, padded_grad)
        return np.sum(padded_mask)

    def step(self, iterations):
        return self.core.step(iterations)

