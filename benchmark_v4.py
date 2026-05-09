import os
import time

import cv2
import numpy as np

from cg_v1 import GridProcessorGPU
from fpie.io import read_images, write_image

def main():
    print("Loading images...")
    try:
        src, mask, tgt = read_images(
            "test2_src.png", "test2_mask.png", "test2_target.png"
        )
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    print("Initializing OpenCL Solver CG v1...")
    proc = GridProcessorGPU()

    n_vars = proc.reset(src, mask, tgt, (0, 0), (260, 260))
    print(f"Number of variables to solve: {n_vars}")

    print("Running solver...")
    start_time = time.time()

    # The step method in GridProcessorGPU automatically loops up to iterations
    # and breaks early if converged.
    result, err = proc.step(5000)

    end_time = time.time()
    print(f"Final Error: {err}")
    print(f"Solved in {end_time - start_time:.4f} seconds")

    write_image("result_cl.jpg", result)
    print("Saved result to result_cl.jpg")


if __name__ == "__main__":
    main()
