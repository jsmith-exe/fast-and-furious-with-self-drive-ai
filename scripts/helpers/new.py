#!/usr/bin/env python3
import cv2
import sys

def main():
    # 1. Print whether OpenCV was built with CUDA
    build_info = cv2.getBuildInformation()
    cuda_enabled = any(line.strip().startswith("CUDA:") and "YES" in line for line in build_info.splitlines())
    
    if not cuda_enabled:
        print("OpenCV NOT built with CUDA support.")
        sys.exit(2)

    # 2. Query how many CUDA devices OpenCV sees
    num_devices = cv2.cuda.getCudaEnabledDeviceCount()
    print("OpenCV built with CUDA support.")
    print(f"Found {num_devices} CUDA-capable device(s).")
    
    # 3. Final pass/fail
    if num_devices > 0:
        print("gpu on")
        sys.exit(0)
    else:
        print("gpu off (no device detected)")
        sys.exit(1)

if __name__ == "__main__":
    main()
