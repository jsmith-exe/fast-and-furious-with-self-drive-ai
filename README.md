# JetRacer: High-Speed Lane Following with MPC

**Author:** Jamie Smith  
**Date:** August 2025

## Overview
This project implements **lane following** on the NVIDIA JetRacer platform using a **Model Predictive Controller (MPC)**. The work also integrates obstacle detection and evasion strategies using both monocular camera and camera–LiDAR fused approaches.

The repository contains:
- LaTeX beamer presentation source (`.tex`).
- PowerPoint (`.pptm`) version of the presentation with embedded demonstration videos.
- Supporting code and demonstration assets.

---

## Project Summary

### Goals
- Replace a basic proportional (P) controller with an advanced MPC to improve stability and speed.
- Integrate real-time obstacle detection and avoidance into the lane-following pipeline.
- Optimize the system for onboard processing on a Jetson Nano.

### JetRacer Platform
- **Compute:** NVIDIA Jetson Nano
- **Sensors:** Monocular forward-facing camera, 360° LiDAR (10 Hz)
- **Drive:** Bicycle-model steering with rear-wheel drive

---

## Lane Following

### Vision Processing
1. Convert camera feed to binary (black & white) for lane detection.
2. Use edge detection to find lane boundaries.
3. Compute **lateral deviation** and **yaw angle** relative to lane center.

<p align="left">
  <img src="media/gifs/lane.gif" alt="Lane Demo" width="800">
</p>

### Controllers
- **P Controller:** Limited to low-speed operation, unstable at higher speeds.
- **PID Controller:** Improved stability, 162% speed increase over P controller.
- **MPC Controller:** Handles constraints, anticipates deviations, achieves up to 249% speed increase over P controller.

<p align="left">
  <img src="media/gifs/mpc_control_736.gif" alt="MPC Demo" width="800">
</p>

---

## Obstacle Detection & Evasion

### First Approach
- HSV color thresholding to segment obstacle.
- Distance estimation from pixel width scaling.
- Generate evasive and return waypoints around the obstacle.

<p align="left">
  <img src="media/gifs/object_distance.gif" alt="Distance Demo" width="800">
</p>

### Second Approach
- Shift detected lane centerline in vision pipeline by fixed pixel offset when obstacle is detected.
- MPC tracks shifted line to naturally steer around obstacle.

<p align="left">
  <img src="media/gifs/alt_path.gif" alt="Alt Path Demo" width="800">
</p>

---

## Limitations
- High processing load on Jetson Nano when running vision, MPC, and ROS simultaneously.
- Sensitive to lighting conditions, lens distortion, and motion blur at high speeds.
- Latency in processing pipeline can affect steering response.

### Potential Solutions
- GPU acceleration for vision processing.
- Offload computation to external PC/server.
- Algorithmic optimizations (reduced search regions, periodic detection).

---

## Demonstrations

The PowerPoint version (`.pptm`) contains embedded videos showing:
- P vs PID controller performance.
- MPC speed and stability improvements.
- Obstacle detection distance measurement.
- Evasion manoeuvres for both approaches.

---

## Contact

**GitHub:** [github.com/jsmith-exe](https://github.com/jsmith-exe)  
![GitHub QR](https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=https://github.com/jsmith-exe)

**LinkedIn:** [linkedin.com/in/jamie-smith-916939371](https://www.linkedin.com/in/jamie-smith-916939371/)  
![LinkedIn QR](https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=https://www.linkedin.com/in/jamie-smith-916939371/)

---
