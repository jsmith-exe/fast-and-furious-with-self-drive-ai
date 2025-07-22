# Fast & Furious with Self Drive AI

**Author:** Jamie Smith  
**Date:** 2025-06-26

---

## 📖 Description

Fast & Furious with Self Drive AI is an autonomous driving system that combines camera (and optional LiDAR) input with a Model Predictive Control (MPC) bicycle model to achieve high-speed lane following and obstacle avoidance. Building on prior research that demonstrated safe but slow self-driving, this project aims to push the performance envelope by increasing speed while maintaining safety.

---

## 🗂️ Codebase Structure

- **controllers/**: Contains the control algorithms for the vehicle. The two main controllers are:
  - `PID.py`: Implements a Proportional-Integral-Derivative controller for basic motion control.
  - `MPC.py`: Implements a Model Predictive Control (MPC) bicycle model for advanced trajectory planning and control.

- **scripts/road_follower_scripts/road_follower.py**: This is the main JetRacer script that runs the self-driving pipeline.
  - It utilizes `road_follower_class.py` for the core logic and methods.
  - The trained neural network model used for road following is stored at `scripts/road_follower_scripts/trained_models/updated_model_trt.pth`.

This structure separates control logic from the main application, making it easier to maintain and extend the system.

---

## 🛠️ Installation

1. Clone the repo:  
   ```bash
   git clone https://gitlab.eeecs.qub.ac.uk/40401789/fast-and-furious-with-self-drive-ai.git
   cd fast-furious-self-drive-ai
   ```
2. Install the required packages:  
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the JetRacer hardware and software environment as per the [JetRacer setup guide](https://docs.nvidia.com/jetson/archives/l4t-archive-r34_1_1/index.html).

---

## 🚀 Usage

To start the self-driving demo, run the following command:  
```bash
python3 scripts/road_follower_scripts/road_follower.py
```

Ensure that the JetRacer is on a flat surface and has enough space to drive around before starting the demo.

---

## 📚 References

1. **Model Predictive Control**: For understanding the MPC algorithm used in this project, refer to [this paper](https://www.example.com).
2. **JetRacer Documentation**: For more details on the hardware and software setup, check the [official documentation](https://docs.nvidia.com/jetson/archives/l4t-archive-r34_1_1/index.html).
3. **JetRacer ROS AI Kit**: The hardware and ROS setup was based on the guide at [Waveshare JetRacer ROS AI Kit](https://www.waveshare.com/wiki/JetRacer_ROS_AI_Kit).
4. **Related Student Project**: Reference implementation and ideas from [Efficient Autonomous Obstacles Avoidance on JetRacer](https://gitlab.eeecs.qub.ac.uk/3048777/csc-3002-efficient-autonomous-obstacles-avoidance-on-jetracer-1-2025).
