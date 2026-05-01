# A Bottom-Up Multi-Cue Detection for Gun Holder

## Overview

This project presents a research-driven computer vision system designed to identify the person holding a gun in images or video streams. Unlike direct end-to-end approaches, this system follows a **bottom-up multi-cue strategy**, where humans and guns are detected separately and then intelligently associated using multiple visual cues.

The objective is to improve reliability in complex real-world scenes where multiple people are present, partial occlusions exist, or direct holder classification is difficult.

---

## Key Idea

Instead of simply detecting a gun holder directly, the system performs:

1. **Human Detection**
2. **Gun Detection**
3. **Multi-Cue Association**
4. **Holder Identification**
5. **Tracking Across Frames**

This bottom-up pipeline improves interpretability and modularity.

---

## Multi-Cue Logic Used

The system combines several cues to determine the true holder:

- **IoU Overlap Prioritization**  
  Measures overlap between gun and person bounding boxes.

- **Pose Keypoint Validation**  
  Uses shoulder → elbow → wrist alignment to verify realistic holding posture.

- **Distance / Depth Reasoning**  
  Selects nearest plausible human when overlap is weak.

- **Final IoU Heuristic**  
  Backup selection logic for ambiguous scenes.

- **Tracking Across Frames**  
  Maintains identity of the detected holder in videos.

---

## Features

- Detects multiple humans and guns independently
- Identifies actual gun holder using layered reasoning
- Works in crowded scenes
- Supports image and video input
- Tracks holder frame-by-frame
- Modular architecture for future upgrades

---

## Tech Stack

- Python
- OpenCV
- YOLO / Object Detection Models
- Pose Estimation
- NumPy
- Deep Learning Frameworks

---

