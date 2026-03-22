***

# MotionGen-AI Diffusion Model - Full Implementation Guide

This guide is the result of troubleshooting every dependency failure in the MDM pipeline, specifically for **Local Windows** and **Colab** environments.

## 1. The Mandatory File Setup (Do This First)
The code will crash unless these files are manually placed in your project root. Do not rely on scripts; verify these exist:

### **A. SMPL Body Models (The "Geometry" Fix)**
You must have the SMPL neutral model. Without it, the `rot2xyz` transformation (converting rotations to 3D points) will fail.
* **Path:** `body_models/smpl/SMPL_NEUTRAL.pkl`
* **Where to get it:** Download from the [SMPL website](https://smpl.is.tue.mpg.de/) (requires account) or copy from your previous research drive.

### **B. T2M Evaluator (The "Referee" Fix)**
This is required to calculate ADE/FDE and FID scores.
* **Path:** `t2m/text_mot_match/model/finest.tar`
* **Status:** This folder **must** be in your root. If it is missing, the evaluator cannot "judge" your motion.

### **C. GloVe Text Embeddings**
* **Path:** `glove/glove.6B.300d.txt`
* **Note:** If you only have the `.zip` or `.gz`, you **must** extract it so the raw `.txt` is visible.

---

## 2. The "Clean" Environment Setup
To stop the **NumPy 2.0 / Chumpy** errors, run this exact block at the top of your notebook or as a script. It "tricks" the old libraries into working with new Python versions.

```python
import numpy as np
import os

# LEGACY PATCH: Fixes 'ImportError: cannot import name float_ from numpy'
def patch_numpy():
    for attr in ['float', 'int', 'bool', 'complex', 'object', 'float_', 'int_']:
        if not hasattr(np, attr):
            setattr(np, attr, float if 'float' in attr else int if 'int' in attr else bool)
    print("NumPy Legacy Patch Applied.")

patch_numpy()

# Stop ALSA/Audio errors on headless servers
os.environ['SDL_VIDEODRIVER'] = 'dummy'
```

---

## 3. Local Training & Evaluation
Use these commands in your terminal. We use `TensorboardPlatform` to ensure your logs are saved locally as `.tfevents` files for plotting.

### **Training**
```powershell
python -m train.train_mdm --save_dir save/run_v15 --dataset humanml --batch_size 32 --train_platform_type TensorboardPlatform --overwrite
```

### **Evaluation (Calculating ADE/FDE)**
```powershell
python -m eval.eval_humanml --model_path .\save\run_v15\model000020000.pt --dataset humanml
```

---

## 4. The Matplotlib Visualizer
Run this Python code to see your progress without needing an internet connection or WandB.

```python
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

# Finds the most recent log in your save folder
event_file = glob.glob("save/run_v15/events.out.tfevents.*")[-1]
ea = EventAccumulator(event_file)
ea.Reload()

# Plot Loss
steps = [x.step for x in ea.Scalars('loss')]
values = [x.value for x in ea.Scalars('loss')]

plt.figure(figsize=(10, 5))
plt.plot(steps, values, color='blue', label='Diffusion Loss')
plt.title('Training Progress: Loss vs Steps')
plt.xlabel('Steps'); plt.ylabel('MSE'); plt.legend()
plt.savefig('docs/loss_plot.png')
plt.show()
```

---

## 5. Blender Visualization Code
To see your model's motion in 3D, open Blender's Scripting tab and use this:

```python
import bpy
import numpy as np

# Load the generated motion .npy file
# Path to your generated file
data = np.load(r"C:\Path\To\Your\Generated_Motion.npy", allow_pickle=True).item()
motion = data['motion']  # Shape usually [22, 3, Frames]

def build_skeleton(frame_idx):
    for i in range(motion.shape[0]):
        x, y, z = motion[i, :, frame_idx]
        # Create a sphere for each joint
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(x, y, z))
        
# Clear scene and build
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
build_skeleton(0)
```
---
## Demo

