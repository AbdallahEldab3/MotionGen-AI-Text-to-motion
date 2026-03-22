import bpy
import numpy as np
import os

input_path = "results.npy"

if not os.path.exists(input_path):
    print(f"file not found at {input_path}")
else:
    data = np.load(input_path, allow_pickle=True).item()
    motion = data['motion'][0] 
    num_joints = motion.shape[0]
    num_frames = motion.shape[2]
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    spheres = []
    for j in range(num_joints):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04)
        sphere = bpy.context.active_object
        sphere.name = f"Joint_{j:02d}"
        spheres.append(sphere)

    print(f"Standing up {num_frames} frames...")
    
    for f in range(num_frames):
        bpy.context.scene.frame_set(f)
        
        for j in range(num_joints):
            raw_x = motion[j, 0, f]
            raw_y = motion[j, 1, f]
            raw_z = motion[j, 2, f]
            x = raw_x
            y = -raw_z
            z = raw_y
            
            spheres[j].location = (x, y, z)
            spheres[j].keyframe_insert(data_path="location", frame=f)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames
    bpy.context.scene.frame_set(0)
