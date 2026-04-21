import numpy as np
import random

# 1. Create an array of 80,000 zeros
col = np.zeros(80000, dtype=np.float32)

# 2. Set numCollimatorLayers
col[0] = 1 

# 3. Set Metadata for Layer 0 (Indices: (0+1)*10 + offset)
id_layer = 0
base_layer = (id_layer + 1) * 10

# Corrected based on the paper's inner structure dimensions!
width_x = 140.0
height_z = 140.0
thickness_y = 6.0
radius = 0.8
num_holes = 1218  # Recalculated for 12.5% of a 140x140 area

col[base_layer + 0] = num_holes    # Number of holes
col[base_layer + 1] = width_x      # Width (X)
col[base_layer + 2] = thickness_y  # Thickness (Y)
col[base_layer + 3] = height_z     # Height (Z)
col[base_layer + 4] = 0.0          # Distance between 1st layer and this layer
col[base_layer + 5] = 4.0          # Total Attenuation (Tungsten at 140keV)
col[base_layer + 6] = 3.8          # PE Attenuation
col[base_layer + 7] = 0.2          # Compton Attenuation

# 4. Generate Random Holes with Collision Checking
print(f"Generating {num_holes} non-overlapping holes. This might take a few seconds...")
holes =[]
min_dist_sq = (2 * radius) ** 2  # Minimum distance squared between two hole centers

while len(holes) < num_holes:
    # Randomly place holes within the plate dimensions (padding by radius so they don't clip the edge)
    pos_x = random.uniform(-(width_x/2) + radius, (width_x/2) - radius)
    pos_z = random.uniform(-(height_z/2) + radius, (height_z/2) - radius)
    
    # Collision check: Ensure this hole doesn't overlap with any existing holes
    collision = False
    for (hx, hz) in holes:
        if (pos_x - hx)**2 + (pos_z - hz)**2 < min_dist_sq:
            collision = True
            break
            
    if not collision:
        holes.append((pos_x, pos_z))

# 5. Populate the array with the generated holes
for i, (pos_x, pos_z) in enumerate(holes):
    # Formula: id_Hole * 9 + 100
    base_hole = i * 9 + 100
    
    col[base_hole + 0] = pos_x       # x center
    col[base_hole + 1] = -thickness_y/2.0         # y1 center (start of hole)
    col[base_hole + 2] = thickness_y/2.0          # y2 center (end of hole)
    col[base_hole + 3] = pos_z       # z center
    col[base_hole + 4] = radius      # R (Radius)
    
    # Hole interior is air, so attenuation is 0
    col[base_hole + 5] = 0.0         # Total Attenuation
    col[base_hole + 6] = 0.0         # PE Attenuation
    col[base_hole + 7] = 0.0         # Compton Attenuation
    
    col[base_hole + 8] = 1.0         # Flag (1 = active)

# 6. Save with the name the C++ code expects
col.tofile("Params_Collimator.dat")

print(f"Successfully created Params_Collimator.dat with {len(holes)} holes.")