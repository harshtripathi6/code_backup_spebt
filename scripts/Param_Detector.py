import numpy as np

# 1. Create the array of 80,000 zeros 
det = np.zeros(80000, dtype=np.float32)

# Total number of detectors (3 * 32 * 16) + (64 * 64) = 5632
total_bins = 5632
det[0] = total_bins

count = 0
current_y = 10.0  # Starting 10mm behind the collimator reference
gap_y = 2.0       # Added a small 2mm gap between layers (adjust as needed based on physical build)

# 3. Loop through the 4 Layers
for layer_id in range(4):
    if layer_id < 3: 
        # Layers 1, 2, and 3 (Mosaic layers)
        nx, nz = 32, 16
        # Transverse face: 3x3. Depth/Thickness: 6.
        dx, dy, dz = 3.0, 6.0, 3.0  
        # Spacing to spread the 32x16 crystals over a 128x128mm FOV
        pitch_x, pitch_z = 4.0, 8.0 
    else:
        # Layer 4 (High Resolution layer)
        nx, nz = 64, 64
        # Transverse face: 2x2. Depth/Thickness: 6.
        dx, dy, dz = 2.0, 6.0, 2.0  
        # Densely packed, so pitch equals dimension
        pitch_x, pitch_z = 2.0, 2.0 
    
    # Calculate the Y-center of the current layer
    center_y = current_y + (dy / 2.0)
    
    for r in range(nz):
        for c in range(nx):
            base = count * 12 + 1
            
            # Position centering using PITCH instead of dimensions
            pos_x = (c - (nx - 1) / 2.0) * pitch_x
            pos_z = (r - (nz - 1) / 2.0) * pitch_z
            
            det[base + 0] = pos_x       # x center
            det[base + 1] = center_y    # y center (depth)
            det[base + 2] = pos_z       # z center
            det[base + 3] = dx          # width
            det[base + 4] = dy          # thickness (depth)
            det[base + 5] = dz          # height
            
            det[base + 6] = 0.35        # total attenuation
            det[base + 7] = 0.30        # photo-electric
            det[base + 8] = 0.05        # compton
            det[base + 9] = 0.10        # energy resolution
            det[base + 10] = 0.0        
            det[base + 11] = 1.0        # active flag
            
            count += 1
            
    # Move the starting Y position for the next layer
    current_y += dy + gap_y

# 4. Save to the file
det.tofile("Params_Detector.dat")

print(f"Successfully created Params_Detector.dat")
print(f"Total Detectors: {count}")

