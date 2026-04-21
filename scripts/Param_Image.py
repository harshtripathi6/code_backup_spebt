import numpy as np

# 1. Create an array of 100 zeros (matching the C++ float* parameter_Image = new float[100])
img = np.zeros(100, dtype=np.float32)

# 2. Fill in the specific values according to the paper's index list
img[0] = 128         # numImageVoxelX
img[1] = 128           # numImageVoxelY
img[2] = 128          # numImageVoxelZ
img[3] = 1          # widthImageVoxelX (mm)
img[4] = 1         # widthImageVoxelY (mm)
img[5] = 1          # widthImageVoxelZ (mm)
img[6] = 1           # numRotation
img[7] = 0.10471976   # anglePerRotation (6 degrees in radians)
img[8] = 0.0          # shiftFOVX
img[9] = 0.0          # shiftFOVY
img[10] = 0.0         # shiftFOVZ
img[11] = 67.0       # FOV2Collimator0

# 3. Save it with the EXACT name the C++ code expects (Params_Image.dat)
img.tofile("Params_Image.dat")

print("Successfully created Params_Image.dat")