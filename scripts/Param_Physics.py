import numpy as np
phy = np.zeros(100, dtype=np.float32)

phy[0] = 1      # flagUsingCompton
phy[1] = 1      # flagSavingPESysmat
phy[4] = 1      # flagUsingSameEnergyWindow
phy[5] = 112.0  # Lower threshold (from paper)
phy[6] = 168.0  # Upper threshold (from paper)
phy[7] = 140.0  # Target energy
phy[8] = 1      # Crystal calculation
phy[9] = 1      # Collimator calculation

phy.tofile("Params_Physics.dat")
print("Physics file created with 20% energy window.")