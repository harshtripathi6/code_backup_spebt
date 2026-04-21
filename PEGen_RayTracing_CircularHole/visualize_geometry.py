import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_block(ax, center, size, color, alpha=0.8, edge=True):
    """Draws a 3D block (crystal/FOV/plate)"""
    x, y, z = center
    dx, dy, dz = size
    xx = [x-dx/2, x+dx/2]; yy =[y-dy/2, y+dy/2]; zz =[z-dz/2, z+dz/2]
    faces = [
        [[xx[0],yy[0],zz[0]],[xx[1],yy[0],zz[0]], [xx[1],yy[1],zz[0]], [xx[0],yy[1],zz[0]]],
        [[xx[0],yy[0],zz[1]], [xx[1],yy[0],zz[1]], [xx[1],yy[1],zz[1]], [xx[0],yy[1],zz[1]]],
        [[xx[0],yy[0],zz[0]], [xx[1],yy[0],zz[0]], [xx[1],yy[0],zz[1]], [xx[0],yy[0],zz[1]]],
        [[xx[0],yy[1],zz[0]], [xx[1],yy[1],zz[0]], [xx[1],yy[1],zz[1]],[xx[0],yy[1],zz[1]]],
        [[xx[0],yy[0],zz[0]], [xx[0],yy[1],zz[0]], [xx[0],yy[1],zz[1]], [xx[0],yy[0],zz[1]]],
        [[xx[1],yy[0],zz[0]], [xx[1],yy[1],zz[0]],[xx[1],yy[1],zz[1]], [xx[1],yy[0],zz[1]]]
    ]
    ec = 'black' if edge else 'none'
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors=ec, alpha=alpha, linewidths=0.1))

def generate_final_report():
    # 1. LOAD ALL DATA DYNAMICALLY
    det_raw = np.fromfile("Params_Detector.dat", dtype=np.float32)
    col_raw = np.fromfile("Params_Collimator.dat", dtype=np.float32)
    img_raw = np.fromfile("Params_Image.dat", dtype=np.float32)
    
    # Parse Detector Array
    total_bins = int(det_raw[0])
    det_data = det_raw[1:1 + total_bins*12].reshape(-1, 12)
    
    # Parse Collimator Array
    num_holes = int(col_raw[10])
    col_w, col_t, col_h = col_raw[11], col_raw[12], col_raw[13]
    hole_data = col_raw[100:100 + num_holes*9].reshape(-1, 9)
    
    # Parse Image/FOV Array
    fov_wx = img_raw[0] * img_raw[3]
    fov_wy = img_raw[1] * img_raw[4]
    fov_wz = img_raw[2] * img_raw[5]
    fov_dist = img_raw[11]

    fig = plt.figure(figsize=(26, 12))
    layer_colors =['#aec7e8', '#7fb3d5', '#1f77b4', '#08306b']
    
    # Indices for 3 layers of 32x16 (512 each) and 1 layer of 64x64 (4096)
    indices =[0, 512, 1024, 1536, total_bins]

    # ---------------------------------------------------------
    # PANEL 1: SIDE VIEW (Detailed Shapes)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(131)
    ax1.set_title("1. SIDE VIEW (Y-Z Plane Projection)", fontsize=14, fontweight='bold')
    
    # Draw FOV
    ax1.add_patch(Rectangle((-fov_dist-fov_wy/2, -fov_wz/2), fov_wy, fov_wz, color='green', alpha=0.3, label='FOV Volume'))
    
    # Draw Collimator Plate
    ax1.add_patch(Rectangle((-col_t/2, -col_h/2), col_t, col_h, color='grey', alpha=0.5, label='Tungsten Plate'))
    
    # Draw Actual Holes as horizontal lines passing through the plate
    for h in hole_data[::5]:  # Plot every 5th hole for visual clarity
        ax1.plot([-col_t/2, col_t/2], [h[3], h[3]], color='white', linewidth=1.0, alpha=0.8)

    # Draw Detector Crystals
    for i in range(4):
        layer = det_data[indices[i]:indices[i+1]]
        sample = layer[::max(1, len(layer)//200)] # Sample for clarity
        ax1.scatter(sample[:, 1], sample[:, 2], s=12, marker='s', color=layer_colors[i], edgecolors='black', linewidths=0.3, label=f'Layer {i+1}')

    ax1.set_xlabel("Y - Depth/Distance from Collimator Face (mm)")
    ax1.set_ylabel("Z - Transverse Height (mm)")
    #ax1.set_xlim(-fov_dist - 20, np.max(det_data[:,1]) + 20)
    ax1.set_xlim(-fov_dist - fov_wy/2 - 20, np.max(det_data[:,1]) + 20)
    ax1.set_ylim(-100, 100)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=10)

    # ---------------------------------------------------------
    # PANEL 2: LITERAL 3D ASSEMBLY (Physical Reality)
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title("2. LITERAL 3D ASSEMBLY", fontsize=14, fontweight='bold')
    
    draw_block(ax2,[0, -fov_dist, 0],[fov_wx, fov_wy, fov_wz], 'green', alpha=0.1)
    draw_block(ax2,[0, 0, 0], [col_w, col_t, col_h], 'grey', alpha=0.2)
    
    for i in range(4):
        layer = det_data[indices[i]:indices[i+1]]
        step = 15 if i < 3 else 80
        for d in layer[::step]:
            draw_block(ax2, [d[0], d[1], d[2]], [d[3], d[4], d[5]], layer_colors[i], alpha=0.6)
            
    ax2.set_xlabel('X (Width)'); ax2.set_ylabel('Y (Depth)'); ax2.set_zlabel('Z (Height)')
    #ax2.set_xlim(-80, 80); ax2.set_ylim(-110, 50); ax2.set_zlim(-80, 80)
    ax2.set_xlim(-80, 80); ax2.set_ylim(-fov_dist - fov_wy/2 - 20, np.max(det_data[:,1]) + 20); ax2.set_zlim(-80, 80)
    ax2.view_init(elev=20, azim=-45)

    # ---------------------------------------------------------
    # PANEL 3: EXPLODED VIEW (The Design Anatomy)
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title("3. EXPLODED VIEW", fontsize=14, fontweight='bold')
    
    z_exp = [150, 100, 50, 0] 
    label_x = -150

    # Draw FOV at the top
    draw_block(ax3,[0, 0, 250], [fov_wx, fov_wz, fov_wy], 'green', alpha=0.15)
    ax3.text(label_x, 0, 250, "FOV Target", color='green', fontweight='bold', ha='right')

    # Draw Collimator Plate with actual hole positions
    draw_block(ax3,[0, 0, 200], [col_w, col_h, col_t], 'grey', alpha=0.25)
    ax3.scatter(hole_data[:, 0], hole_data[:, 3], [200]*num_holes, s=0.3, color='black', alpha=0.5)
    ax3.text(label_x, 0, 200, f"COLLIMATOR\n({num_holes} holes)", color='black', fontweight='bold', ha='right')

    # Draw Detector Layers
    for i in range(4):
        layer = det_data[indices[i]:indices[i+1]]
        step = 10 if i < 3 else 50
        for d in layer[::step]:
            draw_block(ax3, [d[0], d[2], z_exp[i]], [d[3], d[5], d[4]], layer_colors[i])
        ax3.text(label_x, 0, z_exp[i], f"Layer {i+1}", color=layer_colors[i], fontweight='bold', ha='right')

    ax3.set_axis_off()
    ax3.set_box_aspect((1, 1, 1.8))
    ax3.view_init(elev=15, azim=30)

    plt.suptitle(f"SC-SPECT System Hardware Verification Report", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig("final_scientific_report.png", dpi=300)
    print("Success! 'final_scientific_report.png' created.")
    plt.show()

generate_final_report()