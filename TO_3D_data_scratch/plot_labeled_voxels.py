import numpy as np
import os
import imageio
import matplotlib.pyplot as plt

#os.makedirs(os.getcwd()+'/TO_3D_data_scratch/labeled_voxel_figs', exist_ok=True)
os.makedirs(os.getcwd()+'/TO_3D_data_scratch/labeled_voxel_figs_323232', exist_ok=True)

# Load labeled voxel data (tuples of (voxel_array, metric, label))
data = np.load(os.getcwd()+'/TO_3D_data_scratch/data/labeled_voxels_32x32x32.npy', allow_pickle=True)


for i, (voxel_arr, metric, label_val) in enumerate(data):
    label_str = 'positive' if label_val == 1 else 'negative'
    #base_filename = os.getcwd()+f'/TO_3D_data_scratch/labeled_voxel_figs/voxel_{i}_{label_str}'
    base_filename = os.getcwd()+f'/TO_3D_data_scratch/labeled_voxel_figs_323232/voxel_{i}_{label_str}'

    # Save GIF animation of slices
    images = []
    for slice_idx in range(voxel_arr.shape[0]):
        slice_img = (voxel_arr[slice_idx] * 255).astype(np.uint8)
        images.append(slice_img)
    gif_path = f'{base_filename}.gif'
    imageio.mimsave(gif_path, images, duration=1.0)

    # Save 3D voxel plot as PNG
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_arr, edgecolor='k')
    ax.set_title(f'Voxel {i} - {label_str} - Metric: {metric:.2f}')
    plt.axis('off')
    fig.savefig(f'{base_filename}.png')
    plt.close(fig)

print(f'Saved plots for {len(data)} labeled voxel structures in labeled_voxel_figs/')