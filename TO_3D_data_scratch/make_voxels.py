import numpy as np
import os
import imageio
import matplotlib.pyplot as plt

os.makedirs('voxels', exist_ok=True)
os.makedirs('voxel_fig', exist_ok=True)

topo = np.load('data/topologies.npy',allow_pickle=True)
n_arrays = topo.shape[0]
shapes = np.load('data/shapes.npy',allow_pickle=True)

n_voxels_to_make = 10
indices = np.arange(n_arrays)
generated_indices = np.random.choice(indices,size=n_voxels_to_make,replace=False)

for idx in generated_indices:
    shape = tuple(shapes[idx])
    binary_arr = topo[idx].astype(int).flatten().reshape(shape)

    npy_path = f'voxels/voxel_{idx}.npy'
    np.save(npy_path, binary_arr)

    images = []
    for slice_idx in range(binary_arr.shape[0]): 
        slice_img = (binary_arr[slice_idx] * 255).astype(np.uint8)
        images.append(slice_img)
    
    gif_path = f'voxel_fig/voxel_{idx}.gif'
    imageio.mimsave(gif_path, images, duration=1.0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(binary_arr, edgecolor='k')
    ax.set_title(f'3D Voxel Visualization {idx}')
    fig.savefig(f'voxel_fig/voxel_{idx}_3d.png')
    plt.close(fig)