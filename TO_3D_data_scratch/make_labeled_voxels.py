import numpy as np
from scipy.ndimage import label
import os

def compute_internal_void_volume(voxel_arr):
    empty_voxels = np.logical_not(voxel_arr)
    labeled_voids, num_features = label(empty_voxels)
    border_labels = set()
    border_labels.update(np.unique(labeled_voids[0, :, :]))
    border_labels.update(np.unique(labeled_voids[-1, :, :]))
    border_labels.update(np.unique(labeled_voids[:, 0, :]))
    border_labels.update(np.unique(labeled_voids[:, -1, :]))
    border_labels.update(np.unique(labeled_voids[:, :, 0]))
    border_labels.update(np.unique(labeled_voids[:, :, -1]))
    border_labels.discard(0)
    internal_void_volume = 0
    for void_label in range(1, num_features + 1):
        if void_label not in border_labels:
            internal_void_volume += np.sum(labeled_voids == void_label)
    return internal_void_volume

topo = np.load(os.getcwd()+'/TO_3D_data_scratch/data/topologies.npy',allow_pickle=True)
shapes = np.load(os.getcwd()+'/TO_3D_data_scratch/data/shapes.npy',allow_pickle=True)

n_arrays = topo.shape[0]
n_voxels_to_make = 10
indices = np.arange(n_arrays)
selected_indices = np.random.choice(indices, size=n_voxels_to_make, replace=False)

voxel_metric_label_tuples = []

threshold = 50  # Example threshold for internal void volume

for idx in selected_indices:
    shape = tuple(shapes[idx])
    voxel_arr = topo[idx].astype(int).flatten().reshape(shape)
    metric_val = compute_internal_void_volume(voxel_arr)
    print("metric val",metric_val)
    label_val = 1 if metric_val <= threshold else 0
    print("label val",label_val)
    voxel_metric_label_tuples.append((voxel_arr, metric_val, label_val))

# Save as object dtype array to handle mixed shapes and types
np.save(os.getcwd()+'/TO_3D_data_scratch/data/labeled_voxels.npy', np.array(voxel_metric_label_tuples, dtype=object))

print(f'Saved {len(voxel_metric_label_tuples)} labeled voxel structures to labeled_voxels.npy')