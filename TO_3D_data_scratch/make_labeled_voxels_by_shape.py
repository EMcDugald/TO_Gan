import numpy as np
from scipy.ndimage import label
import os

def compute_relative_internal_void_volume(voxel_arr):
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
    total_volume = np.prod(voxel_arr.shape)
    relative_void_volume = internal_void_volume / total_volume
    return relative_void_volume

def generate_voxels_for_shape(shape_tuple, n_samples, topo, shapes, threshold=0.0, outdir='./'):
    indices = [i for i, shp in enumerate(shapes) if tuple(shp) == shape_tuple]
    if len(indices) < n_samples:
        print(f"Warning: Requested {n_samples} samples for shape {shape_tuple}, but only {len(indices)} available. Using all available.")
        selected = indices
    else:
        selected = np.random.choice(indices, n_samples, replace=False)
    results = []
    for idx in selected:
        voxel_arr = topo[idx].astype(int).reshape(shape_tuple)
        rel_void = compute_relative_internal_void_volume(voxel_arr)
        label_val = 1 if rel_void <= threshold else 0
        results.append((voxel_arr, rel_void, label_val))
    fname = f'labeled_voxels_{shape_tuple[0]}x{shape_tuple[1]}x{shape_tuple[2]}.npy'
    np.save(os.path.join(outdir, fname), np.array(results, dtype=object))
    print(f"Saved {len(results)} labeled voxel structures of shape {shape_tuple} to {fname}")

if __name__ == "__main__":
    # Adjust these paths as needed
    topo = np.load('./data/topologies.npy', allow_pickle=True)
    shapes = np.load('./data/shapes.npy', allow_pickle=True)

    unique_shapes = set(tuple(shape) for shape in shapes)
    print("Unique shapes detected:", unique_shapes)

    # User can set this:
    # For a single shape, e.g. (64, 64, 64), set to that tuple
    # For all shapes, set to None
    #target_shape = None  # or e.g., (64, 64, 64)
    target_shape = (32,32,32)
    n_voxels_to_make = 100
    void_threshold = 0.0
    output_dir = './data/'

    if target_shape is None:
        for shape_tuple in unique_shapes:
            generate_voxels_for_shape(
                shape_tuple,
                n_voxels_to_make,
                topo,
                shapes,
                threshold=void_threshold,
                outdir=output_dir
            )
    else:
        if target_shape not in unique_shapes:
            print(f"Error: Target shape {target_shape} not found in your data. Available shapes are: {unique_shapes}")
        else:
            generate_voxels_for_shape(
                target_shape,
                n_voxels_to_make,
                topo,
                shapes,
                threshold=void_threshold,
                outdir=output_dir
            )
