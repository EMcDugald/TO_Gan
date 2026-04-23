import numpy as np
import os

def generate_unconditional_voxels_for_shape(
    shape_tuple,
    n_samples,
    topo,
    shapes,
    outdir="./",
):
    indices = [i for i, shp in enumerate(shapes) if tuple(shp) == shape_tuple]
    if len(indices) < n_samples:
        print(
            f"Warning: Requested {n_samples} samples for shape {shape_tuple}, "
            f"but only {len(indices)} available. Using all available."
        )
        selected = indices
    else:
        selected = np.random.choice(indices, n_samples, replace=False)

    voxels = []
    for idx in selected:
        voxel_arr = topo[idx].astype(np.float32).reshape(shape_tuple)
        voxels.append(voxel_arr)

    X = np.stack(voxels).astype(np.float32)

    fname = (
        f"{len(X)}_uncond_voxels_"
        f"{shape_tuple[0]}x{shape_tuple[1]}x{shape_tuple[2]}.npy"
    )
    data_path = os.path.join(outdir, fname)
    np.save(data_path, X)

    print(f"Saved {len(X)} unconditional voxel structures of shape {shape_tuple} to {fname}")


if __name__ == "__main__":
    data_root = "./data"
    topo = np.load(os.path.join(data_root, "topologies.npy"), allow_pickle=True)
    shapes = np.load(os.path.join(data_root, "shapes.npy"), allow_pickle=True)

    unique_shapes = set(tuple(shape) for shape in shapes)
    print("Unique shapes detected:", unique_shapes)

    target_shape = (32, 32, 32)
    n_voxels_to_make = 5000

    output_dir = "/xdisk/hdb/emcdugald/simple_gan/train_data/323232"
    os.makedirs(output_dir, exist_ok=True)

    if target_shape is None:
        for shape_tuple in unique_shapes:
            generate_unconditional_voxels_for_shape(
                shape_tuple,
                n_voxels_to_make,
                topo,
                shapes,
                outdir=output_dir,
            )
    else:
        if target_shape not in unique_shapes:
            print(
                f"Error: Target shape {target_shape} not found in your data. "
                f"Available shapes are: {unique_shapes}"
            )
        else:
            generate_unconditional_voxels_for_shape(
                target_shape,
                n_voxels_to_make,
                topo,
                shapes,
                outdir=output_dir,
            )