from __future__ import annotations

import h5py
import numpy as np


def save_plant_hdf5(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    ppfd: np.ndarray,
    metadata: dict,
):
    """
    Save mesh and PPFD into HDF5 along with metadata
    """
    with h5py.File(filepath, "w") as f:
        f.create_dataset("vertices", data=vertices)
        f.create_dataset("faces", data=faces)
        f.create_dataset("ppfd", data=ppfd)

        meta_grp = f.create_group("metadata")
        for key, value in metadata.items():
            if isinstance(value, (int, float, str)):
                meta_grp.attrs[key] = value
            elif isinstance(value, (list, tuple, np.ndarray)):
                meta_grp.create_dataset(key, data=value)