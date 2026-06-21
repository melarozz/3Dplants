"""HDF5 storage for point clouds, registration results, and metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)


class HDF5Store:
    """Context-managed HDF5 result store.
    """

    def __init__(self, path: str | PathLike) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: h5py.File | None = None

    def __enter__(self) -> HDF5Store:
        self._file = h5py.File(self._path, "a")
        logger.debug("Opened HDF5 store: %s", self._path)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._file:
            self._file.close()
            logger.debug("Closed HDF5 store: %s", self._path)

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            msg = "Store is not open. Use as context manager."
            raise RuntimeError(msg)
        return self._file

    def save_pointcloud(
        self,
        key: str,
        points: np.ndarray,
    ) -> None:
        """Save a point cloud array under key.
        """
        if key in self.file:
            del self.file[key]
        self.file.create_dataset(
            key,
            data=points.astype(np.float64),
            compression="gzip",
            compression_opts=4,
        )
        logger.debug("Saved dataset '%s' shape=%s", key, points.shape)

    def load_pointcloud(self, key: str) -> np.ndarray:
        """Load a point cloud array from key.
        """
        return np.asarray(self.file[key])

    def save_scalar(self, key: str, value: float) -> None:
        """Save a scalar float value."""
        if key in self.file:
            del self.file[key]
        self.file.create_dataset(key, data=value)
        logger.debug("Saved scalar '%s' = %s", key, value)

    def load_scalar(self, key: str) -> float:
        return float(self.file[key][()])

    def save_dict(
        self,
        group: str,
        data: dict[str, Any],
    ) -> None:
        """Save a flat dict of arrays / scalars into a group.
        """
        for name, value in data.items():
            key = f"{group}/{name}"
            if isinstance(value, np.ndarray):
                self.save_pointcloud(key, value)
            else:
                self.save_scalar(key, float(value))

    def list_keys(self, group: str = "/") -> list[str]:
        """List all dataset keys under group recursively."""
        keys: list[str] = []
        self.file[group].visititems(
            lambda name, obj: keys.append(name)
            if isinstance(obj, h5py.Dataset)
            else None,
        )
        return keys