"""Tests for the HDF5 store module."""

import numpy as np
import pytest

from plant_cpd.hdf5_store import HDF5Store


class TestHDF5Store:
    def test_save_load_pointcloud(self, tmp_h5, sphere_cloud):
        with HDF5Store(tmp_h5) as store:
            store.save_pointcloud("test/cloud", sphere_cloud)
            loaded = store.load_pointcloud("test/cloud")
        np.testing.assert_allclose(loaded, sphere_cloud)

    def test_save_load_scalar(self, tmp_h5):
        with HDF5Store(tmp_h5) as store:
            store.save_scalar("metrics/rmse", 0.0123)
            val = store.load_scalar("metrics/rmse")
        assert abs(val - 0.0123) < 1e-10

    def test_overwrite(self, tmp_h5):
        pts1 = np.ones((10, 3))
        pts2 = np.zeros((5, 3))
        with HDF5Store(tmp_h5) as store:
            store.save_pointcloud("data", pts1)
            store.save_pointcloud("data", pts2)
            loaded = store.load_pointcloud("data")
        assert loaded.shape == (5, 3)

    def test_save_dict(self, tmp_h5, sphere_cloud):
        with HDF5Store(tmp_h5) as store:
            store.save_dict("rigid", {
                "rotation": np.eye(3),
                "scale": 1.0,
            })
            rot = store.load_pointcloud("rigid/rotation")
            s = store.load_scalar("rigid/scale")
        np.testing.assert_allclose(rot, np.eye(3))
        assert abs(s - 1.0) < 1e-10

    def test_list_keys(self, tmp_h5):
        with HDF5Store(tmp_h5) as store:
            store.save_scalar("a/b", 1.0)
            store.save_scalar("a/c", 2.0)
            keys = store.list_keys("a")
        assert set(keys) == {"b", "c"}

    def test_context_manager_required(self, tmp_h5):
        store = HDF5Store(tmp_h5)
        with pytest.raises(RuntimeError, match="not open"):
            store.save_scalar("x", 1.0)

    def test_compression(self, tmp_h5, sphere_cloud):
        """Check that gzip compression is applied."""
        with HDF5Store(tmp_h5) as store:
            store.save_pointcloud("compressed", sphere_cloud)
            ds = store.file["compressed"]
            assert ds.compression == "gzip"