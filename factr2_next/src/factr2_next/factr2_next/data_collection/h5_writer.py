from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


SCHEMA = "factr2_next_h5_v1"


class H5Writer:
    """Tiny writer for the public NEXT free-motion H5 schema."""

    def __init__(self, path, session_name, keys, metadata=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keys = list(keys)
        self.metadata = metadata
        self.last_stamp = None
        self.file = h5py.File(self.path, "w")
        self.file.attrs["schema"] = SCHEMA
        self.file.attrs["session_name"] = str(session_name)
        self.file.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        self.episode_index = -1
        self.episode = None

    def start_episode(self):
        self.episode_index += 1
        self.episode = self.file.create_group(f"ep_{self.episode_index:04d}")
        self.last_stamp = None
        if self.metadata is not None:
            self.metadata["episodes"][self.episode.name.rsplit("/", 1)[-1]] = {}
        return self.episode.name

    def append(self, timestamp_ns, samples):
        if self.episode is None:
            raise RuntimeError("start_episode() must be called before append().")

        missing = [key for key in self.keys if key not in samples]
        if missing:
            raise KeyError(f"Missing samples for keys: {missing}")

        # Validate every stream before resizing any dataset.
        prepared = {k: np.atleast_1d(np.asarray(samples[k], dtype=np.float32)) for k in self.keys}
        if self.metadata is not None:
            if any(a.shape != (7,) or not np.isfinite(a).all() for a in prepared.values()):
                raise ValueError("writer_values")
            if timestamp_ns <= 0 or (self.last_stamp is not None and timestamp_ns <= self.last_stamp):
                raise ValueError("writer_timestamp")
        for key in self.keys:
            group = self.episode.require_group(key)
            data = prepared[key]
            if data.ndim == 0:
                data = data.reshape(1)

            data_ds = self._dataset(group, "data", data.shape, data.dtype)
            time_ds = self._dataset(group, "timestamps", (), np.int64)
            row = data_ds.shape[0]
            data_ds.resize(row + 1, axis=0)
            time_ds.resize(row + 1, axis=0)
            data_ds[row] = data
            time_ds[row] = int(timestamp_ns)
        self.last_stamp = int(timestamp_ns)

    def close(self):
        if self.file is not None:
            if self.metadata is not None:
                for ep in self.file:
                    rows = len(self.file[ep][self.keys[0]]["data"]) if self.keys[0] in self.file[ep] else 0
                    self.metadata['episodes'][ep]['rows'] = rows
                    if rows < 50:
                        self.metadata['episodes'][ep]['excluded_reason'] = 'short_segment_less_than_history'
            self.file.flush()
            self.file.close()
            self.file = None
            if self.metadata is not None:
                from factr2_next.data_collection.quality import sha256, sidecar_path, write_json
                self.metadata['h5_sha256'] = sha256(self.path)
                write_json(sidecar_path(self.path), self.metadata)

    def flush(self):
        if self.file is not None:
            self.file.flush()

    def _dataset(self, group, name, sample_shape, dtype):
        if name in group:
            return group[name]
        shape = (0, *sample_shape)
        maxshape = (None, *sample_shape)
        chunks = (1, *sample_shape)
        return group.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            chunks=chunks,
            dtype=dtype,
        )
