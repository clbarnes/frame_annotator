import contextlib
import itertools
from collections import deque
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from functools import lru_cache
from queue import Queue
from threading import Lock
from typing import Deque
import logging

import imageio
import numpy as np
from skimage.exposure import rescale_intensity


logger = logging.getLogger(__name__)


class FrameAccessor:
    def __init__(self, fpath, **kwargs):
        self.logger = logger.getChild(type(self).__name__)
        self.lock = Lock()

        self.fpath = fpath
        with self.lock:
            self.reader = imageio.get_reader(fpath, mode="I", **kwargs)
            self.len = self.reader.get_length()
        self.logger.info("Detected %s frames", self.len)
        first = self[0]
        self.frame_shape = first.shape
        self.logger.info("Detected frames of shape %s", self.frame_shape)
        self.dtype = first.dtype
        self.logger.info(
            "Detected frames of dtype %s (non-uint8 may be slower)", self.dtype
        )

    def close(self):
        return self.reader.close()

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        with self.lock:
            return self.reader.get_data(item)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class FrameSpooler:
    def __init__(self, fpath, cache_size=100, max_workers=5, **kwargs):
        self.logger = logger.getChild(type(self).__name__)

        frames = FrameAccessor(fpath, **kwargs)
        self.frame_shape = frames.frame_shape
        self.frame_count = len(frames)

        try:
            self.converter = {
                np.dtype("uint8"): self.from_uint8,
                np.dtype("uint16"): self.from_uint16,
            }[frames.dtype]
        except KeyError:
            raise ValueError(f"Image data type not supported: {frames.dtype}")

        self.accessor_pool = Queue()
        self.accessor_pool.put(frames)
        for _ in range(max_workers - 1):
            self.accessor_pool.put(FrameAccessor(fpath, **kwargs))

        self.current_idx = 0

        self.pyg_size = self.frame_shape[1::-1]

        self.half_cache = cache_size // 2

        u8_info = np.iinfo("uint8")
        self.contrast_min = u8_info.min
        self.contrast_max = u8_info.max

        self.contrast_lower = self.contrast_min
        self.contrast_upper = self.contrast_max

        self.idx_in_cache = 0
        cache_size = min(cache_size, len(frames))

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.cache: Deque[Future] = deque(
            [self.fetch_frame(idx) for idx in range(cache_size)], cache_size
        )

    @contextlib.contextmanager
    def frames(self):
        accessor = self.accessor_pool.get(block=True)
        yield accessor
        self.accessor_pool.put(accessor)

    def cache_range(self):
        """in frame number"""
        start = max(self.current_idx - self.idx_in_cache, 0)
        stop = start + len(self.cache)
        return start, stop

    def frame_idx_to_cache_idx(self, frame_idx):
        return frame_idx - self.cache_range()[0]

    def cache_idx_to_frame_idx(self, cache_idx):
        return cache_idx + self.cache_range()[0]

    def renew_cache(self):
        self.logger.debug("renewing cache")
        # 0, +1, -1, +2, -2, +3, -3 etc.
        for idx in itertools.chain.from_iterable(
            zip(
                range(self.idx_in_cache, len(self.cache)),
                range(self.idx_in_cache - 1, 0, -1),
            )
        ):
            self.cache[idx].cancel()
            self.cache[idx] = self.fetch_frame(self.cache_idx_to_frame_idx(idx))

    def update_contrast(self, lower=None, upper=None, freeze_cache=False):
        changed = False

        if lower is not None:
            lower = max(self.contrast_min, lower)
            if lower != self.contrast_lower:
                self.contrast_lower = lower
                changed = True

        if upper is not None:
            upper = min(self.contrast_max, upper)
            if upper != self.contrast_upper:
                self.contrast_upper = upper
                changed = True

        if changed:
            self.logger.debug(
                "updating contrast to %s, %s", self.contrast_lower, self.contrast_upper
            )
            if freeze_cache:
                self.cache[self.idx_in_cache].cancel()
                self.cache[self.idx_in_cache] = self.fetch_frame(self.current_idx)
            else:
                self.renew_cache()
        return changed

    def from_uint8(self, arr):
        return arr

    def from_uint16(self, arr):
        out = (arr // 256).astype("uint8")
        return out

    @property
    def leftmost(self):
        return self.cache[0]

    @property
    def rightmost(self):
        return self.cache[-1]

    @property
    def current(self):
        return self.cache[self.idx_in_cache]

    def prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            if self.idx_in_cache > self.half_cache:
                self.idx_in_cache -= 1
            else:
                self.rightmost.cancel()
                idx = self.current_idx - self.idx_in_cache
                self.cache.appendleft(self.fetch_frame(idx))
        return self.current

    def next(self):
        if self.current_idx < self.frame_count - 1:
            self.current_idx += 1
            if self.idx_in_cache < self.half_cache:
                self.idx_in_cache += 1
            else:
                self.leftmost.cancel()
                idx = self.current_idx + self.idx_in_cache
                self.cache.append(self.fetch_frame(idx))
        return self.current

    def step(self, step=1):
        if not step:
            return self.current
        method = self.prev if step < 0 else self.next
        for _ in range(abs(step)):
            result = method()
        return result

    def fetch_frame(self, idx):
        if 0 <= idx < self.frame_count:
            f = self.executor.submit(
                self._fetch_frame, idx, self.contrast_lower, self.contrast_upper
            )
        else:
            f = Future()
            f.set_result(None)
        return f

    def apply_contrast(self, img, contrast_lower, contrast_upper):
        return rescale_intensity(img, (contrast_lower, contrast_upper))

    @lru_cache(maxsize=100)
    def _fetch_frame(self, idx, contrast_lower, contrast_upper):
        # todo: resize?
        with self.frames() as frames:
            arr = frames[idx]

        arr = self.apply_contrast(self.converter(arr), contrast_lower, contrast_upper)
        return arr

    def close(self):
        for f in self.cache:
            f.cancel()
        self.executor.shutdown()
        self.accessor_pool.put(None)
        while True:
            frames = self.accessor_pool.get()
            if frames is None:
                break
            frames.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
