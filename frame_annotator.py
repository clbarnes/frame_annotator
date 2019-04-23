#!/usr/bin/env python
import itertools
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import deque, defaultdict
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Deque, Tuple, Optional
import logging
from string import ascii_letters
import contextlib

import pandas as pd
import imageio
import numpy as np
import toml
from skimage.exposure import rescale_intensity

with contextlib.redirect_stdout(None):
    import pygame


logger = logging.getLogger(__name__)

here = Path(__file__).absolute()
project_dir = here.parent
config_path = project_dir / "config.toml"
default_config = toml.load(config_path)

DEFAULT_CACHE_SIZE = default_config["settings"]["cache"]
DEFAULT_FPS = default_config["settings"]["fps"]
DEFAULT_THREADS = default_config["settings"]["threads"]

DESCRIPTION = """
- Hold right or left to play the video at a reasonable (and configurable) FPS
- Hold Shift + arrow to attempt to play the video at 10x speed
- Press Ctrl + arrow to step through one frame at a time
- Press any letter key to mark the onset of an event, and Shift + that letter to mark the end of it
  - Marking the onset and end of an event at the same frame will remove both annotations
- Press Space to see which events are currently in progress
- Press Enter to see the table of results in the console
- Press Backspace to see the current frame number and contrast thresholds
- Ctrl + s to save
- Ctrl + h to show this message
""".rstrip()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LETTERS = set(ascii_letters)


class FrameAccessor:
    def __init__(self, fpath, **kwargs):
        self.logger = logger.getChild(type(self).__name__)
        self.lock = Lock()

        self.fpath = fpath
        with self.lock:
            self.reader = imageio.get_reader(fpath, mode='I', **kwargs)
            self.len = self.reader.get_length()
        self.logger.info("Detected %s frames", self.len)
        first = self[0]
        self.frame_shape = first.shape
        self.logger.info("Detected frames of shape %s", self.frame_shape)
        self.dtype = first.dtype
        self.logger.info("Detected frames of dtype %s (non-uint8 may be slower)", self.dtype)

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

        u8_info = np.iinfo('uint8')
        self.contrast_min = u8_info.min
        self.contrast_max = u8_info.max

        self.contrast_lower = self.contrast_min
        self.contrast_upper = self.contrast_max

        self.idx_in_cache = 0
        cache_size = min(cache_size, len(frames))

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.cache: Deque[Future] = deque(
            [self.fetch_frame(idx) for idx in range(cache_size)],
            cache_size
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
            zip(range(self.idx_in_cache, len(self.cache)), range(self.idx_in_cache-1, 0, -1))
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
            self.logger.debug("updating contrast to %s, %s", self.contrast_lower, self.contrast_upper)
            if freeze_cache:
                self.cache[self.idx_in_cache].cancel()
                self.cache[self.idx_in_cache] = self.fetch_frame(self.current_idx)
            else:
                self.renew_cache()
        return changed

    def from_uint8(self, arr):
        return arr

    def from_uint16(self, arr):
        out = (arr//256).astype('uint8')
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

        arr = self.apply_contrast(
            self.converter(arr), contrast_lower, contrast_upper
        )
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


def sort_key(start_stop_event):
    start, stop, event = start_stop_event
    if start is None:
        start = -np.inf
    if stop is None:
        stop = np.inf

    return start, stop, event


class EventLogger:
    def __init__(self, key_mapping=None):
        self.key_mapping = key_mapping or dict()
        self.events = defaultdict(set)

    def name(self, key):
        key = key.lower()
        return self.key_mapping.get(key, key)

    def keys(self):
        return {k.lower() for k in self.events}

    def starts(self):
        for k in self.keys():
            yield k, self.events[k]

    def stops(self):
        for k in self.keys():
            yield k, self.events[k.upper()]

    def delete(self, key, frame):
        self.events[key].discard(frame)

    def log(self, key, frame):
        self.delete(key.swapcase(), frame)
        self.events[key].add(frame)

    def is_active(self, key, frame):
        for start, stop in self.start_stop_pairs(key):
            if start is None:
                return stop > frame
            elif stop is None:
                return start <= frame

            if start <= frame:
                if frame < stop:
                    return True
            else:
                break

        return False

    def get_active(self, frame):
        for k in self.keys():
            if self.is_active(k, frame):
                yield k

    def start_stop_pairs(self, key):
        starts = self.events[key.lower()]
        stops = self.events[key.upper()]

        if not starts and not stops:
            return

        is_active = None

        try:
            first_start = min(starts)
        except ValueError:
            is_active = True

        try:
            first_stop = min(stops)
        except ValueError:
            is_active = False

        if is_active is None:
            is_active = first_stop < first_start

        last_start = None
        for f in range(max(starts | stops) + 1):
            if f in stops and is_active:
                yield last_start, f
                is_active = False
            elif f in starts and not is_active:
                last_start = f
                is_active = True

        if is_active and last_start is not None:
            yield last_start, None

    def to_df(self):
        rows = []
        for key in self.keys():
            for start, stop in self.start_stop_pairs(key):
                rows.append((start, stop, key, self.name(key)))

        return pd.DataFrame(
            sorted(rows, key=sort_key),
            columns=["start", "stop", "key", "event"],
            dtype=object
        )

    def save(self, fpath=None, **kwargs):
        if fpath is None:
            print(str(self))
        else:
            df = self.to_df()
            df.to_csv(fpath, **kwargs)

    def __str__(self):
        output = self.to_df()
        rows = [','.join(output.columns)]
        for row in output.itertuples(index=False):
            rows.append(','.join(row))
        return '\n'.join(rows)

    @classmethod
    def from_df(cls, df: pd.DataFrame, key_mapping=None):
        el = EventLogger()
        for start, stop, key, event in df.itertuples(index=False):
            el.key_mapping[key] = event
            el.events[event] = start
            el.events[event.upper()] = stop

        if key_mapping is not None:
            for k, v in key_mapping.items():
                existing = el.key_mapping.get(k)
                if existing is None:
                    el.key_mapping[k] = v
                elif existing != v:
                    raise ValueError("Given key mapping incompatible with given data")

    @classmethod
    def from_csv(cls, fpath, key_mapping=None):
        return cls.from_df(pd.read_csv(fpath), key_mapping)


class Window:
    def __init__(self, spooler: FrameSpooler, fps=DEFAULT_FPS, key_mapping=None, out_path=None):
        self.logger = logger.getChild(type(self).__name__)

        self.spooler = spooler
        self.fps = fps

        pygame.init()
        self.clock = pygame.time.Clock()
        first = self.spooler.current.result()
        self.im_surf: pygame.Surface = pygame.surfarray.make_surface(first.T)
        width, height = self.im_surf.get_size()
        self.screen = pygame.display.set_mode((width, height))
        self.im_surf.set_palette([(idx, idx, idx) for idx in range(256)])
        self.screen.blit(self.im_surf, (0, 0))
        pygame.display.update()

        self.events = EventLogger.from_csv(out_path, key_mapping) if out_path else EventLogger(key_mapping)
        self.out_path = out_path

    def step(self, step=0, force_update=False):
        if step or force_update:
            arr = self.spooler.step(step).result()

            self.draw_array(arr)

        self.clock.tick(self.fps)

    def active_events(self):
        yield from self.events.get_active(self.spooler.current_idx)

    def handle_events(self) -> Tuple[Optional[int], bool]:
        """
        Hold arrow: 1 in that direction
        Hold shift+arrow: 10 in that direction
        Press Ctrl+arrow: 1 in that direction
        Enter: print results
        lower-case letter: log event initiation / cancel event termination
        upper-case letter: log event termination / cancel event initiation

        :return:
        """
        while pygame.event.peek():
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                return None, False
            if event.type == pygame.KEYDOWN:
                if event.mod & pygame.KMOD_CTRL:
                    if event.key == pygame.K_RIGHT:
                        return 1, True
                    elif event.key == pygame.K_LEFT:
                        return -1, True
                    elif event.key == pygame.K_s:
                        self.save()
                    elif event.key == pygame.K_h:
                        self.print(DESCRIPTION)
                elif event.unicode in LETTERS:
                    self.events.log(event.unicode, self.spooler.current_idx)
                elif event.key == pygame.K_RETURN:
                    df = self.results()
                    self.print(df)
                elif event.key == pygame.K_SPACE:
                    self.print(f"Active events @ frame {self.spooler.current_idx}:\n\t{sorted(self.active_events())}")
                elif event.key == pygame.K_BACKSPACE:
                    self.print(
                        f"Frame {self.spooler.current_idx}, "
                        f"contrast = ({self.spooler.contrast_lower / 255:.02f}, "
                        f"{self.spooler.contrast_upper / 255:.02f})"
                    )
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    self.spooler.renew_cache()
        else:
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_LCTRL]:
                return 0, False
            speed = 10 if pressed[pygame.K_LSHIFT] else 1
            if pressed[pygame.K_RIGHT]:
                return speed, True
            if pressed[pygame.K_LEFT]:
                return -speed, True

            if self._handle_contrast(pressed):
                return 0, True

        return 0, False

    def _handle_contrast(self, pressed):
        mods = pygame.key.get_mods()
        if pressed[pygame.K_UP]:
            if mods & pygame.KMOD_SHIFT:
                self.spooler.update_contrast(upper=self.spooler.contrast_upper + 1, freeze_cache=True)
            else:
                self.spooler.update_contrast(lower=self.spooler.contrast_lower + 1, freeze_cache=True)
            return True
        elif pressed[pygame.K_DOWN]:
            if mods & pygame.KMOD_SHIFT:
                self.spooler.update_contrast(upper=self.spooler.contrast_upper - 1, freeze_cache=True)
            else:
                self.spooler.update_contrast(lower=self.spooler.contrast_lower - 1, freeze_cache=True)
            return True
        return False

    def print(self, *args, **kwargs):
        print_kwargs = {"file": sys.stderr, "flush": True}
        print_kwargs.update(**kwargs)
        print(*args, **print_kwargs)

    def results(self):
        return self.events.to_df()

    def save(self, fpath=None):
        self.events.save(fpath or self.out_path)

    def draw_array(self, arr):
        pygame.surfarray.blit_array(self.im_surf, arr.T)
        self.screen.blit(self.im_surf, (0, 0))

        pygame.display.update()

    def loop(self):
        while True:
            step_or_none, should_update = self.handle_events()
            if step_or_none is None:
                break
            self.step(step_or_none, should_update)

        return self.results()

    def close(self):
        self.spooler.close()
        pygame.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def parse_keys(s):
    d = dict()
    for pair in s.split(','):
        for event, key in pair.split('='):
            event = event.strip()
            key = key.strip().lower()
            if len(key) > 1:
                raise ValueError("keys must be 1 character long")
            d[key] = event
    return d


def parse_args():
    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--write_config", help="Write back the complete config to a file at this path")
    parser.add_argument("--outfile", "-o", help="Path to CSV for loading/saving")
    parser.add_argument("--config", help="Path to TOML file for config")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Maximum frames per second")
    parser.add_argument("--cache", type=int, default=DEFAULT_CACHE_SIZE, help="Approximately how many frames to cache")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="number of threads to use for reading file")
    parser.add_argument(
        "--keys", type=parse_keys, default=default_config["keys"],
        help='Mapping from event name to key, in the format "forward=w,left=a,back=s,right=d"'
    )
    parser.add_argument("infile", nargs='?', default=None, help="Path to multipage TIFF file to read")

    parsed = parser.parse_args()

    if parsed.config:
        config = toml.load(config_path)

        for key in ("fps", "cache", "threads"):
            if getattr(parsed, key) is None:
                setattr(
                    parsed, key,
                    config.get("settings", dict()).get(key, default_config["settings"][key])
                )
        config.get("keys", dict()).update(parsed.keys or default_config.getattr("keys"))
        parsed.keys = config["keys"]

    if parsed.write_config:
        d = {
            "settings": {
                "fps": parsed.fps,
                "cache": parsed.cache,
                "threads": parsed.threads
            }
        }
        if parsed.keys:
            d["keys"] = parsed.keys
        with open(parsed.write_config, 'w') as f:
            toml.dump(d, f)

    return parsed


def main(
    fpath,
    out_path=None,
    cache_size=DEFAULT_CACHE_SIZE,
    max_fps=DEFAULT_FPS,
    threads=DEFAULT_THREADS,
    keys=default_config["keys"]
):
    spooler = FrameSpooler(fpath, cache_size, max_workers=threads)
    with Window(spooler, max_fps, keys, out_path) as w:
        w.loop()
        w.save()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parsed_args = parse_args()
    if parsed_args.infile:
        main(
            parsed_args.infile,
            parsed_args.outfile,
            parsed_args.cache,
            parsed_args.fps,
            parsed_args.threads,
            parsed_args.keys
        )
