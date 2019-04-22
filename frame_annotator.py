#!/usr/bin/env python
import asyncio
import itertools
import sys
import warnings
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import deque, defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from io import StringIO, BytesIO
from typing import Deque, Tuple, Optional
import logging
from string import ascii_letters
import contextlib

import pandas as pd
import imageio
import numpy as np
from async_lru import alru_cache
from skimage.exposure import rescale_intensity

with contextlib.redirect_stdout(None):
    import pygame


DEFAULT_CACHE_SIZE = 500
DEFAULT_FPS = 30

DESCRIPTION = """
- Hold right or left to play the video at a reasonable (and configurable) FPS
- Hold Shift + arrow to attempt to play the video at 10x speed
- Press Ctrl + arrow to step through one frame at a time
- Press any letter key to mark the onset of an event, and Shift + that letter to mark the end of it
  - Marking the onset and end of an event at the same frame will remove both annotations
- Press Space to see which events are currently in progress
- Press Enter to see the table of results in the console
- Press Backspace to see the current frame number
""".rstrip()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LETTERS = set(ascii_letters)

logger = logging.getLogger(__name__)


class FrameAccessor:
    def __init__(self, fpath, **kwargs):
        self.logger = logger.getChild(type(self).__name__)

        self.fpath = fpath
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
        self.logger.debug(f"__getitem__(%s)", item)
        return self.reader.get_data(item)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()


class Result:
    def __init__(self, awa):
        self.awa = awa

    async def result(self):
        if not self.awa.done():
            await self.awa
        return self.awa.result()

    def cancel(self):
        if not self.awa.done():
            self.awa.cancel()


class FrameSpooler:
    def __init__(self, fpath, cache_size=100, max_workers=None, **kwargs):
        self.logger = logger.getChild(type(self).__name__)

        self.frames = FrameAccessor(fpath, **kwargs)

        try:
            self.converter = {
                np.dtype("uint8"): self.from_uint8,
                np.dtype("uint16"): self.from_uint16,
            }[self.frames.dtype]
        except KeyError:
            raise ValueError(f"Image data type not supported: {self.frames.dtype}")

        self.current_idx = 0

        fshape = self.frames.frame_shape

        self.mode = None
        if len(fshape) == 2:
            self.mode = 'P'
        elif len(fshape) == 3:
            if fshape[-1] == 3:
                self.mode = 'RGB'
            elif fshape[-1] == 4:
                self.mode = 'RGBX'
        if self.mode is None:
            raise RuntimeError("Could not infer image mode")

        self.logger.info("Using image mode %s", self.mode)

        self.pyg_size = self.frames.frame_shape[1::-1]

        self.half_cache = cache_size // 2

        u8_info = np.iinfo('uint8')
        self.contrast_min = u8_info.min
        self.contrast_max = u8_info.max

        self.contrast_lower = self.contrast_min
        self.contrast_upper = self.contrast_max

        self.idx_in_cache = 0
        cache_size = min(cache_size, len(self.frames))

        self.executor = ThreadPoolExecutor(max_workers=1)

        self.cache: Deque[Result] = deque(
            [self.fetch_frame(idx) for idx in range(cache_size)],
            cache_size
        )

        self.lock = asyncio.Lock()

        self.logger.debug("mode: %s", self.mode)
        self.logger.debug("pyg_size: %s", self.pyg_size)

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
        needs_renewing = False

        if lower is not None:
            lower = max(self.contrast_min, lower)
            if lower != self.contrast_lower:
                self.contrast_lower = lower
                needs_renewing = True and not freeze_cache

        if upper is not None:
            upper = min(self.contrast_max, upper)
            if upper != self.contrast_upper:
                self.contrast_upper = upper
                needs_renewing = True and not freeze_cache

        self.logger.debug("updating contrast to %s, %s", self.contrast_lower, self.contrast_upper)

        if needs_renewing:
            self.renew_cache()
        else:
            self.cache[self.idx_in_cache].cancel()
            self.cache[self.idx_in_cache] = self.fetch_frame(self.current_idx)
        return needs_renewing

    def from_uint8(self, arr):
        return arr

    def from_uint16(self, arr):
        # stretch = 39, 100
        # out = ((arr//256 - stretch[0]) * 2).astype('uint8')
        out = (arr//256).astype('uint8')
        self.logger.debug("Got img with range %s, %s", out.min(), out.max())
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
        if self.current_idx < len(self.frames) - 1:
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

    async def _noframe(self):
        return None

    def fetch_frame(self, idx):
        if 0 <= idx < len(self.frames):
            coro = self._fetch_frame(idx, self.contrast_lower, self.contrast_upper)
        else:
            coro = self._noframe()
        return Result(asyncio.create_task(coro))

    def apply_contrast(self, img, constrast_lower, contrast_upper):
        return rescale_intensity(img, (constrast_lower, contrast_upper))

    @alru_cache(maxsize=100)
    async def _fetch_frame(self, idx, contrast_lower, contrast_upper) -> bytes:
        # todo: resize?
        return self.apply_contrast(
            self.converter(self.frames[idx]), contrast_lower, contrast_upper
        ).tostring()

    def close(self):
        self.frames.close()
        # self._fetch_frame.close()

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
    def __init__(self):
        self.events = defaultdict(set)

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
        start_stop_event = []
        for key in self.keys():
            for start, stop in self.start_stop_pairs(key):
                start_stop_event.append((start, stop, key))

        return pd.DataFrame(
            sorted(start_stop_event, key=sort_key),
            columns=["start", "stop", "event"],
            dtype=object
        )


class PygameWindow:
    def __init__(self, size, fps=DEFAULT_FPS):
        pygame.init()
        self.size = size
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(size)
        self.surf: pygame.Surface = pygame.image.fromstring(
            np.zeros(self.size, dtype='uint8').tostring(), self.size, 'P'
        )
        self.surf.set_palette([(idx, idx, idx) for idx in range(256)])

    def draw(self):
        self.screen.fill(BLACK)
        self.screen.blit(self.surf, (0, 0))
        pygame.display.update()
        self.clock.tick(self.fps)
        return True

    def draw_array(self, arr: np.ndarray):
        if arr.shape != self.size[::-1] or arr.dtype != np.dtype('uint8'):
            warnings.warn(f"inappropriate array passed ({arr.shape}, {arr.dtype})")
            return False

        self.surf.get_buffer().write(arr.tostring())


        #     self.apply_contrast(
        #         self.converter(self.frames[idx]), contrast_lower, contrast_upper
        #     ).tostring(),
        #     self.pyg_size, self.mode
        # )
        return self.draw()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        pygame.quit()


class Window:
    def __init__(self, spooler: FrameSpooler, fps=DEFAULT_FPS):
        self.logger = logger.getChild(type(self).__name__)

        self.spooler = spooler
        self.fps = fps

        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(spooler.pyg_size)
        self.surf: pygame.Surface = pygame.image.fromstring(
            np.zeros(spooler.pyg_size, dtype='uint8').tostring(), spooler.pyg_size, 'P'
        )
        self.surf.set_palette([(idx, idx, idx) for idx in range(256)])

        self.events = EventLogger()

    async def step(self, step=0, force_update=False):
        if step or force_update:
            # old_frame = self.spooler.current_idx
            # async with self.spooler.lock:
            surf_bytes = await self.spooler.step(step).result()

            self.draw_bytes(surf_bytes)

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
                elif event.unicode in LETTERS:
                    self.events.log(event.unicode, self.spooler.current_idx)
                elif event.key == pygame.K_RETURN:
                    df = self.results()
                    self.print(df)
                elif event.key == pygame.K_SPACE:
                    self.print(f"Active events @ frame {self.spooler.current_idx}:\n\t{sorted(self.active_events())}")
                elif event.key == pygame.K_BACKSPACE:
                    self.print(f"Frame {self.spooler.current_idx}")
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

    def draw_bytes(self, b):
        self.surf.get_buffer().write(b)
        self.screen.fill(BLACK)
        self.screen.blit(self.surf, (0, 0))
        pygame.display.update()

    async def loop(self):
        self.draw_bytes(await self.spooler.current.result())

        while True:
            step_or_none, should_update = self.handle_events()
            if step_or_none is None:
                break
            await self.step(step_or_none, should_update)

        return self.results()

    def close(self):
        self.spooler.close()
        pygame.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def parse_args():
    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    parser.add_argument("infile", help="Path to multipage TIFF file to read")
    parser.add_argument("--outfile", "-o", help="Path to save a CSV to")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Maximum frames per second")
    parser.add_argument("--cache", type=int, default=DEFAULT_CACHE_SIZE, help="Approximately how many frames to cache")

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.DEBUG)
    parsed_args = parse_args()

    asyncio.run(main_loop(parsed_args.infile, parsed_args.outfile, parsed_args.cache, parsed_args.fps))


class TestFetcher:
    def __init__(self, fpath, max_workers=5):
        self.fa = FrameAccessor(fpath)
        self.exe = ThreadPoolExecutor(max_workers)

    def get_frame_fut(self, idx):
        return self.exe.submit(self.get_frame, idx)

    def get_frame(self, idx):
        return self.fa[idx]


def test():
    logging.basicConfig(level=logging.DEBUG)
    parsed_args = parse_args()

    fetcher = TestFetcher(parsed_args.infile, 1)
    print("submitting", flush=True)
    lst = [fetcher.get_frame_fut(idx) for idx in range(100)]
    print("submitted", flush=True)

    for idx, item in enumerate(lst):
        item.result()
        print(f"loaded {idx}", flush=True)

    # fa = FrameAccessor(parsed_args.infile)
    # frames = np.array([rescale_intensity((fa[idx]//256).astype('uint8')) for idx in range(100)], dtype='uint8')
    # size = fa.frame_shape[::-1]
    # with PygameWindow(size) as win:
    #     for frame in frames:
    #         win.draw_array(frame)


async def main_loop(fpath, out_path=None, cache_size=DEFAULT_CACHE_SIZE, max_fps=DEFAULT_FPS):
    spooler = FrameSpooler(fpath, cache_size)
    with Window(spooler, max_fps) as w:
        output = await w.loop()

    if out_path:
        output.to_csv(out_path)
    else:
        print(','.join(output.columns))
        for row in output.itertuples(index=False):
            print(','.join(row))


if __name__ == '__main__':
    # main()
    test()