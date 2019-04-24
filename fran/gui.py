#!/usr/bin/env python
import sys
import os
from functools import wraps
from tkinter import filedialog
from typing import Tuple, Optional, Callable
import logging
from string import ascii_letters
import contextlib
import tkinter as tk

from fran.constants import (
    DEFAULT_CACHE_SIZE, DEFAULT_FPS, DEFAULT_THREADS, DEFAULT_FLIPX,
    DEFAULT_FLIPY, DEFAULT_ROTATE, CONTROLS,
    DEFAULT_KEYS
)
from fran.events import EventLogger
from fran.frames import FrameSpooler

with contextlib.redirect_stdout(None):
    import pygame


logger = logging.getLogger(__name__)

LETTERS = set(ascii_letters)

root_tk = tk.Tk()
root_tk.withdraw()


def noop(arg):
    return arg


class Window:
    def __init__(
        self,
        spooler: FrameSpooler,
        fps=DEFAULT_FPS,
        key_mapping=None,
        out_path=None,
        flipx=False,
        flipy=False,
        rotate=0,
    ):
        self.logger = logger.getChild(type(self).__name__)

        self.spooler = spooler
        self.fps = fps
        self.out_path = out_path

        pygame.init()
        self.clock = pygame.time.Clock()
        first = self.spooler.current.result()
        self.im_surf: pygame.Surface = pygame.surfarray.make_surface(first.T)
        self.im_surf.set_palette([(idx, idx, idx) for idx in range(256)])
        self.transformed_surf: Callable[[], pygame.Surface] = self._make_surf(
            flipx, flipy, rotate
        )
        width, height = self.transformed_surf().get_size()
        self.screen = pygame.display.set_mode((width, height))
        self._blit()
        pygame.display.update()

        if out_path and os.path.exists(out_path):
            self.events = EventLogger.from_csv(out_path, key_mapping)
        else:
            self.events = EventLogger(key_mapping)

    def _make_surf(self, flipx=False, flipy=False, rotate=0):
        def fn():
            surf = self.im_surf
            if flipx or flipy:
                surf = pygame.transform.flip(surf, flipx, flipy)
            if rotate % 360:
                surf = pygame.transform.rotate(surf, rotate)
            return surf

        return fn

    def _blit(self):
        self.screen.blit(self.transformed_surf(), (0, 0))

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
                    if event.key == pygame.K_RIGHT:  # step right
                        self.show_frame_info()
                        return 1, True
                    elif event.key == pygame.K_LEFT:  # step left
                        self.show_frame_info()
                        return -1, True
                    elif event.key == pygame.K_s:  # save
                        self.save()
                    elif event.key == pygame.K_h:  # help
                        self.print(CONTROLS)
                    elif event.key == pygame.K_z:  # undo
                        self.events.undo()
                    elif event.key == pygame.K_r:  # redo
                        self.events.redo()
                    elif event.key == pygame.K_n:  # note
                        self._handle_note()
                elif event.unicode in LETTERS:  # log event
                    self.events.insert(event.unicode, self.spooler.current_idx)
                elif event.key == pygame.K_RETURN:  # show results
                    df = self.results()
                    self.print(df)
                elif event.key == pygame.K_SPACE:  # show active events
                    self.print(
                        f"Active events @ frame {self.spooler.current_idx}:\n\t{self.get_actives_str()}"
                    )
                elif event.key == pygame.K_BACKSPACE:  # show frame info
                    self.show_frame_info()
                elif event.key == pygame.K_DELETE:  # delete a current event
                    self._handle_delete()
            elif event.type == pygame.KEYUP and event.key in (
                pygame.K_UP,
                pygame.K_DOWN,
            ):
                self.show_frame_info()
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

    def show_frame_info(self):
        self.print(
            f"Frame {self.spooler.current_idx}, "
            f"contrast = ({self.spooler.contrast_lower / 255:.02f}, "
            f"{self.spooler.contrast_upper / 255:.02f})"
        )

    @contextlib.contextmanager
    def _handle_event_in_progress(
        self, msg, auto=True
    ) -> Optional[Tuple[str, Tuple[int, int]]]:
        actives = sorted(self.active_events())
        if not actives:
            self.print("No events in progress")
            yield None
        elif len(actives) == 1 and auto:
            self.print(msg)
            k, (start, stop) = actives[0]
            self.print(f"\tAutomatically selecting only event, {k}: {start} -> {stop}")
            yield k, (start, stop)
        else:
            actives_str = "\n\t".join(
                f"{k}: {start} -> {stop}" for k, (start, stop) in actives
            )
            user_val = self.input(
                f"{msg} (press key and then enter; empty for none)\n\t{actives_str}\n> "
            )
            if user_val:
                actives_d = dict(actives)
                key = user_val.lower()
                if key in actives_d:
                    yield key, actives_d[key]
                else:
                    yield None
                    self.print(f"Event '{key}' not in progress")
            else:
                yield None

    def _handle_note(self):
        with self._handle_event_in_progress("Note for which event?") as k_startstop:
            if k_startstop:
                key, (start, stop) = k_startstop
                note = self.input("Enter note: ")
                self.events.insert(key, start or 0, note)

    def _handle_delete(self):
        with self._handle_event_in_progress(
            "Delete which event?", False
        ) as k_startstop:
            if k_startstop:
                key, (start, stop) = k_startstop
                self.events.delete(key, start)
                self.events.delete(key.upper(), stop)

    def get_actives_str(self):
        actives = sorted(self.active_events())
        return "\n\t".join(f"{k}: {start} -> {stop}" for k, (start, stop) in actives)

    def input(self, msg):
        self.print(msg, end="")
        return input().strip()

    def _handle_contrast(self, pressed):
        mods = pygame.key.get_mods()
        if pressed[pygame.K_UP]:
            if mods & pygame.KMOD_SHIFT:
                self.spooler.update_contrast(
                    upper=self.spooler.contrast_upper + 1, freeze_cache=True
                )
            else:
                self.spooler.update_contrast(
                    lower=self.spooler.contrast_lower + 1, freeze_cache=True
                )
            return True
        elif pressed[pygame.K_DOWN]:
            if mods & pygame.KMOD_SHIFT:
                self.spooler.update_contrast(
                    upper=self.spooler.contrast_upper - 1, freeze_cache=True
                )
            else:
                self.spooler.update_contrast(
                    lower=self.spooler.contrast_lower - 1, freeze_cache=True
                )
            return True
        return False

    @wraps(print)
    def print(self, *args, **kwargs):
        print_kwargs = {"file": sys.stderr, "flush": True}
        print_kwargs.update(**kwargs)
        print(*args, **print_kwargs)

    def results(self):
        return self.events.to_df()

    def save(self, fpath=None, ask=True):
        fpath = fpath or self.out_path
        if ask and not fpath:
            fpath = filedialog.asksaveasfilename(
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
            )

        if not fpath:
            fpath = None

        self.events.save(fpath)

    def draw_array(self, arr):
        pygame.surfarray.blit_array(self.im_surf, arr.T)
        self.screen.blit(self.transformed_surf(), (0, 0))

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


def run(
    fpath,
    out_path=None,
    cache_size=DEFAULT_CACHE_SIZE,
    max_fps=DEFAULT_FPS,
    threads=DEFAULT_THREADS,
    keys=DEFAULT_KEYS.copy(),
    flipx=DEFAULT_FLIPX,
    flipy=DEFAULT_FLIPY,
    rotate=DEFAULT_ROTATE,
):
    if not fpath:
        fpath = filedialog.askopenfilename(
            filetypes=(("TIFF files", "*.tif *.tiff"), ("All files", "*.*"))
        )
        if not fpath:
            logger.warning("No path given, exiting")
            return 0
    spooler = FrameSpooler(fpath, cache_size, max_workers=threads)
    with Window(spooler, max_fps, keys, out_path, flipx, flipy, rotate) as w:
        w.loop()
        w.save()

    return 0
