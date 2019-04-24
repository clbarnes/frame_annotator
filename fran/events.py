import itertools
from collections import defaultdict
from enum import Enum
from typing import NamedTuple, Dict, DefaultDict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def sort_key(start_stop_event):
    start, stop, key, event, note = start_stop_event
    if start is None:
        start = -np.inf
    if stop is None:
        stop = np.inf

    return start, stop, key, event, note


class Action(Enum):
    INSERT = "INSERT"
    DELETE = "DELETE"

    def invert(self):
        if self == Action.INSERT:
            return Action.DELETE
        else:
            return Action.INSERT

    def __str__(self):
        return self.value


class LoggedEvent(NamedTuple):
    action: Action
    key: str
    frame: int
    note: str = ""

    def invert(self):
        return LoggedEvent(self.action.invert(), self.key, self.frame, self.note)

    def __str__(self):
        return "{} {} @ {} ({})".format(*self)

    def copy(self, **kwargs):
        d = self._asdict()
        d.update(kwargs)
        return LoggedEvent(**d)


class EventLogger:
    def __init__(self, key_mapping=None):
        self.logger = logger.getChild(type(self).__name__)

        self.key_mapping: Dict[str, str] = key_mapping or dict()
        self.events: DefaultDict[str, Dict[int, str]] = defaultdict(dict)

        self.past: List[LoggedEvent] = []
        self.future: List[LoggedEvent] = []

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

    def _do(self, to_do: LoggedEvent):
        if to_do.action == Action.INSERT:
            return self._insert(to_do)
        else:
            return self._delete(to_do)

    def _insert(self, to_insert: LoggedEvent) -> LoggedEvent:
        note = to_insert.note or self.events[to_insert.key].get(to_insert.frame, "")
        to_insert.copy(action=Action.INSERT, note=note)
        self.events[to_insert.key][to_insert.frame] = note
        return to_insert

    def insert(self, key: str, frame: int, note=""):
        swapped = key.swapcase()
        if frame in self.events[swapped]:
            done = self.delete(swapped, frame)
        else:
            done = self._insert(LoggedEvent(Action.INSERT, key, frame, note))
            self.past.append(done)
            self.future.clear()
        self.logger.info("Logged %s", done)
        return done

    def _delete(self, to_do):
        if to_do.frame is None:
            return None
        note = self.events[to_do.key].pop(to_do.frame, None)
        if note is None:
            return None
        else:
            return to_do.copy(action=Action.DELETE, note=note)

    def delete(self, key, frame):
        done = self._delete(LoggedEvent(action=Action.DELETE, key=key, frame=frame))

        if done is None:
            self.logger.info("Nothing to delete")
            return None
        else:
            self.future.clear()
            self.past.append(done)
            self.logger.info("Logged %s", done)
            return done

    def undo(self):
        if not self.past:
            self.logger.info("Nothing to undo")
            return None
        to_undo = self.past.pop()
        done = self._do(to_undo.invert())
        self.future.append(done)

        self.logger.info("Undid %s", to_undo)

    def redo(self):
        if not self.future:
            self.logger.info("Nothing to redo")
            return None
        to_do = self.future.pop().invert()
        done = self._do(to_do)
        self.past.append(done)

        self.logger.info("Redid %s", done)

    def is_active(self, key, frame) -> Optional[Tuple[Optional[int], Optional[int]]]:
        """return start and stop indices, or None"""
        for start, stop in self.start_stop_pairs(key):
            if start is None:
                if stop > frame:
                    return start, stop
            elif stop is None:
                if start <= frame:
                    return start, stop
            elif start <= frame:
                if frame < stop:
                    return start, stop
            else:
                return None

        return None

    def get_active(self, frame):
        for k in self.keys():
            startstop = self.is_active(k, frame)
            if startstop:
                yield k, startstop

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
        for f in range(max(itertools.chain(starts.keys(), stops.keys())) + 1):
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
                rows.append(
                    (start, stop, key, self.name(key), self.events[key].get(start, ""))
                )

        return pd.DataFrame(
            sorted(rows, key=sort_key),
            columns=["start", "stop", "key", "event", "note"],
            dtype=object,
        )

    def save(self, fpath=None, **kwargs):
        if fpath is None:
            self.logger.info("Printing to stdout")
            print(str(self))
        else:
            df = self.to_df()
            df_kwargs = {"index": False}
            df_kwargs.update(kwargs)
            df.to_csv(fpath, **df_kwargs)
            self.logger.info("Saved to %s", fpath)

    def __str__(self):
        output = self.to_df()
        rows = [",".join(output.columns)]
        for row in output.itertuples(index=False):
            rows.append(",".join(str(item) for item in row))
        return "\n".join(rows)

    @classmethod
    def from_df(cls, df: pd.DataFrame, key_mapping=None):
        el = EventLogger()
        for start, stop, key, event, note in df.itertuples(index=False):
            el.key_mapping[key] = event
            if start is None:
                if note:
                    start = 0
                    el.events[key][start] = note
            else:
                el.events[key][start] = note
            if stop:
                el.events[key.upper()][stop] = ""

        if key_mapping is not None:
            for k, v in key_mapping.items():
                existing = el.key_mapping.get(k)
                if existing is None:
                    el.key_mapping[k] = v
                elif existing != v:
                    raise ValueError("Given key mapping incompatible with given data")
        return el

    @classmethod
    def from_csv(cls, fpath, key_mapping=None):
        df = pd.read_csv(fpath)
        df["note"] = [sanitise_note(item) for item in df["note"]]
        for col in ["start", "stop"]:
            df[col] = np.array(
                [fn_or(item, int, None) for item in df[col]], dtype=object
            )
        return cls.from_df(df, key_mapping)


def sanitise_note(item):
    try:
        if item.lower() != "nan":
            return item
    except AttributeError:
        pass
    return ""


def fn_or(item, fn=int, default=None):
    try:
        return fn(item)
    except ValueError:
        return default