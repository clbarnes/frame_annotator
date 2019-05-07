import logging
import pandas as pd
import numpy as np

from fran.constants import FRAME


def parse_keys(s):
    d = dict()
    for pair in s.split(","):
        key, event = pair.split("=")
        event = event.strip()
        key = key.strip().lower()
        if len(key) > 1:
            raise ValueError("keys must be 1 character long")
        d[key] = event
    return d


def setup_logging(verbosity=0):
    verbosity = verbosity or 0
    logging.addLevelName(FRAME, "FRAME")
    levels = [logging.INFO, logging.DEBUG, FRAME, logging.NOTSET]
    v_idx = min(verbosity, len(levels) - 1)
    logging.basicConfig(level=levels[v_idx])


def load_results(fpath):
    df = pd.read_csv(fpath)
    df["note"] = [sanitise_note(item) for item in df["note"]]
    for col in ["start", "stop"]:
        df[col] = np.array([fn_or(item, int, None) for item in df[col]], dtype=object)
    return df


def dump_results(df: pd.DataFrame, fpath, **kwargs):
    df_kwargs = {"index": False}
    df_kwargs.update(kwargs)
    df.to_csv(fpath, **df_kwargs)


def df_to_str(df):
    rows = [",".join(df.columns)]
    for row in df.itertuples(index=False):
        rows.append(",".join(str(item) for item in row))
    return "\n".join(rows)


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
