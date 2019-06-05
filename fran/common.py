import logging

import pandas as pd

from fran.constants import FRAME_LEVEL


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
    logging.addLevelName(FRAME_LEVEL, "FRAME")
    levels = [logging.INFO, logging.DEBUG, FRAME_LEVEL, logging.NOTSET]
    v_idx = min(verbosity, len(levels) - 1)
    logging.basicConfig(level=levels[v_idx])


NAN_VALS = (
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
    "None",
)


def sanitise_note(item):
    try:
        if item not in NAN_VALS:
            return item.strip()
    except AttributeError:
        pass
    return ""


DTYPES = {
    "start": pd.Int64Dtype(),
    "stop": pd.Int64Dtype(),
    "key": str,
    "event": str,
    "note": sanitise_note,
}


def load_results(fpath):
    df = pd.read_csv(fpath, dtype=DTYPES, na_values=DTYPES)
    return df


def dump_results(df: pd.DataFrame, fpath=None, **kwargs):
    df_kwargs = {"index": False}
    df_kwargs.update(kwargs)
    return df.to_csv(fpath, **df_kwargs)


def df_to_str(df):
    return dump_results(df)


def fn_or(item, fn=int, default=None):
    try:
        return fn(item)
    except ValueError:
        return default
