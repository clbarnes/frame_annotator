from __future__ import annotations
import logging
import sys
from argparse import ArgumentParser, RawTextHelpFormatter, Namespace
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Dict, Any

import toml

from fran import __version__
from fran.common import parse_keys, setup_logging
from fran.gui import run
from fran.constants import (
    CONTROLS,
    DEFAULT_FPS,
    DEFAULT_CACHE_SIZE,
    DEFAULT_THREADS,
    default_config_dict,
)


logger = logging.getLogger(__name__)


@dataclass
class Config:
    infile: Optional[Path] = None
    write_config: Optional[Path] = None
    outfile: Optional[Path] = None
    config: Optional[Path] = None
    fps: Optional[float] = None
    cache: Optional[int] = None
    threads: Optional[int] = None
    keys: Dict[str, str] = field(default_factory=dict)
    flipx: Optional[bool] = None
    flipy: Optional[bool] = None
    rotate: Optional[float] = None
    version: Optional[bool] = None
    verbose: Optional[int] = None
    logfile: Optional[Path] = None

    def replace_from_namespace(self, ns: Namespace) -> Config:
        return self.replace(**{k: v for k, v in vars(ns).items() if v is not None})

    def replace(self, keys=None, **kwargs) -> Config:
        """N.B. 'keys' is an update, not a replacement"""
        if keys:
            union = self.keys.copy()
            union.update(keys)
        else:
            union = self.keys

        return replace(self, keys=union, **kwargs)

    def replace_from_config_dict(self, d: Dict[str, Dict[str, Any]]) -> Config:
        flat = d.get("settings", dict()).copy()
        flat.update(d.get("transform", dict()).copy())
        flat["keys"] = d.get("keys", dict()).copy()
        return self.replace(**flat)

    def replace_from_toml(self, fpath) -> Config:
        return self.replace_from_config_dict(toml.load(fpath))

    def write_toml(self, fpath):
        d = {
            "settings": {"fps": self.fps, "cache": self.cache, "threads": self.threads},
            "transform": {
                "flipx": self.flipx,
                "flipy": self.flipy,
                "rotate": self.rotate,
            },
            "keys": self.keys,
        }
        with open(fpath, "w") as f:
            toml.dump(d, f)


def parse_args(args=None) -> Namespace:
    parser = ArgumentParser(
        description="Log video (multipage TIFF) frames in which an event starts or ends",
        epilog=CONTROLS,
        prog="fran",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--write_config",
        "-w",
        type=Path,
        help="Write back the complete config to a file at this path, then exit",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        type=Path,
        help="Path to CSV for loading/saving. "
        "If no path is selected when you save, a file dialog will open.",
    )
    parser.add_argument("--config", "-c", help="Path to TOML file for config")
    parser.add_argument(
        "--fps",
        "-f",
        type=float,
        default=None,
        help=f"Maximum frames per second; default {DEFAULT_FPS}",
    )
    parser.add_argument(
        "--cache",
        "-n",
        type=int,
        default=None,
        help=f"Number of frames to cache (increase if reading over a network and you have lots of RAM); default {DEFAULT_CACHE_SIZE}",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=None,
        help=f"number of threads to use for reading file (increase if reading over a network); default {DEFAULT_THREADS}",
    )
    parser.add_argument(
        "--keys",
        "-k",
        type=parse_keys,
        default=dict(),
        help='Optional mappings from event name to key, in the format "w=forward,a=left,s=back,d=right". '
        "These are additive with those defined in the config",
    )
    parser.add_argument(
        "--flipx", "-x", action="store_true", default=None, help="Flip image in x"
    )
    parser.add_argument(
        "--flipy", "-y", action="store_true", default=None, help="Flip image in y"
    )
    parser.add_argument(
        "--rotate",
        "-r",
        type=float,
        default=None,
        help="Rotate image (degrees counterclockwise; applied after flipping)",
    )
    parser.add_argument(
        "--version", action="store_true", help="Print the version and then exit"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        help="Increase verbosity of logging (can be repeated). One for DEBUG, two for FRAME.",
    )
    parser.add_argument("--logfile", "-l", help="File to save log to")
    parser.add_argument(
        "infile",
        nargs="?",
        type=Path,
        default=None,
        help="Path to multipage TIFF file to read. If no path is given, a file dialog will open.",
    )

    return parser.parse_args(args)


def generate_config():
    config = Config().replace_from_config_dict(default_config_dict)
    logger.debug("Config including defaults: %s", config)

    parsed = parse_args()
    if parsed.config:
        logger.info("Loading config file from %s", parsed.write_config)
        config = config.replace_from_toml(parsed.config)
        logger.debug("Config including %s: %s", parsed.write_config, config)
    return config.replace_from_namespace(parsed)


def main():
    config = generate_config()

    setup_logging(config.verbose, config.logfile)
    logger.info("Configuration: %s", config)

    if config.version:
        print(__version__)
        sys.exit(0)

    if config.write_config:
        logger.info("Writing config file to %s and exiting", config.write_config)
        config.write_toml(config.write_config)
        sys.exit(0)

    return run(
        config.infile,
        config.outfile,
        config.cache,
        config.fps,
        config.threads,
        config.keys,
        config.flipx,
        config.flipy,
        config.rotate,
    )


if __name__ == "__main__":
    sys.exit(main())
