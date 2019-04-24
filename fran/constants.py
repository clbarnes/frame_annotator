from pathlib import Path

import toml

here = Path(__file__).absolute()
package_dir = here.parent
config_path = package_dir / "config.toml"

default_config = toml.load(config_path)

DEFAULT_CACHE_SIZE = default_config["settings"]["cache"]
DEFAULT_FPS = default_config["settings"]["fps"]
DEFAULT_THREADS = default_config["settings"]["threads"]
DEFAULT_FLIPX = default_config["transform"]["flipx"]
DEFAULT_FLIPY = default_config["transform"]["flipy"]
DEFAULT_ROTATE = default_config["transform"]["rotate"]
DEFAULT_KEYS = default_config["keys"]

CONTROLS = """
Playback
========
LEFT and RIGHT arrows play the video in that direction at the configured FPS.
Hold SHIFT + direction to play at 10x speed.
Hold CTRL + direction to step through one frame at a time.

Events
======
LETTER keys mark the start of an event associated with that letter.
SHIFT + LETTER marks the end of the event.
Events can overlap.
Delete an event initiation by terminating it at the same frame, and vice versa.

Status
======
SPACE shows in-progress events
RETURN shows the current result table in the console
BACKSPACE shows the current frame number and contrast thresholds in the interval [0, 1]

Prompts
=======
DELETE shows a prompt asking which in-progress event to delete
CTRL + n shows a prompt asking which in-progress event to add a note to, and the note

In order to interact with a prompt, you will need to click on the console and enter your response.
Then, click on the fran window to keep annotating.

Other
=====
CTRL + s to save
CTRL + z to undo
CTRL + r to redo
CTRL + h to show this message
""".rstrip()
