# FrAn: FRame ANnotation

Watch a video in multipage TIFF format and label frames where events start or end.

## Installation

```bash
pip install git+https://github.com/clbarnes/frame_annotator
```

## Usage

```help
usage: fran [-h] [--write_config WRITE_CONFIG] [--outfile OUTFILE]
            [--config CONFIG] [--fps FPS] [--cache CACHE] [--threads THREADS]
            [--keys KEYS] [--flipx] [--flipy] [--rotate ROTATE]
            [infile]

Log video (multipage TIFF) frames in which an event starts or ends

positional arguments:
  infile                Path to multipage TIFF file to read. If no path is given, a file dialog will open.

optional arguments:
  -h, --help            show this help message and exit
  --write_config WRITE_CONFIG
                        Write back the complete config to a file at this path, then exit
  --outfile OUTFILE, -o OUTFILE
                        Path to CSV for loading/saving. If no path is selected when you save, a file dialog will open.
  --config CONFIG       Path to TOML file for config
  --fps FPS             Maximum frames per second; default 30
  --cache CACHE         Approximately how many frames to cache (increase if reading over a network and you have lots of RAM); default 500
  --threads THREADS     number of threads to use for reading file (increase if reading over a network); default 3
  --keys KEYS           Optional mappings from event name to key, in the format "w=forward,a=left,s=back,d=right"
  --flipx               Flip image in x
  --flipy               Flip image in y
  --rotate ROTATE       Rotate image (degrees counterclockwise; applied after flipping)

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
  
```

### Examples

```bash
# run with default settings: a file dialog will ask where your TIFF file is and where to save the CSV
fran

# run with 5 image-reading threads (more than the default 3)
# and a 1000-frame cache (more than the default 500)
fran --threads 5 --cache 1000

# copy the default config to a file, which you can edit
fran --write_config my_default_config.toml

# run with a given config file
fran --config my_config.toml

# give the input and output files to avoid file dialogs
fran my_image_file.tif --outfile my_results.csv

# flip the image in x and then rotate counterclockwise by 45 degrees
fran my_image_file.tif --flipx --rotate 45

```

To add event names (with mappings from their associate keys) to your config, the `[keys]` section should look like

```toml
[keys]
w = "forward"
a = "left"
s = "back"
d = "right"

```

See [the default config file](fran/config.toml) for the defaults.

## Output

If `--outfile` is given, saving writes to the file in CSV format.
Otherwise, it writes to stdout (all other messages are on stderr).

If given output file already exists, events will be loaded from it.
This is not possible if the output file is selected in the GUI.

e.g.

```csv
start,stop,key,event,note
120,500,f,forward,"this event is a nice event"
505,530,b,backward,
650,None,r,right,"this doesn't finish in the video"
```
