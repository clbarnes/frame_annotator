# FrAn: FRame ANnotation

Watch a video in multipage TIFF format and label frames where events start or end.

Written for python 3.7.

## Installation

```bash
pip install fran
```

If you think you may already have a version of `fran` installed, add the `--upgrade` option.

## Usage

### fran

```_fran
usage: fran [-h] [--write_config WRITE_CONFIG] [--outfile OUTFILE]
            [--config CONFIG] [--fps FPS] [--cache CACHE] [--threads THREADS]
            [--keys KEYS] [--flipx] [--flipy] [--rotate ROTATE] [--version]
            [--verbose] [--logfile LOGFILE]
            [infile]

Log video (multipage TIFF) frames in which an event starts or ends

positional arguments:
  infile                Path to multipage TIFF file to read. If no path is given, a file dialog will open.

optional arguments:
  -h, --help            show this help message and exit
  --write_config WRITE_CONFIG, -w WRITE_CONFIG
                        Write back the complete config to a file at this path, then exit
  --outfile OUTFILE, -o OUTFILE
                        Path to CSV for loading/saving. If no path is selected when you save, a file dialog will open.
  --config CONFIG, -c CONFIG
                        Path to TOML file for config
  --fps FPS, -f FPS     Maximum frames per second; default 30
  --cache CACHE, -n CACHE
                        Number of frames to cache (increase if reading over a network and you have lots of RAM); default 500
  --threads THREADS, -t THREADS
                        number of threads to use for reading file (increase if reading over a network); default 3
  --keys KEYS, -k KEYS  Optional mappings from event name to key, in the format "w=forward,a=left,s=back,d=right". These are additive with those defined in the config
  --flipx, -x           Flip image in x
  --flipy, -y           Flip image in y
  --rotate ROTATE, -r ROTATE
                        Rotate image (degrees counterclockwise; applied after flipping)
  --version             Print the version and then exit
  --verbose, -v         Increase verbosity of logging (can be repeated). One for DEBUG, two for FRAME.
  --logfile LOGFILE, -l LOGFILE
                        File to save log to

Playback
========
LEFT and RIGHT arrows play the video in that direction at the configured FPS.
Hold SHIFT + direction to play at 10x speed.
Press COMMA or PERIOD to step one frame left or right, respectively.

Contrast
========
Contrast is controlled by rescaling the pixel intensities between a min and max threshold.
Hold UP/DOWN to change the lower threshold (increasing it blacks out dim pixels).
Hold SHIFT + UP/DOWN to change the upper threshold (decreasing it whites out bright pixels).

Events
======
LETTER keys mark the start of an event associated with that letter.
SHIFT + LETTER marks the end of the event.
Events can overlap, but events of the same type cannot contain each other or start on the same frame.

Status
======
SPACE shows in-progress events.
RETURN shows the current result table in the console.
BACKSPACE shows the current frame number and contrast thresholds in the interval [0, 1].

Prompts
=======
DELETE shows a prompt asking which in-progress event to delete, then a confirmation.
SHIFT + delete shows a prompt asking which keypress on this frame to delete.
CTRL + n shows a prompt asking which in-progress event whose note to edit (if there is more than one), then the new value.

Prompts may appear in unexpected places on your screen.

Debug
=====
CTRL + d to dump the event log for debugging

Other
=====
CTRL + j to Jump to a frame
CTRL + s to Save
CTRL + z to undo
CTRL + r to Redo
CTRL + h to show this Help message
```

#### Examples

```bash
# run with default settings: a file dialog will ask where your TIFF file is and where to save the CSV
fran

# run with 5 image-reading threads (more than the default 3)
# and a 1000-frame cache (more than the default 500)
fran --threads 5 --cache 1000

# copy the default config to a file, which you can edit
fran --write_config my_config.toml

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

See [the default config file](fran/config.toml) for the defaults and other settings which can be configured.

#### Output

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

#### Example workflow

1. Install fran (`pip install fran`)
2. Write a basic config file to your working directory (`fran --write_config my_project/project_config.toml`)
3. Edit that config file as you like, with the image transform, event names and so on
4. Start annotating! `fran my_project/my_video.tiff --config my_project/project_config.toml --outfile my_project/results.csv`

### fran-rename

A utility to rename events in an existing results file.

```_fran_rename
usage: fran-rename [-h] [--config CONFIG] [--keys KEYS] [--version] [--print]
                   results

Rename all of the events in a given results file, with a key mapping specified
in a config TOML and/or command line argument.

positional arguments:
  results               Path to CSV to update.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Path to TOML file for config
  --keys KEYS, -k KEYS  Optional mappings from event name to key, in the
                        format "w=forward,a=left,s=back,d=right". These are
                        additive with those defined in the config
  --version             Print the version and then exit
  --print, -p           Print results to stdout, instead of saving file back
```

## MacOS

[MacOS disagrees with pygame](https://bitbucket.org/pygame/pygame/issues/203/window-does-not-get-focus-on-os-x-with)
 if installed in a virtual environment:
 use the system python (if 3.7+ is available) or one installed with homebrew.


## Contributing

```sh
# Install development dependencies
pip install -r requirements.txt
# Install package in editable mode
pip install -e .
# Install pre-commit hooks for lint/formatting
pre-commit install
```

## Attributions

Test data [JeffT1_le.tif](./data/JeffT1_le.tif) from [ImageJ image repository](https://imagej.nih.gov/ij/images/t1-head-raw.zip).

Thanks to Nadine Randel for feature requests and bug reports.
