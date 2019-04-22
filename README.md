# Frame annotator

For watching a video in multipage TIFF format and labelling frames where events start or end.

## Installation

```bash
pip install git+https://github.com/clbarnes/frame_annotator
```

## Usage

```
usage: frame_annotator.py [-h] [--outfile OUTFILE] [--fps FPS] [--cache CACHE]
                          infile

- Hold right or left to play the video at a reasonable (and configurable) FPS
- Hold Shift + arrow to attempt to play the video at 10x speed
- Press Ctrl + arrow to step through one frame at a time
- Press any letter key to mark the onset of an event, and Shift + that letter to mark the end of it
  - Marking the onset and end of an event at the same frame will remove both annotations
- Press Space to see which events are currently in progress
- Press Enter to see the table of results in the console
- Press Backspace to see the current frame number

positional arguments:
  infile                Path to multipage TIFF file to read

optional arguments:
  -h, --help            show this help message and exit
  --outfile OUTFILE, -o OUTFILE
                        Path to save a CSV to
  --fps FPS             Maximum frames per second
  --cache CACHE         Approximately how many frames to cache
```