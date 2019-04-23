# Frame annotator

For watching a video in multipage TIFF format and labelling frames where events start or end.

## Installation

```bash
pip install git+https://github.com/clbarnes/frame_annotator
```

## Usage

```
usage: frame_annotator.py [-h] [--write_config WRITE_CONFIG]
                          [--outfile OUTFILE] [--config CONFIG] [--fps FPS]
                          [--cache CACHE] [--threads THREADS] [--keys KEYS]
                          [infile]

- Hold right or left to play the video at a reasonable (and configurable) FPS
- Hold Shift + arrow to attempt to play the video at 10x speed
- Press Ctrl + arrow to step through one frame at a time
- Press any letter key to mark the onset of an event, and Shift + that letter to mark the end of it
  - Marking the onset and end of an event at the same frame will remove both annotations
- Press Space to see which events are currently in progress
- Press Delete to show a prompt asking which in-progress event to delete
  - You will need to select the console to enter it, then re-select the annotator window
- Press Enter to see the table of results in the console
- Press Backspace to see the current frame number and contrast thresholds
- Ctrl + s to save
- Ctrl + z to undo
- Ctrl + r to redo
- Ctrl + n to show a prompt asking which in-progress event to note
  - You will need to select the console to enter it, then re-select the annotator window
- Ctrl + h to show this message

positional arguments:
  infile                Path to multipage TIFF file to read

optional arguments:
  -h, --help            show this help message and exit
  --write_config WRITE_CONFIG
                        Write back the complete config to a file at this path
  --outfile OUTFILE, -o OUTFILE
                        Path to CSV for loading/saving
  --config CONFIG       Path to TOML file for config
  --fps FPS             Maximum frames per second
  --cache CACHE         Approximately how many frames to cache
  --threads THREADS     number of threads to use for reading file
  --keys KEYS           Mapping from event name to key, in the format "forward=w,left=a,back=s,right=d"
  
```