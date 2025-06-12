# Line Detector

This project provides a simple script to detect lines from a camera feed using OpenCV.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Single frame and save output:

```bash
python line_detector.py --camera-id 0 --mode single --save out.png
```

Stream mode:

```bash
python line_detector.py --mode stream
```

Use `--help` to see all parameters.
