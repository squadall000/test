# Irregularity Detection Demo

This project provides a small prototype for detecting surface irregularities in
camera images or still photos. It uses OpenCV to replace the background with
black, detect bumps or dents using Laplacian or Canny methods, and overlay the
detected regions in color.

## Usage

```bash
pip install opencv-python numpy scikit-image
python main.py --camera 0
```

See `python main.py --help` for available options such as `--image` for testing a
single image, `--save` to write output video, and `--threshold` to change the
detection sensitivity.
