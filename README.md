# Surface Defect Detection Demo

This demo implements a basic pipeline for capturing images from a camera and detecting surface defects using OpenCV.

```
python -m src.main --source 0 --mode stream --config configs/default.yaml --every 5 --save-input
```

The script saves processed frames and masks to the `output/` directory and logs messages to `app.log`.

