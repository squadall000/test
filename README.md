# Surface Defect Detector

This project provides a simple Python application for capturing images from a camera and detecting surface defects.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m src.main --source 0 --mode stream --config configs/default.yaml --every 5 --save-input
```

Results and metadata are saved to the `output/` directory.
