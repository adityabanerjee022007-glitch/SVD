# Image Compression using SVD

A Python implementation of grayscale image compression using Singular Value Decomposition (SVD).

## Requirements

- Python 3
- NumPy
- Matplotlib
- Pillow

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Place an image named `photo.jpg` in the repository folder, then run:

```bash
python SVD.py
```

If `photo.jpg` is not found, the script generates a synthetic test image automatically.

The script demonstrates reconstructions using the top 10, 30, and 60 singular values.
