# Qwen Image Edit GUI

A Gradio-based GUI for the Qwen-Image-Edit-2511 model, optimized for Apple Silicon (MPS).

## Features

- Image editing with AI using text prompts
- Multi-image support (face swapping, combining people, style transfer)
- Progress tracking during inference
- Optimized for Apple Metal GPU (MPS)

## Requirements

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3) or NVIDIA GPU
- ~20GB disk space for model download
- 16GB+ RAM recommended

## Installation

```bash
# Create virtual environment
python3 -m venv venv

# Install dependencies
./venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# Run the app
./venv/bin/python app.py

# Or use the start script
./start.sh
```

Then open http://localhost:7860 in your browser.

## How to Use

1. Click "Load Model" (first run downloads ~20GB model)
2. Upload main image (Figure 1)
3. Optionally upload reference image (Figure 2)
4. Enter your edit prompt
5. Click "Edit Image"

### Example Prompts

**Single image:**
- "Make the background a sunset beach"
- "Change the person's hair color to blue"

**Multi-image:**
- "Replace the face in Figure 1 with the face from Figure 2"
- "Put the person from Figure 1 next to the person from Figure 2"

## License

Apache 2.0 (same as Qwen-Image-Edit-2511 model)
