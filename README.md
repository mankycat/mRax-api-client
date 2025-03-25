# MedRAX REST API and Client

This project provides a REST API wrapper for MedRAX medical imaging analysis and a Python client to interact with it.

## Server Setup

1. Install dependencies:
```bash
pip install fastapi uvicorn requests scikit-learn matplotlib seaborn
```

2. Run the API server:
```bash
python api.py
```

The server will run on `http://localhost:8000`

## Client Usage

### Single Image Inference
```bash
# Basic usage
python client.py single path/to/image.jpg

# With user message
python client.py single path/to/image.jpg \
  --user-message "Please focus on the lung area"
```

### Batch Image Inference
```bash
# Single directory scan
python client.py batch path/to/images/

# Recursive directory scan
python client.py batch path/to/images/ --recursive

# With user message
python client.py batch path/to/images/ --user-message "Analyze for pneumothorax"

# With Excel ground truth
python client.py batch path/to/images/ \
  --ground_truth_excel ground_truth.xlsx \
  --labels "Normal" "Abnormal"
```

### Confusion Matrix Reports

When using Excel ground truth, the client will generate:
1. `confusion_matrix.png` - Visual matrix plot
2. `confusion_matrix_report.xlsx` - Detailed report with:
   - File paths
   - Ground truth labels
   - Predicted labels

The Excel ground truth file should contain:
- `SCHE_NO` column with image identifiers (filename stems)
- `REP` column with ground truth labels

Example Excel format:
| SCHE_NO     | REP  |
|-------------|------|
| 11403020401 | Abnormal |
| 11403050064 | Abnormal |
| 11403050075 | Normal |

### Health Check
```bash
python client.py health
```

## API Endpoints

- `POST /inference` - Single image inference
- `POST /batch_inference` - Batch image inference
- `GET /health` - Health check

## Response Format

### Single Inference
```json
{
  "status": "success",
  "result": {
    "messages": [
      {
        "content": "analysis results...",
        "role": "assistant"
      }
    ]
  },
  "display_image": "path/to/processed/image.jpg"
}
```

### Batch Inference
```json
{
  "status": "completed",
  "results": [
    {
      "filename": "image1.jpg",
      "result": {
        "messages": [
          {
            "content": "analysis results...",
            "role": "assistant"
          }
        ]
      },
      "display_image": "path/to/processed/image1.jpg"
    }
  ]
}
```

## Confusion Matrix

When providing ground truth labels and class labels, the client will:
1. Generate a confusion matrix
2. Save it as `confusion_matrix.png`
3. Return the matrix data in JSON format

Example output:
```json
{
  "matrix": [[10, 2], [1, 12]],
  "labels": ["Normal", "Abnormal"],
  "plot_path": "confusion_matrix.png"
}
```

## Requirements

- Python 3.8+
- MedRAX server running on GPU
- Required Python packages (see Server Setup)
