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

# With Excel ground truth (basic keyword matching)
python client.py batch path/to/images/ \
  --ground_truth_excel ground_truth.xlsx \
  --labels "Normal" "Abnormal"

# With OpenAI API for better classification
python client.py batch path/to/images/ \
  --ground_truth_excel ground_truth.xlsx \
  --labels "Normal" "Abnormal" \
  --openai-api-key "your_api_key_here" \
  --openai-model "gpt-4"

# Multi-image cases with case-level aggregation
python client.py batch path/to/multi_image_cases/ --recursive \
  --ground_truth_excel ground_truth.xlsx \
  --labels "Normal" "Abnormal" "Unknown" \
  --voting-threshold 0.6
```

### Confusion Matrix Reports

When using Excel ground truth, the client will generate:
1. `confusion_matrix.png` - Visual matrix plot
2. `confusion_matrix_report.xlsx` - Detailed report with multiple sheets:
   - **Case_Level**: Aggregated results for each case with voting statistics
   - **Image_Level**: Individual image predictions with detailed information

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
1. Parse AI responses to determine Normal/Abnormal classification
   - Uses OpenAI API if --openai-api-key provided
   - Falls back to keyword matching otherwise
2. For multi-image cases:
   - Group images by case number (SCHE_NO)
   - Aggregate predictions using voting mechanism
   - Apply voting threshold to determine final case classification
3. Generate a confusion matrix at the case level
4. Save it as `confusion_matrix.png`
5. Create detailed Excel report with case and image statistics

Example output:
```json
{
  "matrix": [[10, 2, 0], [1, 12, 1]],
  "labels": ["Normal", "Abnormal", "Unknown"],
  "plot_path": "confusion_matrix.png",
  "report_path": "confusion_matrix_report.xlsx",
  "total_cases": 26,
  "total_images": 342
}
```

### Case-Level Voting

The `--voting-threshold` parameter (default: 0.5) controls how image predictions are aggregated:
- If ≥ threshold% of images are classified as "Abnormal", the case is "Abnormal"
- If ≥ threshold% of images are classified as "Normal", the case is "Normal"
- Otherwise, the case is "Unknown"

## Requirements

- Python 3.8+
- MedRAX server running on GPU
- Required Python packages (see Server Setup)
