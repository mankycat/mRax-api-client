import requests
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import glob
import pandas as pd
from typing import List, Dict, Optional, Union
import json
import os

class MedRAXClient:
    def __init__(self, base_url: str = "http://116.50.47.34:58585"):
        self.base_url = base_url
        
    def send_single_image(self, image_path: str, user_message: str = None) -> Dict:
        """Send a single image for inference with optional user message"""
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f)}
            data = {'user_message': user_message} if user_message else None
            response = requests.post(f"{self.base_url}/inference", files=files, data=data)
        return response.json()
    
    def send_batch_images(self, image_paths: List[str], user_message: str = None) -> Dict:
        """Send multiple images for batch inference with optional user message"""
        files = []
        file_handles = []
        try:
            for path in image_paths:
                f = open(path, 'rb')
                file_handles.append(f)
                files.append(('files', (Path(path).name, f)))
            
            data = {'user_message': user_message} if user_message else None
            response = requests.post(f"{self.base_url}/batch_inference", files=files, data=data)
            return response.json()
        finally:
            for f in file_handles:
                f.close()
    
    def find_png_files(self, root_dir: str) -> List[str]:
        """Recursively find all PNG files in directory and subdirectories"""
        return glob.glob(os.path.join(root_dir, '**', '*.png'), recursive=True)

    def load_ground_truth(self, excel_path: str) -> Dict[str, str]:
        """Load ground truth labels from Excel file"""
        df = pd.read_excel(excel_path)
        ground_truth = {}
        for _, row in df.iterrows():
            if 'SCHE_NO' in row and 'REP' in row:
                ground_truth[row['SCHE_NO']] = row['REP']
        return ground_truth

    def calculate_confusion_matrix(self, 
                                 predictions: List[Dict], 
                                 ground_truth: Dict[str, str],
                                 labels: Optional[List[str]] = None) -> Dict:
        """Calculate and visualize confusion matrix with detailed Excel report"""
        # Extract true and predicted labels
        y_true = []
        y_pred = []
        report_data = []
        
        for pred in predictions:
            filename = pred['filename']
            sche_no = Path(filename).stem
            pred_label = pred['result']['messages'][-1]['content']
            
            if sche_no in ground_truth:
                y_true.append(ground_truth[sche_no])
                y_pred.append(pred_label)
                report_data.append({
                    'File': filename,
                    'Ground Truth': ground_truth[sche_no],
                    'Prediction': pred_label
                })

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()

        # Save detailed report
        report_df = pd.DataFrame(report_data)
        report_df.to_excel('confusion_matrix_report.xlsx', index=False)
        
        return {
            "matrix": cm.tolist(),
            "labels": labels,
            "plot_path": "confusion_matrix.png",
            "report_path": "confusion_matrix_report.xlsx"
        }
    
    def health_check(self) -> Dict:
        """Check server health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

def main():
    parser = argparse.ArgumentParser(description='MedRAX Client')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Single image parser
    single_parser = subparsers.add_parser('single')
    single_parser.add_argument('image_path', help='Path to image file')
    single_parser.add_argument('--user-message', help='Optional message to include with the image')
    
    # Batch images parser
    batch_parser = subparsers.add_parser('batch')
    batch_parser.add_argument('path', help='Path to image file or directory')
    batch_parser.add_argument('--recursive', action='store_true', 
                            help='Recursively scan directory for PNG files')
    batch_parser.add_argument('--user-message', help='Optional message to include with all images')
    batch_parser.add_argument('--ground_truth_excel', 
                            help='Excel file containing ground truth labels')
    batch_parser.add_argument('--labels', nargs='+',
                            help='Class labels for confusion matrix')
    
    # Health check parser
    health_parser = subparsers.add_parser('health')
    
    args = parser.parse_args()
    client = MedRAXClient()
    
    if args.command == 'single':
        result = client.send_single_image(args.image_path, args.user_message)
        print(json.dumps(result, indent=2))
        
    elif args.command == 'batch':
        # Get image paths
        if os.path.isdir(args.path):
            if args.recursive:
                image_paths = client.find_png_files(args.path)
            else:
                image_paths = glob.glob(os.path.join(args.path, '*.png'))
        else:
            image_paths = [args.path]
            
        if not image_paths:
            print("Error: No PNG files found")
            return
            
        # Process batch
        result = client.send_batch_images(image_paths, args.user_message)
        print(json.dumps(result, indent=2))
        
        # Handle confusion matrix if requested
        if args.ground_truth_excel and args.labels:
            # Load ground truth from Excel
            ground_truth = client.load_ground_truth(args.ground_truth_excel)
            
            # Prepare predictions with filenames
            predictions = []
            for res in result['results']:
                if 'result' in res:
                    predictions.append({
                        'filename': res['filename'],
                        'result': res['result']
                    })
            
            # Calculate confusion matrix
            cm_result = client.calculate_confusion_matrix(
                predictions, ground_truth, args.labels
            )
            print("\nConfusion Matrix Results:")
            print(f"- Matrix plot: {cm_result['plot_path']}")
            print(f"- Detailed report: {cm_result['report_path']}")
            
    elif args.command == 'health':
        result = client.health_check()
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
