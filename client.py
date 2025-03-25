import requests
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json

class MedRAXClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def send_single_image(self, image_path: str) -> Dict:
        """Send a single image for inference"""
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f)}
            response = requests.post(f"{self.base_url}/inference", files=files)
        return response.json()
    
    def send_batch_images(self, image_paths: List[str]) -> Dict:
        """Send multiple images for batch inference"""
        files = []
        for path in image_paths:
            with open(path, 'rb') as f:
                files.append(('files', (Path(path).name, f)))
        
        response = requests.post(f"{self.base_url}/batch_inference", files=files)
        return response.json()
    
    def calculate_confusion_matrix(self, 
                                 predictions: List[str], 
                                 ground_truth: List[str],
                                 labels: Optional[List[str]] = None) -> Dict:
        """Calculate and visualize confusion matrix"""
        cm = confusion_matrix(ground_truth, predictions, labels=labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        return {
            "matrix": cm.tolist(),
            "labels": labels,
            "plot_path": "confusion_matrix.png"
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
    
    # Batch images parser
    batch_parser = subparsers.add_parser('batch')
    batch_parser.add_argument('image_paths', nargs='+', help='Paths to image files')
    batch_parser.add_argument('--ground_truth', nargs='+', 
                            help='Ground truth labels for confusion matrix')
    batch_parser.add_argument('--labels', nargs='+',
                            help='Class labels for confusion matrix')
    
    # Health check parser
    health_parser = subparsers.add_parser('health')
    
    args = parser.parse_args()
    client = MedRAXClient()
    
    if args.command == 'single':
        result = client.send_single_image(args.image_path)
        print(json.dumps(result, indent=2))
        
    elif args.command == 'batch':
        result = client.send_batch_images(args.image_paths)
        print(json.dumps(result, indent=2))
        
        if args.ground_truth and args.labels:
            # Extract predictions from results
            predictions = [r['result']['messages'][-1]['content'] 
                         for r in result['results'] 
                         if 'result' in r]
            
            cm_result = client.calculate_confusion_matrix(
                predictions, args.ground_truth, args.labels
            )
            print("\nConfusion Matrix:")
            print(json.dumps(cm_result, indent=2))
            print(f"Plot saved to {cm_result['plot_path']}")
            
    elif args.command == 'health':
        result = client.health_check()
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
