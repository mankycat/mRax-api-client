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
import time # Added for potential delay/retry logic if needed

class MedRAXClient:
    def __init__(self, base_url: str = "http://0.0.0.0:8585",
                openai_api_key: str = None,
                openai_endpoint: str = "https://api.openai.com/v1/chat/completions",
                openai_model: str = "gpt-4o-mini"):
        self.base_url = base_url
        self.openai_api_key = openai_api_key
        self.openai_endpoint = openai_endpoint
        self.openai_model = openai_model

    def send_single_image(self, image_path: str, user_message: str = None, force_tool: str = None) -> Dict:
        """Send a single image for inference with optional user message and force_tool"""
        with open(image_path, 'rb') as f:
            # Send only base filename in request
            base_filename = Path(image_path).name
            files = {'file': (base_filename, f)}
            data = {}
            if user_message:
                data['user_message'] = user_message
            if force_tool:
                data['force_tool'] = force_tool

            # Debug print
            print(f"[DEBUG] Sending request with data: {data}, file: {base_filename}")

            try:
                response = requests.post(f"{self.base_url}/inference", files=files, data=data)
                response.raise_for_status()
                result = response.json()
                # Add the original full path back for consistency if needed downstream,
                # although single image processing might not use it like batch does.
                result['filename'] = image_path
                return result
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Failed to process image {image_path}: {e}")
                # Return original path in error case as well
                return {
                    "status": "failed",
                    "error": str(e),
                    "filename": image_path
                }

    def send_batch_images(self, image_paths: List[str], user_message: str = None, force_tool: str = None) -> Dict:
        """
        Send multiple images for batch inference with optional user message and force_tool.
        Sends only base filenames in the request but restores original paths in the response.
        """
        request_files_list = []
        failed_to_open = []
        # Map base filename sent in request -> original full path
        base_filename_to_original_path = {}

        # Prepare file data and mapping
        for path in image_paths:
            try:
                with open(path, 'rb') as f:
                    content = f.read()
                    base_filename = Path(path).name
                    # Store mapping - handle potential duplicate base filenames if necessary
                    if base_filename in base_filename_to_original_path:
                        # If duplicate base names exist, append a counter or use full path as key?
                        # For now, let's assume base names are unique enough within a batch or overwrite is acceptable.
                        # A more robust solution might involve passing a unique ID per file.
                        print(f"[WARNING] Duplicate base filename '{base_filename}' detected. Mapping might be ambiguous.")
                    base_filename_to_original_path[base_filename] = path
                    request_files_list.append(('files', (base_filename, content)))
            except Exception as e:
                print(f"[ERROR] Failed to open {path}: {e}")
                # Store failures to include in the final results
                failed_to_open.append({
                    "filename": path, # Use original path for error reporting
                    "status": "failed",
                    "error": f"Failed to open file: {str(e)}"
                })

        # If no files could be prepared, return early
        if not request_files_list:
            return {"status": "failed", "error": "No files could be prepared for upload.", "results": failed_to_open}

        # Prepare form data
        data = {}
        if user_message:
            data['user_message'] = user_message
        if force_tool:
            data['force_tool'] = force_tool

        # Make the request
        try:
            print(f"[DEBUG] Sending batch request with {len(request_files_list)} files...")
            response = requests.post(f"{self.base_url}/batch_inference", files=request_files_list, data=data)
            response.raise_for_status()
            batch_result = response.json()
            print(f"[DEBUG] Received batch response status: {batch_result.get('status')}")

            # Restore original paths in the results from the server
            restored_results = []
            if "results" in batch_result:
                for item in batch_result["results"]:
                    # The server should return the base filename it received
                    base_filename_received = item.get('filename')
                    if base_filename_received in base_filename_to_original_path:
                        # Replace base filename with original full path
                        item['filename'] = base_filename_to_original_path[base_filename_received]
                        restored_results.append(item)
                    else:
                        print(f"[WARNING] Received result for filename '{base_filename_received}' which was not in the original request mapping.")
                        # Keep the result but maybe flag it or use the received name
                        restored_results.append(item)

            # Combine successfully processed results with files that failed to open
            final_results = restored_results + failed_to_open

            return {
                "status": batch_result.get("status", "unknown"), # Propagate server status
                "results": final_results
            }

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Batch inference request failed: {e}")
            # Include files that failed to open in the error response
            return {
                "status": "failed",
                "error": f"Batch request failed: {str(e)}",
                "results": failed_to_open # Only include pre-request failures here
            }

    def find_png_files(self, root_dir: str) -> List[str]:
        """Recursively find all PNG files in directory and subdirectories"""
        return glob.glob(os.path.join(root_dir, '**', '*.png'), recursive=True)

    def group_images_by_case(self, image_paths: List[str]) -> Dict[str, List[str]]:
        """Group image paths by case number (SCHE_NO)"""
        cases = {}
        for path in image_paths:
            # Ensure path is a string before processing
            if not isinstance(path, str):
                print(f"[WARNING] Skipping non-string path: {path}")
                continue

            full_path = Path(path).absolute()
            sche_no = None

            # Extract SCHE_NO from path components
            for part in full_path.parts:
                # Check for all-digit SCHE_NO (original format)
                if part.isdigit() and len(part) >= 8:
                    sche_no = part
                    break
                # Check for P-prefixed SCHE_NO format (e.g., P1312110476)
                elif part.startswith('P') and part[1:].isdigit() and len(part) >= 9:
                    sche_no = part
                    break

            if sche_no:
                if sche_no not in cases:
                    cases[sche_no] = []
                cases[sche_no].append(path)
            else:
                print(f"[WARNING] Could not extract SCHE_NO from: {path}")

        # Print summary of cases
        print(f"[INFO] Found {len(cases)} cases with {sum(len(imgs) for imgs in cases.values())} images")
        # for sche_no, imgs in cases.items():
        #     print(f"[INFO] Case {sche_no}: {len(imgs)} images")

        return cases

    def load_ground_truth(self, excel_path: str) -> Dict[str, str]:
        """Load ground truth labels from Excel file and translate to English"""
        try:
            df = pd.read_excel(excel_path)
            ground_truth = {}
            translation = {
                '正常': 'Normal',
                '異常': 'Abnormal'
            }
            required_cols = ['SCHE_NO', 'REP']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Excel file must contain columns: {required_cols}")

            for _, row in df.iterrows():
                # Check for NaN or missing values before processing
                if pd.notna(row['SCHE_NO']) and pd.notna(row['REP']):
                    # Convert SCHE_NO to string and strip any whitespace
                    sche_no = str(row['SCHE_NO']).strip()
                    rep_value = str(row['REP']).strip()
                    # Translate Chinese labels to English
                    ground_truth[sche_no] = translation.get(rep_value, rep_value) # Keep original if no translation
            print(f"[DEBUG] Loaded {len(ground_truth)} ground truth entries.")
            # print(f"[DEBUG] Loaded ground truth sample: {dict(list(ground_truth.items())[:5])}")
            return ground_truth
        except FileNotFoundError:
            print(f"[ERROR] Ground truth Excel file not found: {excel_path}")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load or parse ground truth Excel file {excel_path}: {e}")
            raise

    def parse_ai_message(self, message: Union[str, None]) -> str:
        """Parse AI message to determine Normal/Abnormal classification"""
        if not message or not isinstance(message, str):
             print("[DEBUG] Invalid or empty AI message received for parsing.")
             return "Unknown"

        print(f"\n[DEBUG] Parsing AI message (first 200 chars): {message[:200]}...")

        # Use OpenAI API for more accurate classification if key is provided
        if not self.openai_api_key:
            print("[WARNING] OpenAI API key not provided. Classification may be less accurate.")
            # Basic fallback logic if no API key
            message_lower = message.lower()
            if "normal" in message_lower and "abnormal" not in message_lower:
                print("[DEBUG] Basic Keyword match: Normal")
                return "Normal"
            elif "abnormal" in message_lower:
                 print("[DEBUG] Basic Keyword match: Abnormal")
                 return "Abnormal"
            else:
                 print("[DEBUG] Basic Keyword match: Unknown")
                 return "Unknown"

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        # Improved prompt for clarity
        prompt = f"""Please analyze the following radiology report text and classify the overall finding as 'Normal', 'Abnormal', or 'Unknown'. Focus on the final impression or summary if available. Respond ONLY with one of these three words.

Report Text:
"{message}"

Classification:"""

        data = {
            "model": self.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 10 # Limit response length
        }

        try:
            response = requests.post(self.openai_endpoint, headers=headers, json=data, timeout=15) # Added timeout
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            # Validate response
            valid_responses = ["Normal", "Abnormal", "Unknown"]
            if content in valid_responses:
                print(f"[DEBUG] OpenAI classification result: {content}")
                return content
            else:
                print(f"[WARNING] OpenAI returned unexpected classification: '{content}'. Defaulting to Unknown.")
                return "Unknown"
        except requests.exceptions.Timeout:
            print("[ERROR] OpenAI API request timed out.")
            return "Unknown"
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] OpenAI API request failed: {e}")
            # Attempt to parse error response from OpenAI if available
            try:
                error_details = response.json()
                print(f"[ERROR] OpenAI API error details: {error_details}")
            except Exception:
                pass # Ignore if error response is not JSON
            return "Unknown"
        except Exception as e:
            print(f"[ERROR] Unexpected error during OpenAI API call: {e}")
            return "Unknown"


    def aggregate_case_predictions(self, case_predictions: List[Dict], voting_threshold: float = 0.5) -> str:
        """
        Aggregate multiple image predictions for a single case using parsed labels.
        """
        label_counts = {"Normal": 0, "Abnormal": 0, "Unknown": 0}
        valid_predictions = 0

        for pred in case_predictions:
            try:
                # Ensure the structure is as expected after API response processing
                if 'result' in pred and 'messages' in pred['result'] and pred['result']['messages']:
                    # Access the content of the last message (assuming it's the assistant's response)
                    ai_message_content = pred['result']['messages'][-1].get('content')
                    if ai_message_content:
                        pred_label = self.parse_ai_message(ai_message_content)
                        if pred_label in label_counts:
                            label_counts[pred_label] += 1
                            valid_predictions += 1
                        else:
                             print(f"[WARNING] Unknown label '{pred_label}' returned by parse_ai_message.")
                             label_counts["Unknown"] += 1
                    else:
                        print("[WARNING] 'content' key missing in the last message.")
                        label_counts["Unknown"] += 1
                else:
                    print(f"[WARNING] Prediction result structure unexpected or empty: {pred.get('result')}")
                    label_counts["Unknown"] += 1
            except Exception as e:
                print(f"[ERROR] Failed to parse prediction content for file {pred.get('filename', 'Unknown')}: {e}")
                label_counts["Unknown"] += 1

        # If no valid predictions could be parsed, return Unknown
        if valid_predictions == 0:
            print("[DEBUG] No valid predictions found for case aggregation.")
            return "Unknown"

        # Calculate percentages based on valid predictions
        normal_pct = label_counts["Normal"] / valid_predictions
        abnormal_pct = label_counts["Abnormal"] / valid_predictions

        print(f"[DEBUG] Case voting results (based on {valid_predictions} valid predictions): "
              f"Normal={label_counts['Normal']} ({normal_pct:.2f}), "
              f"Abnormal={label_counts['Abnormal']} ({abnormal_pct:.2f}), "
              f"Unknown={label_counts['Unknown']}")

        # Determine final label based on voting threshold applied to valid predictions
        if abnormal_pct >= voting_threshold:
            return "Abnormal"
        # If not Abnormal, check if Normal meets threshold (or is the majority if Abnormal didn't meet threshold)
        # This logic prioritizes Abnormal if it meets threshold, otherwise defaults towards Normal if it's strong enough.
        elif normal_pct >= voting_threshold or label_counts["Normal"] >= label_counts["Abnormal"]:
             return "Normal"
        # If neither Normal nor Abnormal clearly dominate based on threshold
        else:
            # If Unknown votes are significant, maybe return Unknown? Or default to Normal/Abnormal based on counts?
            # Current logic defaults to Normal if Abnormal doesn't meet threshold. Let's refine.
            # If Abnormal doesn't meet threshold, return Normal only if Normal votes > Abnormal votes. Otherwise Unknown.
            if label_counts["Normal"] > label_counts["Abnormal"]:
                 return "Normal"
            else: # Includes cases where Abnormal > Normal but didn't meet threshold, or counts are equal.
                 return "Unknown"


    def calculate_confusion_matrix(self,
                                 predictions: List[Dict],
                                 ground_truth: Dict[str, str],
                                 labels: Optional[List[str]] = None,
                                 voting_threshold: float = 0.5) -> Dict:
        """Calculate and visualize confusion matrix with detailed Excel report"""
        print(f"\n[DEBUG] Calculating confusion matrix with {len(predictions)} prediction results.")
        # print(f"[DEBUG] Ground truth keys (first 5): {list(ground_truth.keys())[:5]}")
        print(f"[DEBUG] Labels for matrix: {labels}")

        if not labels:
            print("[WARNING] No labels provided for confusion matrix. Using default ['Normal', 'Abnormal', 'Unknown'].")
            labels = ['Normal', 'Abnormal', 'Unknown']

        # Group predictions by SCHE_NO using the full path now present in 'filename'
        case_predictions_grouped = {}
        processed_files_count = 0
        for pred_result in predictions:
             # Check if the prediction result itself is valid
            if not isinstance(pred_result, dict) or 'filename' not in pred_result:
                print(f"[WARNING] Skipping invalid prediction result item: {pred_result}")
                continue

            filename = pred_result['filename'] # Should now be the full path
            processed_files_count += 1

            # Ensure filename is a string before creating Path object
            if not isinstance(filename, str):
                 print(f"[WARNING] Skipping prediction with non-string filename: {filename}")
                 continue

            full_path = Path(filename).absolute()
            sche_no = None
            for part in full_path.parts:
                if part.isdigit() and len(part) >= 8:
                    sche_no = part
                    break
                elif part.startswith('P') and part[1:].isdigit() and len(part) >= 9:
                    sche_no = part
                    break

            if not sche_no:
                print(f"[ERROR] Could not extract SCHE_NO from path: {filename}")
                continue # Skip this prediction if SCHE_NO cannot be determined

            if sche_no not in case_predictions_grouped:
                case_predictions_grouped[sche_no] = []
            case_predictions_grouped[sche_no].append(pred_result) # Append the whole result dict

        print(f"[INFO] Processed {processed_files_count} prediction results, grouped into {len(case_predictions_grouped)} cases.")

        # Process each case to get aggregated predictions
        y_true = []
        y_pred = []
        case_report_data = []
        image_report_data = [] # For detailed image-level reporting

        for sche_no, case_preds_list in case_predictions_grouped.items():
            print(f"\n[DEBUG] Processing case {sche_no} with {len(case_preds_list)} prediction results")

            # Check if case exists in ground truth
            if sche_no not in ground_truth:
                print(f"[ERROR] SCHE_NO {sche_no} not found in ground truth. Skipping case.")
                # Add all images for this case to report as excluded
                for pred_item in case_preds_list:
                    image_report_data.append({
                        'File': pred_item.get('filename', 'Unknown Path'),
                        'SCHE_NO': sche_no,
                        'Included_In_Matrix': 'No',
                        'Reason': 'SCHE_NO not found in ground truth',
                        'Image_Prediction': 'N/A',
                        'Case_Prediction': 'N/A',
                        'Ground_Truth': 'N/A'
                    })
                continue # Skip to the next case

            true_label = ground_truth[sche_no]

            # Aggregate predictions for this case
            # Pass the list of prediction dicts for this case
            case_final_pred = self.aggregate_case_predictions(case_preds_list, voting_threshold)
            print(f"[DEBUG] Case {sche_no} final prediction: {case_final_pred} (Ground truth: {true_label})")

            # Add to confusion matrix data if prediction is valid
            if case_final_pred in labels:
                y_true.append(true_label)
                y_pred.append(case_final_pred)
            else:
                 print(f"[WARNING] Case {sche_no} final prediction '{case_final_pred}' not in labels {labels}. Excluding from matrix.")
                 # Still add to reports, but mark as excluded from matrix calculation itself

            # --- Detailed Reporting ---
            # Collect individual image predictions for the report
            num_normal = 0
            num_abnormal = 0
            num_unknown = 0
            for pred_item in case_preds_list:
                 img_filename = pred_item.get('filename', 'Unknown Path')
                 img_pred_label = "Error" # Default if parsing fails
                 ai_message_snippet = "N/A"
                 reason = "Successfully processed"
                 included = "Yes"

                 try:
                     if 'result' in pred_item and 'messages' in pred_item['result'] and pred_item['result']['messages']:
                         ai_message_content = pred_item['result']['messages'][-1].get('content')
                         if ai_message_content:
                             img_pred_label = self.parse_ai_message(ai_message_content)
                             ai_message_snippet = ai_message_content
                             if img_pred_label == "Normal": num_normal += 1
                             elif img_pred_label == "Abnormal": num_abnormal += 1
                             else: num_unknown += 1
                         else:
                             reason = "AI message content missing"
                             img_pred_label = "Error"
                             included = "No (Processing Error)"
                     else:
                         reason = "Unexpected result structure"
                         img_pred_label = "Error"
                         included = "No (Processing Error)"
                 except Exception as e:
                     reason = f"Parsing error: {str(e)}"
                     img_pred_label = "Error"
                     included = "No (Processing Error)"
                     print(f"[ERROR] Failed to process image {img_filename} report data: {e}")

                 image_report_data.append({
                     'File': img_filename,
                     'SCHE_NO': sche_no,
                     'Included_In_Matrix': included, # Reflects if individual image processing was ok
                     'Reason': reason,
                     'Image_Prediction': img_pred_label,
                     'Ground_Truth': true_label,
                     'AI_Message_Snippet': ai_message_snippet
                 })

            # Add aggregated data to case report
            case_report_data.append({
                'SCHE_NO': sche_no,
                'Total_Images': len(case_preds_list),
                'Ground_Truth': true_label,
                'Case_Prediction': case_final_pred, # The aggregated prediction
                'Normal_Count': num_normal,
                'Abnormal_Count': num_abnormal,
                'Unknown_Count': num_unknown,
                'Match_Ground_Truth': case_final_pred == true_label if case_final_pred != "Unknown" else "N/A" # Compare only if prediction is not Unknown
            })
            # --- End Detailed Reporting ---


        # Ensure we have data before creating matrix
        if not y_true or not y_pred:
             print("[ERROR] No valid cases found to generate confusion matrix.")
             cm = [[0]*len(labels)]*len(labels) # Return zero matrix
        else:
            # Ensure all labels in y_true/y_pred are actually in the provided labels list
            valid_labels = set(labels)
            # Filter y_true first
            y_true_filtered = [label for label in y_true if label in valid_labels]
            # Filter y_pred based on the indices where y_true was valid
            y_pred_filtered = [y_pred[i] for i, label in enumerate(y_true) if label in valid_labels]
            # Further filter y_pred_filtered to ensure its values are also in labels
            final_y_pred = [pred for pred in y_pred_filtered if pred in valid_labels]
            # Ensure y_true and y_pred still align after filtering y_pred
            final_y_true = [y_true_filtered[i] for i, pred in enumerate(y_pred_filtered) if pred in valid_labels]


            if not final_y_true or not final_y_pred:
                 print("[ERROR] No valid predictions matching the provided labels found after filtering.")
                 cm = [[0]*len(labels)]*len(labels)
            else:
                 # Generate confusion matrix using filtered lists and specified labels
                 print(f"[DEBUG] Generating matrix with {len(final_y_true)} true labels and {len(final_y_pred)} predicted labels.")
                 cm = confusion_matrix(final_y_true, final_y_pred, labels=labels)


        # Plot confusion matrix
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix (Case Level)')
            plot_path = 'confusion_matrix.png'
            plt.savefig(plot_path)
            plt.close()
            print(f"[INFO] Confusion matrix plot saved to {plot_path}")
        except Exception as e:
            print(f"[ERROR] Failed to plot confusion matrix: {e}")
            plot_path = None

        # Save detailed reports to Excel
        report_path = 'confusion_matrix_report.xlsx'
        try:
            case_df = pd.DataFrame(case_report_data)
            image_df = pd.DataFrame(image_report_data)

            with pd.ExcelWriter(report_path) as writer:
                case_df.to_excel(writer, sheet_name='Case_Level_Summary', index=False)
                image_df.to_excel(writer, sheet_name='Image_Level_Detail', index=False)
            print(f"[INFO] Detailed report saved to {report_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save detailed report to Excel: {e}")
            report_path = None

        # Print summary
        print(f"\n[REPORT] Total cases processed for matrix: {len(case_report_data)}")
        print(f"[REPORT] Total images processed: {len(image_report_data)}")
        print(f"[REPORT] Cases matching ground truth (excluding Unknown predictions): {sum(1 for case in case_report_data if case['Match_Ground_Truth'] == True)}")

        return {
            "matrix": cm.tolist() if 'cm' in locals() and isinstance(cm, list) else [], # Ensure cm is list
            "labels": labels,
            "plot_path": plot_path,
            "report_path": report_path,
            "total_cases": len(case_report_data),
            "total_images": len(image_report_data)
        }

    def health_check(self) -> Dict:
        """Check server health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Health check failed: {e}")
            return {"status": "unreachable", "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='MedRAX Client')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Single image parser
    single_parser = subparsers.add_parser('single')
    single_parser.add_argument('image_path', help='Path to image file')
    single_parser.add_argument('--user-message', help='Optional message to include with the image')
    single_parser.add_argument('--force-tool', help='Force the use of a specific tool (e.g., LlavaMedTool)')

    # Batch images parser
    batch_parser = subparsers.add_parser('batch')
    batch_parser.add_argument('path', help='Path to image file or directory containing images')
    batch_parser.add_argument('--recursive', action='store_true',
                            help='Recursively scan directory for PNG files')
    batch_parser.add_argument('--user-message', help='Optional message to include with all images')
    batch_parser.add_argument('--force-tool', help='Force the use of a specific tool (e.g., LlavaMedTool)')
    batch_parser.add_argument('--ground_truth_excel', required=True, # Make GT mandatory for batch confusion matrix
                            help='Excel file containing ground truth labels (SCHE_NO, REP columns)')
    batch_parser.add_argument('--labels', nargs='+', required=True, # Make labels mandatory
                            help='Class labels for confusion matrix (e.g., Normal Abnormal Unknown)')
    batch_parser.add_argument('--voting-threshold', type=float, default=0.5,
                            help='Threshold for case-level voting (0.0 to 1.0, default: 0.5)')
    batch_parser.add_argument('--openai-api-key',
                            help='OpenAI API key for enhanced classification (required if not using basic keyword matching)')
    batch_parser.add_argument('--openai-endpoint',
                            help='OpenAI API endpoint',
                            default="https://api.openai.com/v1/chat/completions")
    batch_parser.add_argument('--openai-model',
                            help='OpenAI model name for classification',
                            default="gpt-4o-mini") # Consider faster/cheaper model if only for classification
    batch_parser.add_argument('--chunk-size', type=int, default=20, # Reduced default chunk size
                            help='Number of images to send per batch request (default: 20)')

    # Health check parser
    health_parser = subparsers.add_parser('health')

    args = parser.parse_args()

    # Initialize client - pass OpenAI details only if command is batch
    # Check if OpenAI key is provided for batch mode, as it's needed for reliable parsing
    if args.command == 'batch' and not args.openai_api_key:
         print("[WARNING] OpenAI API key not provided for batch mode. AI message parsing will rely on basic keyword matching and may be inaccurate.")
         # Decide if this should be a fatal error or just a warning
         # exit(1) # Uncomment to make API key mandatory

    client = MedRAXClient(
        openai_api_key=getattr(args, 'openai_api_key', None),
        openai_endpoint=getattr(args, 'openai_endpoint', "https://api.openai.com/v1/chat/completions"),
        openai_model=getattr(args, 'openai_model', "gpt-4o-mini")
    )


    if args.command == 'single':
        result = client.send_single_image(args.image_path, args.user_message, args.force_tool)
        print(json.dumps(result, indent=2))

    elif args.command == 'batch':
        # Get image paths
        image_paths = []
        if os.path.isdir(args.path):
            print(f"[INFO] Scanning directory: {args.path} (Recursive: {args.recursive})")
            if args.recursive:
                image_paths = client.find_png_files(args.path)
            else:
                image_paths = glob.glob(os.path.join(args.path, '*.png'))
            print(f"[INFO] Found {len(image_paths)} PNG files.")
        elif os.path.isfile(args.path):
             if args.path.lower().endswith('.png'):
                 image_paths = [args.path]
                 print(f"[INFO] Processing single file: {args.path}")
             else:
                  print(f"[ERROR] Specified file is not a PNG image: {args.path}")
                  return # Exit if single file is not PNG
        else:
             print(f"[ERROR] Path specified is not a valid file or directory: {args.path}")
             return # Exit if path is invalid

        if not image_paths:
            print("[ERROR] No PNG files found to process.")
            return # Exit if no files found

        # Process batch in chunks
        all_results_list = [] # Store results from successful chunks
        chunk_size = args.chunk_size
        if chunk_size <= 0:
             print("[ERROR] Chunk size must be positive.")
             return
        num_chunks = (len(image_paths) + chunk_size - 1) // chunk_size

        print(f"[INFO] Processing {len(image_paths)} images in {num_chunks} chunks of size {chunk_size}...")

        for i in range(0, len(image_paths), chunk_size):
            chunk_paths = image_paths[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            print(f"\n[INFO] Sending chunk {chunk_num}/{num_chunks} ({len(chunk_paths)} images)...")

            # Add a small delay between chunks if needed
            # time.sleep(1)

            chunk_response = client.send_batch_images(chunk_paths, args.user_message, args.force_tool)

            # Check status and results carefully
            if chunk_response and isinstance(chunk_response, dict):
                 chunk_status = chunk_response.get("status")
                 chunk_data = chunk_response.get("results", []) # Default to empty list

                 if chunk_status != "failed" and chunk_data:
                     print(f"[INFO] Chunk {chunk_num} processed successfully with {len(chunk_data)} results.")
                     all_results_list.extend(chunk_data)
                 elif chunk_data: # Even if status is failed, maybe partial results exist
                      print(f"[WARNING] Chunk {chunk_num} reported status '{chunk_status}' but contained {len(chunk_data)} results. Adding them.")
                      all_results_list.extend(chunk_data)
                 else: # No results data or status indicates failure
                      print(f"[ERROR] Chunk {chunk_num} failed. Status: '{chunk_status}'. Error: {chunk_response.get('error', 'N/A')}")
                      # Decide on error handling: continue, stop, retry?
                      # For now, continue to next chunk.
            else:
                 print(f"[ERROR] Chunk {chunk_num} returned an invalid or unexpected response format: {chunk_response}")


        print(f"\n[INFO] Finished processing all chunks. Total results collected: {len(all_results_list)}")

        # --- Confusion Matrix Calculation ---
        # Ensure ground truth and labels were provided (already checked by argparse 'required=True')
        try:
            # Load ground truth from Excel
            ground_truth = client.load_ground_truth(args.ground_truth_excel)

            # Validate voting threshold
            if not (0.0 <= args.voting_threshold <= 1.0):
                 print("[ERROR] Voting threshold must be between 0.0 and 1.0.")
                 return

            # Calculate confusion matrix using the aggregated results
            cm_result = client.calculate_confusion_matrix(
                all_results_list, # Use the aggregated list
                ground_truth,
                args.labels,
                args.voting_threshold
            )
            print("\n--- Confusion Matrix Results ---")
            if cm_result.get("plot_path"):
                 print(f"- Matrix plot saved to: {cm_result['plot_path']}")
            if cm_result.get("report_path"):
                 print(f"- Detailed report saved to: {cm_result['report_path']}")
            print(f"- Total cases processed for matrix: {cm_result.get('total_cases', 0)}")
            print(f"- Total images processed: {cm_result.get('total_images', 0)}")
            print(f"- Matrix (Labels: {cm_result.get('labels')}):")
            # Pretty print the matrix
            matrix = cm_result.get('matrix', [])
            labels = cm_result.get('labels', [])
            if matrix and labels:
                 header = "Pred -> | " + " | ".join(f"{l:<10}" for l in labels) + " |"
                 print("-" * len(header))
                 print(header)
                 print("-" * len(header))
                 for i, row in enumerate(matrix):
                      row_str = f"True {labels[i]:<6}| " + " | ".join(f"{x:<10}" for x in row) + " |"
                      print(row_str)
                 print("-" * len(header))
            else:
                 print("  (Matrix data not available)")
            print("--------------------------------")


        except FileNotFoundError:
             print(f"[ERROR] Ground truth file not found: {args.ground_truth_excel}")
        except ValueError as ve: # Catch specific errors like missing columns
             print(f"[ERROR] Problem with ground truth file: {ve}")
        except Exception as e:
             print(f"[ERROR] An unexpected error occurred during confusion matrix calculation: {e}")


    elif args.command == 'health':
        result = client.health_check()
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
