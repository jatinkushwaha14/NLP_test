import argparse
import sys
import logging
from pathlib import Path
import json
from pdf2image import convert_from_path
from PIL import Image
import torch
import cv2
import numpy as np

import re
import csv
from config import (
    SAMPLE_SCANS_DIR, SAMPLE_OUTPUT_DIR, RAW_OUTPUT_DIR,
    HEADER_CSV, LINES_CSV, MODEL_NAME, DEVICE, SUPPORTED_EXTENSIONS
)

def preprocess_image_for_ocr(pil_image):
    # Convert PIL image to OpenCV format
    cv_image = np.array(pil_image)
    if cv_image.ndim == 3:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter (smooth while preserving edges)
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Thresholding (binarization)
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert back to PIL
    processed_pil = Image.fromarray(thresh)

    return processed_pil

def extract_fields_with_regex(text):

    from config import PATTERNS
    
    fields = {}
    for field_name, pattern_list in PATTERNS.items():
        fields[field_name] = ""  
        for i, pattern in enumerate(pattern_list):

            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted_text = match.group(1).strip()
                extracted_text = re.sub(r'\s+', ' ', extracted_text)
                fields[field_name] = extracted_text
                logger.info(f"**Found {field_name}: '{extracted_text}' using pattern {i}")
                break 
        
        if not fields[field_name]:
            logger.debug(f"No match found for {field_name}")
    
    return fields

def save_extracted_fields_to_csv(all_results, csv_path):
    csv_data = []
    
    for file_results in all_results:
        for result in file_results:
            # Extract fields from successful OCR
            if result.get("extracted_fields"):
                fields = result["extracted_fields"]
                csv_data.append({
                    "filename": result["file"],
                    "vendor_name": fields.get("vendor_name", ""),
                    "invoice_number": fields.get("invoice_number", ""),  
                    "invoice_date": fields.get("invoice_date", ""),
                    "currency": fields.get("currency", ""),
                    "total_amount": fields.get("total_amount", "")
                })
    
    # Write CSV
    if csv_data:
        fieldnames = ["filename", "vendor_name", "invoice_number", "invoice_date", "currency", "total_amount"]
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        logger.info(f"CSV saved to {csv_path} with {len(csv_data)} records")
    else:
        logger.warning("No extracted fields found for CSV generation")


def save_line_items_to_csv(all_results, csv_path):
    """Save extracted line items to CSV."""
    csv_data = []
    
    for file_results in all_results:
        for result in file_results:
            filename = result["file"]

            csv_data.append({
                "filename": filename,
                "line_number": 1,
                "description": "Sample line item", 
                "quantity": 1,
                "unit_price": 0.00,
                "amount": 0.00
            })
    
    # Write CSV
    if csv_data:
        fieldnames = ["filename", "line_number", "description", "quantity", "unit_price", "amount"]
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        logger.info(f"Lines CSV saved to {csv_path} with {len(csv_data)} records")

    
def process_file_dolphin_only(file_path, dolphin):
    results = []
    
    if file_path.suffix.lower() == '.pdf':
        # First try direct PDF text extraction
        logger.info(f"Using Dolphin OCR for {file_path.name}")
        images = convert_pdf_to_images(file_path)
        for i, image in enumerate(images):
            image = preprocess_image_for_ocr(image)
            page_name = f"{file_path.stem}_page_{i+1}"
            ocr_text = process_image_with_dolphin(image, dolphin, page_name)

            results.append({
                "file": file_path.name,
                "method": "dolphin_ocr",
                "page": i+1,
                "raw_text": ocr_text,
                "text_length": len(ocr_text)
            })
            if ocr_text and len(ocr_text) > 50:
                extracted_fields = extract_fields_with_regex(ocr_text)
                results[-1]["extracted_fields"] = extracted_fields

    return results


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def load_dolphin_model():
    try:
        logger.info(f"Loading Dolphin model:{MODEL_NAME}")
        from transformers import AutoProcessor, VisionEncoderDecoderModel
        import torch
        
        # Auto-detect device
        if DEVICE == "auto":
            if torch.backends.mps.is_available():
                device = "mps"  # Mac GPU (Metal)
            elif torch.cuda.is_available():
                device = "cuda"  # NVIDIA GPU
            else:
                device = "cpu"
        else:
            device = DEVICE
            
        logger.info(f"Using device:{device}")

        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
        
        logger.info("Dolphin model loaded")
        return {"model": model, "processor": processor, "device": device, "tokenizer": processor.tokenizer}
        
    except Exception as e:
        logger.error(f"Failed to load Dolphin model: {e}")
        return None

def convert_pdf_to_images(pdf_path):
    try:
        logger.info(f"Converting PDF to images: {pdf_path.name}")
        images = convert_from_path(
            pdf_path, 
            dpi=300,  # Increased from 200
            grayscale=True,  # Better for OCR
            fmt='PNG'
        )
        logger.info(f"Converted {len(images)} pages from PDF")
        return images
    except Exception as e:
        logger.error(f"Failed to convert PDF {pdf_path.name}: {e}")
        return []


def find_supported_files(input_dir):
    input_path = Path(input_dir)
    supported_files = []
    
    for ext in SUPPORTED_EXTENSIONS:
        files = list(input_path.glob(f"*{ext}"))
        supported_files.extend(files)
    
    logger.info(f"Found {len(supported_files)} supported files")
    for file in supported_files:
        logger.info(f" - {file.name}")
    
    return supported_files

def process_image_with_dolphin(image, dolphin_model, file_name):
    try:
        logger.info(f"Running Dolphin document parsing on {file_name}")
        
        model = dolphin_model["model"]
        processor = dolphin_model["processor"]
        device = dolphin_model["device"]
        tokenizer = processor.tokenizer
        # Use Dolphin's correct prompt for document parsing
        prompt = "Read text in the image."
        full_prompt = f"<s>{prompt} <Answer/>"  
        
        # Prepare inputs for Dolphin
        inputs = processor(image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)
        
        prompt_ids = tokenizer(full_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        decoder_attention_mask = torch.ones_like(prompt_ids).to(device)
        # Generate document structure
        with torch.no_grad():
            outputs = model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=prompt_ids,
                decoder_attention_mask=decoder_attention_mask,
                max_new_tokens=2048,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode output
        parsed_content = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Remove prompt from output
        if prompt in parsed_content:
            parsed_content = parsed_content.replace(prompt, "").strip()
        if "<Answer/>" in parsed_content:
            parsed_content = parsed_content.split("<Answer/>")[-1].strip()
            
        logger.info(f"Dolphin document parsing completed for {file_name}")
        logger.info(f"Parsed content length: {len(parsed_content)}")
        logger.info(f"Content preview: {parsed_content[:200]}...")
        
        return parsed_content
        
    except Exception as e:
        logger.error(f"Dolphin document parsing failed for {file_name}: {e}")
        return ""
    


def main():
    parser = argparse.ArgumentParser(description="Convert scanned invoices to CSV/JSON using Dolphin OCR")
    parser.add_argument("--in_dir", type=str, required=True, help="Input directory containing scanned invoices")
    parser.add_argument("--out_csv", type=str, default="invoices_header.csv", help="Output CSV file for invoice headers")
    parser.add_argument("--out_json", type=str, default="invoices_raw.json", help="Output JSON file for raw OCR data")
    
    args = parser.parse_args()

    logger.info(f"Input directory: {args.in_dir}")
    logger.info(f"Output CSV: {args.out_csv}")
    logger.info(f"Output JSON: {args.out_json}")

    input_path = Path(args.in_dir)
    if not input_path.exists():
        logger.error(f"Input directory '{args.in_dir}' does not exist")
        sys.exit(1)
    
    SAMPLE_OUTPUT_DIR.mkdir(exist_ok=True)
    RAW_OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info("Output directories created")

    dolphin = load_dolphin_model()
    if dolphin is None:
        logger.error("Failed to load Dolphin model. Exiting.")
        sys.exit(1)
    
    supported_files = find_supported_files(args.in_dir)
    if not supported_files:
        logger.warning("No supported files found")
        return
    all_results = []
    for file_path in supported_files:
        logger.info(f"Processing: {file_path.name}")
        
        results = process_file_dolphin_only(file_path, dolphin)
        all_results.append(results)
        
        # Save individual JSON results
        if results:
            output_file = RAW_OUTPUT_DIR / f"{file_path.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Processed output saved to {output_file}")

    # Generate final CSV with all extracted fields

    # Generate final CSV with all extracted fields  
    header_csv_path = SAMPLE_OUTPUT_DIR / HEADER_CSV
    lines_csv_path = SAMPLE_OUTPUT_DIR / LINES_CSV

    save_extracted_fields_to_csv(all_results, header_csv_path)
    save_line_items_to_csv(all_results, lines_csv_path)  # We'll create this next

    logger.info(f"Header CSV saved to: {header_csv_path}")
    logger.info(f"Lines CSV saved to: {lines_csv_path}")
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()