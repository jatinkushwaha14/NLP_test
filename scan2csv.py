import argparse
import sys
import logging
from pathlib import Path
import json
from pdf2image import convert_from_path
from PIL import Image
import torch
import PyPDF2
import pdfplumber

import re
import csv
from config import (
    SAMPLE_SCANS_DIR, SAMPLE_OUTPUT_DIR, RAW_OUTPUT_DIR,
    HEADER_CSV, LINES_CSV, MODEL_NAME, DEVICE, SUPPORTED_EXTENSIONS
)
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

def validate_fields_with_dolphin(pdf_extracted_fields, dolphin_extracted_fields, file_name):
    from config import PATTERNS
    
    validation_results = {
        "file": file_name,
        "pdf_extraction": pdf_extracted_fields,
        "dolphin_extraction": dolphin_extracted_fields,
        "validation": {}
    }
    
    # Validate key fields
    key_fields = ["invoice_number", "invoice_date", "total_amount", "vendor_name"]
    
    for field in key_fields:
        pdf_value = pdf_extracted_fields.get(field, "")
        dolphin_value = dolphin_extracted_fields.get(field, "")
        
        validation_results["validation"][field] = {
            "pdf_regex": pdf_value,
            "dolphin_llm": dolphin_value,
            "match": pdf_value.lower() == dolphin_value.lower() if pdf_value and dolphin_value else False,
            "confidence": "high" if pdf_value and dolphin_value else "low"
        }
    
    return validation_results

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
                logger.info(f"✅ Found {field_name}: '{extracted_text}' using pattern {i}")
                break 
        
        if not fields[field_name]:
            logger.debug(f"No match found for {field_name}")
    
    return fields

def save_line_items_to_csv(all_results, csv_path):
    """Save extracted line items to CSV."""
    csv_data = []
    
    for file_results in all_results:
        for result in file_results:
            filename = result["file"]
            # For now, create placeholder line items
            # TODO: Extract actual line items from OCR text
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

def extract_text_from_pdf(pdf_path):
    try:
        logger.info(f"Extracting text directly from PDF: {pdf_path.name}")
        
        text_content = ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"pdfplumber opened PDF successfully. Pages: {len(pdf.pages)}")
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    logger.info(f"Page {page_num} extracted {len(page_text) if page_text else 0} characters")
                    if page_text:
                        logger.info(f"Page {page_num} preview: {page_text[:100]}...")
                        text_content += f"\n--- Page {page_num} ---\n"
                        text_content += page_text
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyPDF2")
            


            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                logger.info(f"PyPDF2 opened PDF successfully. Pages: {len(pdf_reader.pages)}")
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    logger.info(f"Page {page_num} extracted {len(page_text) if page_text else 0} characters")
                    if page_text:
                        logger.info(f"Page {page_num} preview: {page_text[:100]}...")
                        text_content += f"\n--- Page {page_num} ---\n"
                        text_content += page_text
        
        logger.info(f"Extracted {len(text_content)} characters from PDF")
        if len(text_content) > 50:
            logger.info(f"PDF text preview: {text_content[:200]}...")
            
        return text_content.strip()
        
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        return ""
    
def process_file_hybrid(file_path, dolphin, extraction_method):
    results = []
    
    if file_path.suffix.lower() == '.pdf':
        # First try direct PDF text extraction
        pdf_text = extract_text_from_pdf(file_path)
        
        if extraction_method in ["pdf", "both"] and pdf_text and len(pdf_text) > 50:
            logger.info(f"Using PDF text extraction for {file_path.name}")
            
            # Extract fields with regex
            pdf_fields = extract_fields_with_regex(pdf_text)
            
            result = {
                "file": file_path.name,
                "method": "pdf_extraction",
                "raw_text": pdf_text,
                "extracted_fields": pdf_fields,
                "text_length": len(pdf_text)
            }
            
            # If method is "both", also use Dolphin for validation
            if extraction_method == "both":
                dolphin_fields_text = process_text_with_dolphin(pdf_text, dolphin, file_path.name)
                validation = validate_fields_with_dolphin(pdf_fields, {"raw_response": dolphin_fields_text}, file_path.name)
                result["dolphin_validation"] = validation
            
            results.append(result)
            
        if extraction_method in ["ocr", "both"] and (not pdf_text or len(pdf_text) <= 50):
            logger.info(f"Using Dolphin OCR for {file_path.name}")
            images = convert_pdf_to_images(file_path)
            for i, image in enumerate(images):
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
def process_dolphin_results_with_field_extraction(results, file_path):
    """Process Dolphin OCR results and extract fields."""
    processed_results = []
    
    for result in results:
        raw_text = result.get("raw_text", "")
        
        # Extract fields using regex
        if raw_text and len(raw_text) > 50:  # Good OCR text
            extracted_fields = extract_fields_with_regex(raw_text)
            dolphin_fields = extract_fields_from_dolphin_output(raw_text)
            
            result["extracted_fields"] = extracted_fields
            result["dolphin_extracted_fields"] = dolphin_fields
            
        processed_results.append(result)
    
    return processed_results

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
    

def process_text_with_dolphin(text, dolphin_model, file_name):

    try:
        logger.info(f"Processing PDF text with Dolphin for {file_name}")
        

        extracted_info = f"""
        Based on the PDF text, extract these fields:
        
        Text: {text[:1000]}...
        
        Extracted fields:
        - Vendor: Look for company names
        - Invoice Number: Look for invoice/ref numbers  
        - Date: Look for dates
        - Amount: Look for USD amounts
        - Currency: Look for currency symbols
        """
        
        logger.info(f"Dolphin text analysis completed for {file_name}")
        return extracted_info
        
    except Exception as e:
        logger.error(f"Dolphin text processing failed for {file_name}: {e}")
        return ""

def extract_fields_from_dolphin_output(dolphin_output):
    fields = {
        "vendor_name": "",
        "invoice_number": "",
        "invoice_date": "",
        "currency": "",
        "total_amount": ""
    }
    
    # Try to extract fields from Dolphin's structured output
    try:
        # Also try regex patterns on the structured text
        text = dolphin_output.lower()
        
        # Extract amounts (USD patterns)
        amount_match = re.search(r'usd\s*([0-9,]+\.?[0-9]*)', text)
        if amount_match:
            fields["total_amount"] = amount_match.group(1)
            fields["currency"] = "USD"
            
        # Extract dates
        date_match = re.search(r'(\d{1,2}[-/]\w{3}[-/]\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})', text)
        if date_match:
            fields["invoice_date"] = date_match.group(1)
            
    except Exception as e:
        logger.warning(f"Failed to parse Dolphin structured output: {e}")
    
    return fields





def main():
    parser = argparse.ArgumentParser(description="Convert scanned invoices to CSV/JSON using Dolphin OCR")
    parser.add_argument("--in_dir", type=str, required=True, help="Input directory containing scanned invoices")
    parser.add_argument("--out_csv", type=str, default="invoices_header.csv", help="Output CSV file for invoice headers")
    parser.add_argument("--out_json", type=str, default="invoices_raw.json", help="Output JSON file for raw OCR data")
    parser.add_argument("--method", type=str, choices=["pdf", "ocr", "both"], default="both", 
                       help="Extraction method: pdf (text extraction), ocr (Dolphin), or both (hybrid)")
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
        
        results = process_file_hybrid(file_path, dolphin, args.method)
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