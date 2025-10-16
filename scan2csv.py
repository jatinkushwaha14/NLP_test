import argparse
import sys
import logging
from pathlib import Path
import json
from pdf2image import convert_from_path
from PIL import Image
import torch


from config import (
    SAMPLE_SCANS_DIR, SAMPLE_OUTPUT_DIR, RAW_OUTPUT_DIR,
    HEADER_CSV, LINES_CSV, MODEL_NAME, DEVICE, SUPPORTED_EXTENSIONS
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def load_dolphin_model():
    try:
        logger.info(f"Loading Dolphin model:{MODEL_NAME}")
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch
        
        # Auto-detect device
        if DEVICE == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = DEVICE
            
        logger.info(f"Using device:{device}")

        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME)
        model.to(device)
        
        logger.info("Dolphin model loaded")
        return {"model": model, "processor": processor, "device": device}
        
    except Exception as e:
        logger.error(f"Failed to load Dolphin model: {e}")
        return None

def convert_pdf_to_images(pdf_path):
    try:
        logger.info(f"Converting PDF to images: {pdf_path.name}")
        images = convert_from_path(pdf_path, dpi=200)
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

def process_image_with_dolphin(image,dolphin_model,file_name):
    try:
        logger.info(f"Running OCR on {file_name}")
        
        model = dolphin_model["model"]
        processor = dolphin_model["processor"]
        device = dolphin_model["device"]
        
        # Prepare the image for Dolphin
        inputs = processor(images=image,return_tensors="pt")
        inputs = inputs.to(device)
        
        # Generate OCR output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1000)
        
        # Decode the result
        generated_text = processor.batch_decode(generated_ids,skip_special_tokens=True)[0]
        
        logger.info(f"OCR completed for {file_name}")
        return generated_text
        
    except Exception as e:
        logger.error(f"OCR failed for {file_name}: {e}")
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

    for file_path in supported_files:
        logger.info(f"Processing: {file_path.name}")
        
        if file_path.suffix.lower() == '.pdf':
            images = convert_pdf_to_images(file_path)
            logger.info(f"PDF converted to {len(images)} images")
        else:
            try:
                image = Image.open(file_path)
                images = [image]
                logger.info("Image loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load image {file_path.name}: {e}")
                continue
        
        # Process images with Dolphin OCR
        ocr_results = []
        for i, image in enumerate(images):
            page_name = f"{file_path.stem}_page_{i+1}"
            ocr_text = process_image_with_dolphin(image, dolphin, page_name)
            
            if ocr_text:
                ocr_results.append({
                    "file": file_path.name,
                    "page": i+1,
                    "ocr_text": ocr_text
                })
        
        # Save raw OCR output to JSON
        if ocr_results:
            output_file = RAW_OUTPUT_DIR / f"{file_path.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(ocr_results, f, indent=2)
            logger.info(f"Raw OCR saved to {output_file}")


    logger.info("Processing complete!")

if __name__ == "__main__":
    main()