import argparse
import sys
import logging
from pathlib import Path
import json
from pdf2image import convert_from_path
from PIL import Image



from config import (
    SAMPLE_SCANS_DIR, SAMPLE_OUTPUT_DIR, RAW_OUTPUT_DIR,
    HEADER_CSV, LINES_CSV, MODEL_NAME, DEVICE, SUPPORTED_EXTENSIONS
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def load_dolphin_model():
    try:
        logger.info(f"Loading Dolphin model: {MODEL_NAME}")
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
        
        # TODO: Process images with Dolphin OCR
        logger.info("OCR processing will be implemented next")

    logger.info("Processing complete!")

if __name__ == "__main__":
    main()