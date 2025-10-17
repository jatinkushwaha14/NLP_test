
from pathlib import Path

BASE_DIR = Path(__file__).parent
SAMPLE_SCANS_DIR = BASE_DIR / "sample_scans"
SAMPLE_OUTPUT_DIR = BASE_DIR / "sample_output"
RAW_OUTPUT_DIR = SAMPLE_OUTPUT_DIR / "raw"

HEADER_CSV = "invoices_header.csv"
LINES_CSV = "invoices_lines.csv"

MODEL_NAME = "ByteDance/Dolphin"
DEVICE = "auto"  # Will auto-detect GPU/CPU

# OCR extraction patterns (regex)
PATTERNS = {
    "vendor_name": [
        # Generic patterns that work for any company name
        r"(?i)(?:from|vendor|bill\s+from|company|seller|supplier)[\s:]+([A-Z][A-Za-z\s.,&'-]{5,50})",  # After keywords
        r"^([A-Z][A-Z\s&.,'-]{5,50}(?:Ltd|LLC|Inc|Corp|Limited|Group|Company)?)",  # Company name at start of line
        r"(?i)(?:invoice\s+from|billed\s+by)[\s:]+([A-Z][A-Za-z\s.,&'-]{5,50})",  # After invoice-specific keywords
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}(?:\s+(?:Ltd|LLC|Inc|Corp|Limited|Group|Company|Co\.))?)",  # Title case company names
    ],
    
    "invoice_number": [
        # Generic invoice number patterns
        r"(?i)(?:invoice\s*(?:no|number|#)|inv\s*(?:no|#)|bill\s*(?:no|#)|ref(?:erence)?\s*(?:no|#))[\s:]*([A-Za-z0-9\-/]{3,20})",
        r"(?i)(?:invoice|bill)[\s:]+([A-Z]{2,4}[\-]?\d{4,10})",  # Like "INV-123456" or "BILL 456789"
        r"\b([A-Z]{2}\d{6,10})\b",  # Like "IN123456789"
        r"#\s*(\d{4,10})",  # Like "#123456"
    ],
    
    "invoice_date": [
        # Multiple date formats
        r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})",  # "17 Jul 2020" or "17 July 2020"
        r"(\d{1,2}[-/](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-/]\d{4})",  # "17-Jul-2020"
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})",  # "17/07/2020"
        r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",  # "2020-07-17" (ISO format)
        r"(?i)(?:date|invoice\s+date|bill\s+date|dated)[\s:]+(\d{1,2}[-/\s](?:\w{3}|\d{1,2})[-/\s]\d{2,4})",
    ],
    
    "currency": [
        # Standard currency codes
        r"\b(USD|EUR|GBP|INR|CAD|AUD|JPY|CNY|SGD|HKD|CHF|SEK|NOK|DKK|ZAR|NZD|MXN)\b",
        r"(?:^|\s)([A-Z]{3})(?:\s|$)",  # Any 3-letter currency code
    ],
    
    "total_amount": [
        # Various total amount patterns
        r"(?i)(?:total|grand\s+total|amount\s+due|total\s+due|final\s+total|balance\s+due|net\s+total)[\s:$€£¥]*([0-9,]+\.?[0-9]*)",
        r"(?:USD|EUR|GBP|INR|CAD|AUD|SGD)\s*([0-9,]+\.?[0-9]{2})",  # Currency code followed by amount
        r"[$€£¥]\s*([0-9,]+\.?[0-9]{2})",  # Currency symbol followed by amount
        r"\b([0-9]{1,3}(?:,[0-9]{3})+\.[0-9]{2})\b",  # Formatted numbers like "2,342,194.62"
        r"(?i)total[\s:]+([0-9,]+\.?[0-9]*)",  # Simple "Total: 12345.67"
    ],
}





SUPPORTED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
