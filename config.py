
import os
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
        r"(?i)(GuiComGroup[^\n]*)",
        r"(?i)(First Pacific Company[^\n]*)",
        r"(?i)(SHUM YIP[^\n]*)",
        r"(?i)(?:vendor|from|bill from|company|seller)[\s:]+([A-Za-z][A-Za-z\s.,&-]{5,50})",
        r"^([A-Z][A-Za-z\s.,&-]{10,50})$",
    ],
    "invoice_number": [
        r"(?i)(?:invoice\s*no|number|inv\s*no|bill\s*no|ref\s*no)[\s:#]+([A-Za-z0-9\-/]{3,20})",
        r"(\d{2}-\w{3}-\d{4})",  # Date patterns like invoice numbers
        r"(PR\d+\s+\d+\s+PA\d+)",  # Your specific pattern
    ],
    "invoice_date": [
        r"(\d{1,2}\s+\w{3}\s+\d{4})",  # "17 Jul 2020"
        r"(\d{1,2}[-/]\w{3}[-/]\d{4})",  # "17-Jul-2020"
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})",  # "17/07/2020"
        r"(?i)(?:date|invoice date|bill date)[\s:]+(\d{1,2}[-/\s]\w{3}[-/\s]\d{4})",
    ],
    "currency": [
        r"\b(USD|EUR|GBP|INR|CAD|AUD|JPY|CNY)\b",
        r"([A-Z]{3})[\s$][\d,]+\.?\d*",
    ],
    "total_amount": [
        r"(?i)(?:total|grand total|amount due|total due|final total|balance)[\s:$]*([0-9,]+\.?[0-9]*)",
        r"USD\s*([0-9,]+\.?[0-9]*)",  # "USD 2,342,194.62"
        r"\$\s*([0-9,]+\.?[0-9]*)",   # "$2,342,194.62"
        r"([0-9]{1,3}(?:,[0-9]{3})*\.[0-9]{2})",  # "2,342,194.62"
    ],
}


TABLE_KEYWORDS = [
"description", "quantity", "qty", "price", "amount", "total", 
"item", "product", "service", "rate", "unit", "cost"
]

SUPPORTED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]


MAX_WORKERS = 4
TIMEOUT_SECONDS = 120