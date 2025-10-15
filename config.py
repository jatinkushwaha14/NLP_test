
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
r"(?i)(?:vendor|from|bill\s+from|company|seller)[\s:]*([^\n\r]{1,100})",
r"(?i)(?:issued\s+by|billed\s+by)[\s:]*([^\n\r]{1,100})",
r"^([A-Z][A-Za-z\s&.,()-]{5,50})\s*\n",
],
"invoice_number": [
r"(?i)(?:invoice\s*(?:no|number|#)?|inv\s*(?:no|#)?|bill\s*(?:no|#)?)[\s:]*([A-Za-z0-9-/]{3,20})",
r"(?i)(?:receipt\s*(?:no|#)?)[\s:]*([A-Za-z0-9-/]{3,20})",
],
"invoice_date": [
r"(?i)(?:date|invoice\s+date|bill\s+date)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
r"(?i)(?:date|invoice\s+date|bill\s+date)[\s:]*(\d{4}[/-]\d{1,2}[/-]\d{1,2})",
r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
],
"currency": [
r"(?i)(USD|EUR|GBP|INR|CAD|AUD|JPY|CNY|\$|€|£|₹|¥)",
r"([A-Z]{3})\s*[\d,.]",
],
"total_amount": [
r"(?i)(?:total|grand\s+total|amount\s+due|total\s+due|final\s+total)[\s:]*([A-Z]{0,3}\s*[\$€£₹¥]?\s*[\d,]+\.?\d*)",
r"(?i)(?:balance\s+due|net\s+total)[\s:]*([A-Z]{0,3}\s*[\$€£₹¥]?\s*[\d,]+\.?\d*)",
],
}

TABLE_KEYWORDS = [
"description", "quantity", "qty", "price", "amount", "total", 
"item", "product", "service", "rate", "unit", "cost"
]

SUPPORTED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]


MAX_WORKERS = 4
TIMEOUT_SECONDS = 120