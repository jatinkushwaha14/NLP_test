import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert scanned invoices to CSV/JSON using Dolphin OCR")
    parser.add_argument("--in_dir", type=str, required=True, help="Input directory containing scanned invoices")
    parser.add_argument("--out_csv", type=str, default="invoices_header.csv", help="Output CSV file for invoice headers")
    parser.add_argument("--out_json", type=str, default="invoices_raw.json", help="Output JSON file for raw OCR data")

    args = parser.parse_args()

    print(f"Input directory: {args.in_dir}")
    print(f"Output CSV: {args.out_csv}")
    print(f"Output JSON: {args.out_json}")

    # TODO: Implement OCR pipeline
    print("Pipeline not implemented yet")

if __name__ == "__main__":
    main()