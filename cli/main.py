import argparse
from pathlib import Path
from core.ocr import OCRService
from core.extract import InvoiceFieldExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError("File not found")

    ocr = OCRService()
    extractor = InvoiceFieldExtractor()

    ocr_result = ocr.extract_text(path)

    # Safety check
    if not ocr_result or not ocr_result.get("text", "").strip():
        print("No text detected in document")
        return

    print("OCR TEXT:")
    print(ocr_result.get("text", "").encode('utf-8').decode('utf-8', errors='replace'))
    print("\n" + "=" * 40)

    data = extractor.extract(ocr_result)

    print("\n" + "=" * 40)
    print("EXTRACTED FIELDS")
    print("=" * 40)
    for k, v in data.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
