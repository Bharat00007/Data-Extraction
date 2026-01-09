# Invoice Data Extraction System (OCR-Based)

A Python-based system for extracting structured data from invoices (PDFs and images) using Optical Character Recognition (OCR) and rule-based parsing.  
The project is designed to handle real-world invoices, including scanned, skewed, and low-quality documents, and is suitable for backend integration with web applications (e.g., React via APIs).

---

## Features

- Supports **PDF and image invoices**
- Hybrid extraction:
  - Native PDF text extraction (pdfplumber)
  - OCR fallback using Tesseract
- Advanced image preprocessing:
  - Deskewing
  - Noise reduction
  - Contrast enhancement (CLAHE)
  - Adaptive thresholding
- Extracts key invoice fields:
  - Supplier GSTIN
  - Customer GSTIN
  - VAT number
  - Invoice number
  - Grand total
  - Currency (INR / USD / EUR)
  - Bank name
  - Account number
  - IFSC code
- OCR error correction for GSTIN and numeric fields
- Heuristic OCR quality assessment
- Command-line interface (CLI) for testing and debugging

---


## Requirements

- Python **3.8+**
- Tesseract OCR (installed system-wide)

### Python Dependencies

Install required packages:

```bash
pip install -r requirements.txt

