from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from pathlib import Path

app = FastAPI(title="Invoice Data Extraction API", description="API for extracting data from invoice PDFs and images")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize services (lazy-loaded to avoid import errors)
ocr_service = None
extractor = None

def init_services():
    global ocr_service, extractor
    if ocr_service is None:
        try:
            from core.ocr import OCRService
            from core.extract import InvoiceFieldExtractor
            ocr_service = OCRService()
            extractor = InvoiceFieldExtractor()
        except Exception as e:
            print(f"Warning: Could not initialize OCR services: {e}")
            raise HTTPException(status_code=503, detail="OCR services not available")

@app.get("/")
async def root():
    return {"message": "Invoice Data Extraction API", "version": "1.0.0", "status": "ok"}

@app.post("/extract")
async def extract_invoice_data(file: UploadFile = File(...)):
    """
    Extract invoice data from uploaded PDF or image file.
    """
    init_services()
    
    # Validate file type
    allowed_extensions = {".pdf", ".jpg", ".jpeg", ".png"}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only PDF, JPG, JPEG, and PNG files are allowed")

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    try:
        # Process the file
        path = Path(tmp_path)
        ocr_result = ocr_service.extract_text(path)

        # Safety check
        if not ocr_result or not ocr_result.get("text", "").strip():
            raise HTTPException(status_code=400, detail="No text detected in document")

        # Extract data
        result = extractor.extract(ocr_result)

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

