from pathlib import Path
import pdfplumber
import pytesseract
from PIL import Image
import numpy as np
import cv2
from collections import defaultdict
import re


class OCRService:
    """
    Output format:
    {
        "text": full_text,
        "blocks": [
            {
                "text": "...",
                "x": int,
                "y": int,
                "w": int,
                "h": int,
                "page": int
            }
        ]
    }
    """

    def extract_text(self, path: Path) -> dict:
        if path.suffix.lower() == ".pdf":
            return self._extract_pdf(path)
        return self._extract_image(path)

    # ---------- PDF ----------
    def _extract_pdf(self, path: Path) -> dict:
        full_text = []
        blocks = []

        with pdfplumber.open(path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                # 1️⃣ Try native text — but don’t trust blindly
                native_text = page.extract_text() or ""
                if len(native_text.strip()) > 200:
                    full_text.append(native_text)

                # 2️⃣ Always OCR page image
                img = page.to_image(resolution=300).original
                t, b = self._ocr_image(img, page_idx)
                full_text.append(t)
                blocks.extend(b)

        return {
            "text": "\n".join(full_text),
            "blocks": blocks
        }

    # ---------- IMAGE ----------
    def _extract_image(self, path: Path) -> dict:
        image = Image.open(path)
        text, blocks = self._ocr_image(image, page_idx=0)
        return {
            "text": text,
            "blocks": blocks
        }

    # ---------- CORE OCR ----------
    def _ocr_image(self, image: Image.Image, page_idx: int):
        img = np.array(image)

        # Basic preprocessing
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Detect language for better OCR
        detected_lang = self._detect_language(gray)

        # Try basic OCR first
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31, 15
        )

        config = self._get_ocr_config(detected_lang)
        data = pytesseract.image_to_data(
            thresh,
            output_type=pytesseract.Output.DICT,
            config=config
        )

        # Check if we got reasonable text (more than 50 non-empty strings)
        text_count = len([t for t in data["text"] if t.strip()])
        if text_count < 50:
            # Try enhanced preprocessing
            enhanced_thresh = self._preprocess_image(img)
            enhanced_data = pytesseract.image_to_data(
                enhanced_thresh,
                output_type=pytesseract.Output.DICT,
                config=config
            )
            enhanced_text_count = len([t for t in enhanced_data["text"] if t.strip()])
            if enhanced_text_count > text_count:
                data = enhanced_data

        # ---- GROUP WORDS INTO LINES ----
        lines = defaultdict(list)

        for i, txt in enumerate(data["text"]):
            txt = txt.strip()
            if not txt:
                continue

            line_id = (
                data["page_num"][i],
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i],
            )

            lines[line_id].append({
                "text": txt.upper(),
                "x": data["left"][i],
                "y": data["top"][i],
                "w": data["width"][i],
                "h": data["height"][i],
            })

        blocks = []
        full_text_lines = []

        for words in lines.values():
            words.sort(key=lambda w: w["x"])

            text = " ".join(w["text"] for w in words)

            x = min(w["x"] for w in words)
            y = min(w["y"] for w in words)
            w = max(w["x"] + w["w"] for w in words) - x
            h = max(w["y"] + w["h"] for w in words) - y

            blocks.append({
                "text": text,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "page": page_idx
            })

            full_text_lines.append(text)

        return "\n".join(full_text_lines), blocks

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for better OCR accuracy on invoices."""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Deskewing for skewed images
        gray = self._deskew_image(gray)

        # Noise reduction
        gray = cv2.medianBlur(gray, 3)

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Sharpening for blurry text
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)

        # Try multiple thresholding methods
        thresh_methods = [
            cv2.adaptiveThreshold(
                cleaned, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            ),
            cv2.adaptiveThreshold(
                cleaned, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                15, 3
            ),
            cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)[1],
        ]

        # Select the best threshold based on text quality
        best_thresh = thresh_methods[0]
        best_score = 0

        for thresh in thresh_methods:
            text = pytesseract.image_to_string(thresh, config="--oem 3 --psm 6")
            # Score based on text length and alphanumeric content
            score = len(text.strip())
            alpha_num = len([c for c in text if c.isalnum()])
            score += alpha_num * 0.5

            if score > best_score:
                best_score = score
                best_thresh = thresh

        return best_thresh

    def _deskew_image(self, img: np.ndarray) -> np.ndarray:
        """Deskew image to correct for rotation/skew."""
        # Find all contours
        contours, _ = cv2.findContours(
            cv2.bitwise_not(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return img

        # Find the largest contour (likely the main text area)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]

        # Correct angle if needed
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        # Only deskew if angle is significant
        if abs(angle) > 1:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return deskewed

        return img

    def _detect_language(self, img: np.ndarray) -> str:
        """Detect the primary language in the image."""
        try:
            # Try to detect script using OSD (Orientation and Script Detection)
            osd = pytesseract.image_to_osd(img)
            script_lines = [line for line in osd.split('\n') if 'Script:' in line]

            if script_lines:
                script = script_lines[0].split(':')[1].strip()
                # Map common scripts to language codes
                script_to_lang = {
                    'Latin': 'eng',
                    'Devanagari': 'hin',
                    'Arabic': 'ara',
                    'Cyrillic': 'rus',
                    'Chinese': 'chi_sim',
                    'Japanese': 'jpn',
                    'Korean': 'kor',
                }
                return script_to_lang.get(script, 'eng')

            # Fallback: try OCR with different languages and see which gives most text
            langs_to_try = ['eng', 'hin', 'ara', 'rus']
            best_lang = 'eng'
            best_score = 0

            for lang in langs_to_try:
                try:
                    text = pytesseract.image_to_string(img, lang=lang, config="--oem 3 --psm 6")
                    score = len(text.strip())
                    if score > best_score:
                        best_score = score
                        best_lang = lang
                except:
                    continue

            return best_lang

        except:
            return 'eng'  # Default to English

    def _get_ocr_config(self, lang: str) -> str:
        """Get appropriate OCR configuration for the detected language."""
        base_config = "--oem 3 --psm 6"

        # Language-specific configurations
        lang_configs = {
            'eng': base_config,
            'hin': f"{base_config} -l hin+eng",  # Hindi with English fallback
            'ara': f"{base_config} -l ara+eng",  # Arabic with English fallback
            'rus': f"{base_config} -l rus+eng",  # Russian with English fallback
            'chi_sim': f"{base_config} -l chi_sim+eng",  # Chinese with English fallback
            'jpn': f"{base_config} -l jpn+eng",  # Japanese with English fallback
            'kor': f"{base_config} -l kor+eng",  # Korean with English fallback
        }

        return lang_configs.get(lang, base_config)
