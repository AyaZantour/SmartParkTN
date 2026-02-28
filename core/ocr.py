"""
SmartParkTN – OCR engine (PaddleOCR + Arabic support)
Reads Tunisian license plates and validates the format.

Tunisian plate formats:
  Modern  : NNN تونس NNNN  (Arabic: تونس = Tunisia)
  Latin   : NNN TN NNNN    (same, Latin script)
  Legacy  : NNN RS NNNN    (older format)
  Numeric : NNNNNNN        (pure digits, legacy)
"""
from __future__ import annotations
import re, os
import numpy as np
import cv2
from typing import Optional, Tuple, List
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

_CONF_THRESH = float(os.getenv("OCR_CONF_THRESHOLD", "0.55"))

# ── Arabic "تونس" variants (OCR may misread some strokes) ─────────────────
# All of these mean "Tunisia" in Arabic and must map to "TN"
_TOUNES_VARIANTS = [
    "تونس", "تو نس", "توﻧﺲ", "تﻮنس", "تونـس", "ﺗﻮﻧﺲ",
    # Common OCR misreads of تونس:
    "نونس", "تونت", "توبس", "تونب", "تونح", "توكس",
    "توئس", "ثونس", "تونص", "تونز",
]

# Arabic-Indic digit map (٠١٢... → 0123...)
_AR_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Common OCR character confusion corrections for plate context
_CHAR_FIXES = {
    "O": "0", "Q": "0",          # O/Q → 0
    "I": "1", "L": "1",          # I/L → 1
    "Z": "2",                     # Z → 2
    "S": "5",                     # S → 5  (in digit positions)
    "G": "6", "b": "6",          # G → 6
    "B": "8",                     # B → 8
    "A": "4",                     # A → 4  (rare)
}

# Tunisian plate regex patterns (flexible separators)
_PATTERNS = [
    re.compile(r"(\d{1,3})\s*TN\s*(\d{1,4})", re.IGNORECASE),   # 100 TN 1234 (modern)
    re.compile(r"(\d{1,3})\s*RS\s*(\d{1,4})", re.IGNORECASE),   # 100 RS 1234 (legacy)
    re.compile(r"\d{4,8}"),                                        # pure numeric
]


def _map_tounes(text: str) -> str:
    """Replace all Arabic 'تونس' variants with 'TN' before stripping Arabic."""
    for variant in _TOUNES_VARIANTS:
        text = text.replace(variant, " TN ")
    return text


def _arabic_to_latin(text: str) -> str:
    """Convert Arabic-Indic digits to Western digits."""
    return text.translate(_AR_DIGIT_MAP)


def _apply_char_fixes(text: str) -> str:
    """
    Fix common OCR confusion characters in numeric positions.
    Only replaces alpha chars that appear between/around digit blocks.
    E.g. '1O0 TN 123S' → '100 TN 1235'
    """
    # Fix digits-only segments: replace confusable letters with digits
    def fix_segment(seg: str) -> str:
        if re.match(r'^[0-9A-Z]+$', seg):
            result = []
            for i, ch in enumerate(seg):
                # Check if context is numeric (neighbours are digits)
                neighbours_digits = (
                    (i > 0 and seg[i-1].isdigit()) or
                    (i < len(seg)-1 and seg[i+1].isdigit())
                )
                if ch in _CHAR_FIXES and neighbours_digits:
                    result.append(_CHAR_FIXES[ch])
                else:
                    result.append(ch)
            return "".join(result)
        return seg

    parts = text.split()
    fixed = []
    for part in parts:
        if part.upper() not in ("TN", "RS"):
            fixed.append(fix_segment(part.upper()))
        else:
            fixed.append(part.upper())
    return " ".join(fixed)


def _normalize(raw: str) -> str:
    """Full normalization pipeline: Arabic digits → Arabic words → ASCII."""
    text = _arabic_to_latin(raw.strip())   # ١٢٣ → 123
    text = _map_tounes(text)                # تونس → TN
    text = text.upper()
    text = re.sub(r"[^A-Z0-9\s]", " ", text)  # strip remaining non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _validate(text: str) -> Optional[str]:
    """Return canonical plate string if it matches a known Tunisian format."""
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            # Reconstruct canonical form for TN/RS plates
            if pat.groups == 2:
                left, right = m.group(1), m.group(2)
                sep = "TN" if "TN" in text else "RS"
                return f"{left} {sep} {right}"
            return text
    return None


# ── Image preprocessing variants ──────────────────────────────────────────

def _deskew(img: np.ndarray) -> np.ndarray:
    """Correct skew up to ±15 degrees using projection profile method."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        return img
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def _preprocess_variants(img: np.ndarray) -> List[np.ndarray]:
    """
    Return a list of preprocessed image variants for multi-pass OCR.
    Each variant targets different real-world conditions.
    """
    if img is None or img.size == 0:
        return [img]

    # Upscale if too small (minimum 300px wide for good OCR)
    h, w = img.shape[:2]
    if w < 300:
        scale = 300 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)

    variants = []

    # Variant 1: CLAHE + bilateral (standard, good for normal conditions)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    g1 = clahe.apply(gray)
    g1 = cv2.bilateralFilter(g1, 9, 75, 75)
    variants.append(cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR))

    # Variant 2: Sharpened (good for blurry/low-res plates)
    v2 = cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR)
    variants.append(_sharpen(v2))

    # Variant 3: Otsu threshold (good for high-contrast plates)
    _, g3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(g3, cv2.COLOR_GRAY2BGR))

    # Variant 4: Inverted Otsu (for dark-background plates)
    g4 = cv2.bitwise_not(g3)
    variants.append(cv2.cvtColor(g4, cv2.COLOR_GRAY2BGR))

    # Variant 5: Deskewed + CLAHE (good for angled plates)
    try:
        deskewed = _deskew(cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR))
        variants.append(deskewed)
    except Exception:
        pass

    # Variant 6: Adaptive threshold (good for uneven lighting / night)
    g6 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 15, 8)
    g6 = cv2.morphologyEx(g6, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    variants.append(cv2.cvtColor(g6, cv2.COLOR_GRAY2BGR))

    return variants


# Keep old name for backward compat
def _preprocess(img: np.ndarray) -> np.ndarray:
    return _preprocess_variants(img)[0]


class PlateOCR:
    def __init__(self):
        self._ocr_en = None   # English OCR (digits + TN/RS)
        self._ocr_ar = None   # Arabic OCR (detects تونس)
        self._init_paddle()

    def _init_paddle(self):
        try:
            from paddleocr import PaddleOCR
            # Try GPU first, fall back to CPU
            use_gpu = True
            try:
                self._ocr_en = PaddleOCR(use_angle_cls=True, lang="en",
                                         use_gpu=True, show_log=False)
                logger.info("PaddleOCR EN initialised (GPU)")
            except Exception:
                use_gpu = False
                self._ocr_en = PaddleOCR(use_angle_cls=True, lang="en",
                                         use_gpu=False, show_log=False)
                logger.info("PaddleOCR EN initialised (CPU)")

            # Arabic OCR for تونس detection
            try:
                self._ocr_ar = PaddleOCR(use_angle_cls=True, lang="arabic",
                                         use_gpu=use_gpu, show_log=False)
                logger.info("PaddleOCR AR initialised (Arabic/تونس support)")
            except Exception as e_ar:
                logger.warning(f"Arabic OCR unavailable ({e_ar}) – using EN-only mode")
                self._ocr_ar = None

        except Exception as e:
            logger.error(f"PaddleOCR unavailable: {e}")
            self._ocr_en = None
            self._ocr_ar = None

    def _ocr_pass(self, ocr_instance, img: np.ndarray) -> Tuple[str, float]:
        """Run a single OCR pass; return (raw_text, mean_confidence)."""
        try:
            result = ocr_instance.ocr(img, cls=True)
            if not result or not result[0]:
                return "", 0.0
            texts, confs = [], []
            for line in result[0]:
                if line and len(line) >= 2:
                    texts.append(line[1][0])
                    confs.append(float(line[1][1]))
            if not texts:
                return "", 0.0
            return " ".join(texts), sum(confs) / len(confs)
        except Exception as e:
            logger.debug(f"OCR pass error: {e}")
            return "", 0.0

    def read(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Returns (plate_text, confidence).
        Runs multiple preprocessing variants + dual-language OCR (EN + AR),
        picks the best validated result.
        """
        if img is None or img.size == 0:
            return None, 0.0

        if self._ocr_en is None:
            return self._tesseract_fallback(_preprocess(img))

        variants = _preprocess_variants(img)
        best_plate, best_conf = None, 0.0

        for variant in variants:
            # Pass 1: English OCR
            raw_en, conf_en = self._ocr_pass(self._ocr_en, variant)

            # Pass 2: Arabic OCR (to catch تونس)
            raw_ar, conf_ar = ("", 0.0)
            if self._ocr_ar is not None:
                raw_ar, conf_ar = self._ocr_pass(self._ocr_ar, variant)

            # Merge: if Arabic pass found تونس, inject it into English result
            combined_raw = self._merge_en_ar(raw_en, raw_ar)
            conf = max(conf_en, conf_ar) if combined_raw else conf_en

            norm = _normalize(combined_raw)
            norm = _apply_char_fixes(norm)
            validated = _validate(norm)

            candidate = validated or norm
            if candidate and conf > best_conf:
                best_plate, best_conf = candidate, conf

        if best_plate is None:
            return None, 0.0

        if best_conf < _CONF_THRESH:
            logger.debug(f"Low confidence OCR: {best_plate} ({best_conf:.2f})")
            return best_plate, best_conf   # still return, let caller decide

        return best_plate, best_conf

    @staticmethod
    def _merge_en_ar(raw_en: str, raw_ar: str) -> str:
        """
        Merge English and Arabic OCR results.
        If Arabic pass contains تونس (or variant), inject 'TN' into the
        English result replacing any garbled middle segment.
        """
        if not raw_ar:
            return raw_en

        has_tounes = any(v in raw_ar for v in _TOUNES_VARIANTS) or "تونس" in raw_ar
        if not has_tounes:
            return raw_en  # Arabic pass found nothing useful

        # Arabic detected تونس – normalise AR result and prefer it for the
        # middle token, but keep EN digits as they're more accurate
        norm_ar = _normalize(raw_ar)   # will convert تونس → TN
        norm_en = _normalize(raw_en) if raw_en else ""

        # If EN already has TN/RS, trust it
        if re.search(r'\bTN\b|\bRS\b', norm_en, re.IGNORECASE):
            return raw_en

        # Otherwise use AR-normalised text (has TN injected)
        return norm_ar if "TN" in norm_ar else raw_en

    @staticmethod
    def _tesseract_fallback(img: np.ndarray) -> Tuple[Optional[str], float]:
        """Try pytesseract; if unavailable return mock."""
        try:
            import pytesseract
            config = r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
            raw = pytesseract.image_to_string(img, config=config).strip()
            norm = _normalize(raw)
            norm = _apply_char_fixes(norm)
            return norm or None, 0.60
        except Exception:
            logger.warning("Using mock OCR – install PaddleOCR or pytesseract")
            return "MOCK_PLATE", 0.50
