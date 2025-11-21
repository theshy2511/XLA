"""
person3_recognition.py
Module nhận dạng ký tự - NGƯỜI 3 (Phần B)
Nhiệm vụ: OCR
"""
import cv2
import numpy as np
import pytesseract
from config import OCR_WHITELIST, OCR_CONFIG


class LicensePlateRecognizer:
    """
    Nhận dạng ký tự từ ảnh biển số

    Input: cropped_plate hoặc list characters
    Output: recognized_text
    """

    def __init__(self):
        self.ocr_image = None
        self.raw_text = None

    def recognize(self, plate_image):
        """
        Nhận dạng text từ biển số (toàn bộ)

        Args:
            plate_image: Ảnh biển số đã crop (grayscale)

        Returns:
            str: Text nhận dạng được
        """
        # Otsu threshold cho OCR
        (T, self.ocr_image) = cv2.threshold(plate_image, 0, 255,
                                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Đảo ngược nếu cần (chữ đen nền trắng)
        if np.mean(self.ocr_image) < 127:
            self.ocr_image = cv2.bitwise_not(self.ocr_image)

        # OCR với Tesseract
        config = OCR_CONFIG + OCR_WHITELIST
        self.raw_text = pytesseract.image_to_string(self.ocr_image, config=config)

        # Làm sạch
        cleaned = "".join(char for char in self.raw_text if char in OCR_WHITELIST)
        cleaned = cleaned.replace('\n', ' ').replace('\f', '').strip()

        if len(cleaned) < 4:
            return "N/A"

        return cleaned

    def recognize_characters(self, characters_list):
        """
        Nhận dạng từng ký tự riêng lẻ (nâng cao)

        Args:
            characters_list: List dict từ CharacterSegmenter

        Returns:
            str: Text ghép từ các ký tự
        """
        results = []

        for char_data in characters_list:
            char_img = char_data['image']

            # Resize để OCR tốt hơn
            h, w = char_img.shape
            if h < 20 or w < 10:
                scale = max(20 / h, 10 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                char_img = cv2.resize(char_img, (new_w, new_h))

            # OCR single character
            config = '--oem 3 --psm 10 -c tessedit_char_whitelist=' + OCR_WHITELIST
            text = pytesseract.image_to_string(char_img, config=config)

            # Clean
            cleaned = "".join(c for c in text if c in OCR_WHITELIST).strip()

            if cleaned:
                results.append(cleaned)

        return ''.join(results) if results else "N/A"

    def get_ocr_image(self):
        """Lấy ảnh đã xử lý cho OCR (để debug)"""
        return self.ocr_image


# Test module
if __name__ == '__main__':
    import sys
    from person1_preprocessing import ImagePreprocessor
    from person2_detection import LicensePlateDetector
    from person3_segmentation import CharacterSegmenter

    if len(sys.argv) < 2:
        print("Usage: python person3_recognition.py <image_path>")
        sys.exit(1)

    # Full pipeline test
    img = cv2.imread(sys.argv[1])
    import imutils

    img = imutils.resize(img, width=600)

    # Bước 1-2-3
    preprocessor = ImagePreprocessor()
    gray, thresh, _ = preprocessor.process(img)

    detector = LicensePlateDetector()
    plate, info = detector.detect(gray, thresh)

    segmenter = CharacterSegmenter()
    visual, characters = segmenter.segment(plate)

    # Bước 4: OCR
    recognizer = LicensePlateRecognizer()

    # Method 1: OCR toàn bộ
    text1 = recognizer.recognize(plate)
    print(f"✓ OCR toàn bộ: {text1}")

    # Method 2: OCR từng ký tự
    text2 = recognizer.recognize_characters(characters)
    print(f"✓ OCR từng ký tự: {text2}")

    cv2.imshow('OCR Image', recognizer.get_ocr_image())
    cv2.waitKey(0)
