"""
image_processor.py
Module xử lý ảnh và nhận dạng biển số
"""

import cv2
import numpy as np
import imutils
import pytesseract
from config import *


# Cấu hình Tesseract
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


class LicensePlateProcessor:
    """Class xử lý nhận dạng biển số"""

    def __init__(self, image_path):
        self.image_path = image_path
        self.img = None
        self.gray = None

        # Các bước xử lý
        self.blackhat = None
        self.sobel = None
        self.thresh = None

        # Kết quả
        self.plate_contour = None
        self.plate_info = None
        self.plate_image = None
        self.ocr_image = None
        self.recognized_text = None

    def load_image(self):
        """Đọc và chuẩn hóa ảnh"""
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            raise ValueError('Không thể đọc ảnh!')

        self.img = imutils.resize(self.img, width=IMAGE_WIDTH)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return self.img

    def apply_blackhat(self):
        """Bước 1: Black-hat Transform"""
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        self.blackhat = cv2.morphologyEx(self.gray, cv2.MORPH_BLACKHAT, rectKernel)
        return self.blackhat

    def apply_sobel(self):
        """Bước 2: Sobel Gradient X"""
        gradX = cv2.Sobel(self.blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        self.sobel = gradX.astype("uint8")
        return self.sobel

    def apply_threshold(self):
        """Bước 3: Morphology và Threshold"""
        gradX = cv2.GaussianBlur(self.sobel, (5, 5), 0)
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, squareKernel)
        self.thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        self.thresh = cv2.dilate(self.thresh, None, iterations=2)
        self.thresh = cv2.erode(self.thresh, None, iterations=1)
        return self.thresh

    def find_plate_contour(self):
        """Bước 4: Tìm contour biển số"""
        contours = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        best_candidate = None
        best_solidity = 0
        best_info = {}

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            aspectRatio = w / float(h)
            solidity = cv2.contourArea(c) / float(w * h)

            # Kiểm tra điều kiện xe máy
            is_moto = (PLATE_DETECTION['moto']['aspect_ratio_min'] < aspectRatio <
                       PLATE_DETECTION['moto']['aspect_ratio_max'] and
                       w > PLATE_DETECTION['moto']['width_min'] and
                       h > PLATE_DETECTION['moto']['height_min'] and
                       solidity > PLATE_DETECTION['moto']['solidity_min'])

            # Kiểm tra điều kiện xe hơi
            is_car = (PLATE_DETECTION['car']['aspect_ratio_min'] < aspectRatio <
                      PLATE_DETECTION['car']['aspect_ratio_max'] and
                      w > PLATE_DETECTION['car']['width_min'] and
                      h > PLATE_DETECTION['car']['height_min'] and
                      solidity > PLATE_DETECTION['car']['solidity_min'])

            if (is_moto or is_car) and solidity > best_solidity:
                best_solidity = solidity
                best_candidate = c
                best_info = {
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'aspectRatio': aspectRatio,
                    'solidity': solidity,
                    'type': 'Xe máy' if is_moto else 'Xe hơi',
                    'area': cv2.contourArea(c)
                }

        if best_candidate is None:
            raise ValueError('Không phát hiện được biển số!')

        self.plate_contour = best_candidate
        self.plate_info = best_info
        return best_candidate, best_info

    def extract_plate(self):
        """Bước 5: Cô lập biển số bằng Masking"""
        (x, y, w, h) = cv2.boundingRect(self.plate_contour)

        # Masking
        mask = np.zeros(self.gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [self.plate_contour], -1, 255, -1)
        isolated_plate = cv2.bitwise_and(self.gray, self.gray, mask=mask)
        self.plate_image = isolated_plate[y:y + h, x:x + w]

        # Otsu Threshold cho OCR
        (T, ocr_thresh) = cv2.threshold(self.plate_image, 0, 255,
                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Đảo ngược nếu cần
        if np.mean(ocr_thresh[mask[y:y + h, x:x + w] > 0]) < 127:
            ocr_thresh = cv2.bitwise_not(ocr_thresh)

        self.ocr_image = ocr_thresh
        return self.plate_image, ocr_thresh

    def recognize_text(self):
        """Bước 6: Tesseract OCR"""
        config = OCR_CONFIG + OCR_WHITELIST
        text = pytesseract.image_to_string(self.ocr_image, config=config)

        # Dọn dẹp
        cleaned = "".join(char for char in text if char in OCR_WHITELIST)
        cleaned = cleaned.replace('\n', ' ').replace('\f', '').strip()

        if len(cleaned) < 4:
            cleaned = "N/A"

        self.recognized_text = cleaned
        return cleaned

    def process_full(self):
        """Xử lý toàn bộ pipeline"""
        self.load_image()
        self.apply_blackhat()
        self.apply_sobel()
        self.apply_threshold()
        self.find_plate_contour()
        self.extract_plate()
        self.recognize_text()
        return self.recognized_text
