"""
person1_preprocessing.py
Module tiền xử lý ảnh - NGƯỜI 1
Nhiệm vụ: Blackhat + Sobel + Threshold
"""
import cv2
import numpy as np
import imutils


class ImagePreprocessor:
    """
    Tiền xử lý ảnh đầu vào

    Input: Ảnh gốc (BGR)
    Output: (gray, threshold, processing_steps)
    """

    def __init__(self):
        self.gray = None
        self.blackhat = None
        self.sobel = None
        self.threshold = None

    def process(self, image):
        """
        Xử lý ảnh qua các bước tiền xử lý

        Args:
            image: Ảnh BGR từ cv2.imread() hoặc đã resize

        Returns:
            tuple: (gray, threshold, processing_steps)
            processing_steps = dict với các key: 'blackhat', 'sobel', 'threshold'
        """
        # Chuyển sang ảnh xám
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # BƯỚC 1: Blackhat transform
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        self.blackhat = cv2.morphologyEx(self.gray, cv2.MORPH_BLACKHAT, rectKernel)

        # BƯỚC 2: Sobel gradient X
        gradX = cv2.Sobel(self.blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        self.sobel = gradX.astype("uint8")

        # BƯỚC 3: Morphology + Threshold
        gradX = cv2.GaussianBlur(self.sobel, (5, 5), 0)
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, squareKernel)
        self.threshold = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Dilate + Erode
        self.threshold = cv2.dilate(self.threshold, None, iterations=2)
        self.threshold = cv2.erode(self.threshold, None, iterations=1)

        # Trả về kết quả
        processing_steps = {
            'blackhat': self.blackhat.copy(),
            'sobel': self.sobel.copy(),
            'threshold': self.threshold.copy()
        }

        return self.gray.copy(), self.threshold.copy(), processing_steps


# Test module
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python person1_preprocessing.py <image_path>")
        sys.exit(1)

    preprocessor = ImagePreprocessor()
    img = cv2.imread(sys.argv[1])
    img = imutils.resize(img, width=600)

    gray, thresh, steps = preprocessor.process(img)

    print("✓ Tiền xử lý hoàn thành!")
    print(f"  - Gray shape: {gray.shape}")
    print(f"  - Threshold shape: {thresh.shape}")

    cv2.imshow('Blackhat', steps['blackhat'])
    cv2.imshow('Sobel', steps['sobel'])
    cv2.imshow('Threshold', thresh)
    cv2.waitKey(0)
