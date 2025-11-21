"""
person2_detection.py
Module phát hiện biển số - NGƯỜI 2
Nhiệm vụ: Tìm contour + Masking + Crop
"""
import cv2
import numpy as np
import imutils
from config import PLATE_DETECTION


class LicensePlateDetector:
    """
    Phát hiện và crop biển số từ ảnh

    Input: gray_image, threshold_image
    Output: cropped_plate, plate_info
    """

    def __init__(self):
        self.best_contour = None
        self.plate_info = None

    def detect(self, gray_image, threshold_image):
        """
        Phát hiện biển số

        Args:
            gray_image: Ảnh xám gốc
            threshold_image: Ảnh threshold từ preprocessing

        Returns:
            tuple: (cropped_plate, plate_info)

        Raises:
            ValueError: Nếu không tìm thấy biển số
        """
        # Tìm contours
        contours = cv2.findContours(threshold_image.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # Tìm contour tốt nhất
        best_solidity = 0

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            aspectRatio = w / float(h)
            solidity = cv2.contourArea(c) / float(w * h)

            # Điều kiện xe máy
            is_moto = (PLATE_DETECTION['moto']['aspect_ratio_min'] < aspectRatio <
                       PLATE_DETECTION['moto']['aspect_ratio_max'] and
                       w > PLATE_DETECTION['moto']['width_min'] and
                       h > PLATE_DETECTION['moto']['height_min'] and
                       solidity > PLATE_DETECTION['moto']['solidity_min'])

            # Điều kiện xe hơi
            is_car = (PLATE_DETECTION['car']['aspect_ratio_min'] < aspectRatio <
                      PLATE_DETECTION['car']['aspect_ratio_max'] and
                      w > PLATE_DETECTION['car']['width_min'] and
                      h > PLATE_DETECTION['car']['height_min'] and
                      solidity > PLATE_DETECTION['car']['solidity_min'])

            if (is_moto or is_car) and solidity > best_solidity:
                best_solidity = solidity
                self.best_contour = c
                self.plate_info = {
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'aspectRatio': aspectRatio,
                    'solidity': solidity,
                    'type': 'Xe máy' if is_moto else 'Xe hơi',
                    'area': cv2.contourArea(c)
                }

        if self.best_contour is None:
            raise ValueError('Không tìm thấy biển số!')

        # Masking để cô lập
        (x, y, w, h) = cv2.boundingRect(self.best_contour)
        mask = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [self.best_contour], -1, 255, -1)
        isolated_plate = cv2.bitwise_and(gray_image, gray_image, mask=mask)

        # Crop biển số
        cropped_plate = isolated_plate[y:y + h, x:x + w]

        return cropped_plate, self.plate_info


# Test module
if __name__ == '__main__':
    import sys
    from person1_preprocessing import ImagePreprocessor

    if len(sys.argv) < 2:
        print("Usage: python person2_detection.py <image_path>")
        sys.exit(1)

    # Pipeline test
    preprocessor = ImagePreprocessor()
    img = cv2.imread(sys.argv[1])
    img = imutils.resize(img, width=600)
    gray, thresh, _ = preprocessor.process(img)

    detector = LicensePlateDetector()
    plate, info = detector.detect(gray, thresh)

    print("✓ Phát hiện biển số thành công!")
    print(f"  - Loại: {info['type']}")
    print(f"  - Vị trí: ({info['x']}, {info['y']})")
    print(f"  - Kích thước: {info['w']}x{info['h']}")
    print(f"  - Tỉ lệ: {info['aspectRatio']:.2f}")
    print(f"  - Solidity: {info['solidity']:.3f}")

    cv2.imshow('Detected Plate', plate)
    cv2.waitKey(0)
