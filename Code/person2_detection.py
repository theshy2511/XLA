"""
person2_detection.py
Module phát hiện biển số - NGƯỜI 2
Nhiệm vụ: Tìm contour + Masking + Crop
Cập nhật: Tách riêng logic Biển Dài (Xe hơi) và Biển Vuông (Xe máy/Xe hơi)
"""
import cv2
import numpy as np
import imutils
from config import PLATE_DETECTION


class LicensePlateDetector:
    """
    Phát hiện và crop biển số từ ảnh
    Input: gray_image, threshold_image (từ Person 1)
    Output: cropped_plate, plate_info
    """

    def __init__(self):
        self.best_contour = None
        self.plate_info = None

    def detect(self, gray_image, threshold_image):
        """
        Phát hiện biển số với chiến thuật phân loại hình dáng (Dài vs Vuông)
        """
        
        # -----------------------------------------------------------
        # BƯỚC 1: XỬ LÝ MORPHOLOGY (Tạo khối)
        # -----------------------------------------------------------
        # Kernel chữ nhật dài để nối các ký tự số (quan trọng cho biển dài)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        morphed = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, rect_kernel)

        # -----------------------------------------------------------
        # BƯỚC 2: TÌM CONTOUR
        # -----------------------------------------------------------
        contours = cv2.findContours(morphed.copy(), 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # Lấy top 15 contour có diện tích lớn nhất
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        # -----------------------------------------------------------
        # BƯỚC 3: CHIẾN THUẬT TÌM KIẾM (2 VÒNG)
        # -----------------------------------------------------------
        
        # VÒNG 1: CHẾ ĐỘ CHẶT CHẼ (STRICT)
        # Ưu tiên tìm những biển số có hình dáng chuẩn, rõ nét
        candidate, info = self._find_best_candidate(contours, strict_mode=True)

        # VÒNG 2: CHẾ ĐỘ NỚI LỎNG (FALLBACK)
        # Nếu vòng 1 thất bại, mở rộng phạm vi tìm kiếm (chấp nhận méo, nghiêng)
        if candidate is None:
            # print("Strict mode failed. Switching to Loose mode...")
            candidate, info = self._find_best_candidate(contours, strict_mode=False)

        # Nếu vẫn không tìm thấy
        if candidate is None:
            raise ValueError('Không tìm thấy biển số (No contour match)!')

        self.best_contour = candidate
        self.plate_info = info

        # -----------------------------------------------------------
        # BƯỚC 4: CẮT ẢNH & PADDING
        # -----------------------------------------------------------
        return self._crop_plate(gray_image, candidate)

    def _find_best_candidate(self, contours, strict_mode=True):
        """
        Hàm lọc contour theo 2 nhóm riêng biệt: BIỂN DÀI và BIỂN VUÔNG
        """
        best_candidate = None
        max_area = 0
        best_info = {}

        # CẤU HÌNH THAM SỐ (Tách biệt 2 loại biển)
        if strict_mode:
            # --- Cấu hình CHẶT (Ảnh đẹp) ---
            
            # 1. Biển Dài (Chắc chắn là Xe Hơi)
            # Tỉ lệ thường từ 3.0 đến 5.0
            long_cfg = {
                'min_ratio': 2.5, 'max_ratio': 6.0,
                'min_w': 80, 'min_h': 20, 'min_solidity': 0.4
            }
            
            # 2. Biển Vuông (Xe Máy HOẶC Xe Hơi biển vuông)
            # Tỉ lệ thường từ 1.2 đến 1.5
            square_cfg = {
                'min_ratio': 0.6, 'max_ratio': 2.0,
                'min_w': 40, 'min_h': 40, 'min_solidity': 0.4
            }
        else:
            # --- Cấu hình LỎNG (Ảnh nghiêng/mờ) ---
            
            # Nới lỏng tỉ lệ và độ đặc (Solidity)
            long_cfg = {
                'min_ratio': 2.2, 'max_ratio': 8.0, # Chấp nhận rất dài hoặc nghiêng
                'min_w': 60, 'min_h': 15, 'min_solidity': 0.15 # Chấp nhận nhiễu
            }
            
            square_cfg = {
                'min_ratio': 0.5, 'max_ratio': 2.4, # Chấp nhận hơi dẹt hoặc vuông méo
                'min_w': 30, 'min_h': 30, 'min_solidity': 0.15
            }

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            aspectRatio = w / float(h)
            area = cv2.contourArea(c)
            
            # Tính độ đặc
            hull = cv2.convexHull(c)
            hullArea = cv2.contourArea(hull)
            solidity = area / float(hullArea) if hullArea > 0 else 0

            # --- KIỂM TRA LOGIC 1: BIỂN DÀI (Xe Hơi) ---
            is_long = (long_cfg['min_ratio'] <= aspectRatio <= long_cfg['max_ratio'] and
                       w > long_cfg['min_w'] and 
                       h > long_cfg['min_h'] and
                       solidity > long_cfg['min_solidity'])

            # --- KIỂM TRA LOGIC 2: BIỂN VUÔNG (Xe Máy / Xe Hơi) ---
            is_square = (square_cfg['min_ratio'] <= aspectRatio <= square_cfg['max_ratio'] and
                         w > square_cfg['min_w'] and 
                         h > square_cfg['min_h'] and
                         solidity > square_cfg['min_solidity'])

            if is_long or is_square:
                # Nếu tìm thấy ứng viên tiềm năng, ưu tiên cái to nhất (Diện tích lớn nhất)
                if area > max_area:
                    max_area = area
                    best_candidate = c
                    
                    # Xác định loại xe dựa trên hình dáng biển
                    detected_type = 'Xe Hơi (Biển Dài)' if is_long else 'Xe Máy/Hơi (Vuông)'
                    
                    best_info = {
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'aspectRatio': aspectRatio,
                        'solidity': solidity,
                        'type': detected_type,
                        'area': area
                    }
        
        return best_candidate, best_info

    def _crop_plate(self, image, contour):
        """Cắt vùng biển số và thêm padding"""
        (x, y, w, h) = cv2.boundingRect(contour)
        h_img, w_img = image.shape[:2]
        
        # Padding an toàn (3-5 pixel)
        padding = 4
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(w_img, x + w + padding)
        y_end = min(h_img, y + h + padding)
        
        # Update info
        if self.plate_info:
            self.plate_info['x'] = x_start
            self.plate_info['y'] = y_start
            self.plate_info['w'] = x_end - x_start
            self.plate_info['h'] = y_end - y_start

        cropped_plate = image[y_start:y_end, x_start:x_end]
        return cropped_plate, self.plate_info


# Unit Test độc lập
if __name__ == '__main__':
    import sys
    from person1_preprocessing import ImagePreprocessor

    if len(sys.argv) < 2:
        print("Usage: python person2_detection.py <image_path>")
        sys.exit(1)

    try:
        preprocessor = ImagePreprocessor()
        img = cv2.imread(sys.argv[1])
        if img is None: 
            raise Exception("Cannot read image")
            
        img = imutils.resize(img, width=600)
        gray, thresh, _ = preprocessor.process(img)

        detector = LicensePlateDetector()
        plate, info = detector.detect(gray, thresh)

        print("\n" + "="*40)
        print("✓ KẾT QUẢ PHÁT HIỆN")
        print("="*40)
        print(f"  - Loại:       {info['type']}")
        print(f"  - Tỉ lệ W/H:  {info['aspectRatio']:.2f}")
        print(f"  - Độ đặc:     {info['solidity']:.2f}")
        print("="*40)

        cv2.imshow('Threshold', thresh)
        cv2.imshow('Detected Plate', plate)
        cv2.waitKey(0)

    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")