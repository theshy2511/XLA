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
        self.best_candidate = None
        self.plate_info = None

    # ============================================================
    #           LỌC CONTOUR CHUẨN BIỂN SỐ
    # ============================================================
    def _score_contour(self, c, img_h, img_w, edges, thresh_img):
        """
        Chấm điểm contour dựa trên:
        - Kích thước / tỉ lệ
        - Vị trí tương đối theo chiều dọc
        - Độ đặc (solidity)
        - Mật độ biên (edge density)
        - Tỉ lệ pixel trắng trong ảnh threshold
        """

        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect = w / float(h)

        # Loại contour khổng lồ bao gần hết ảnh
        if w > img_w * 0.8 or h > img_h * 0.7:
            return -1

        # Loại contour quá nhỏ
        if w < 40 or h < 20 or area < 400:
            return -1

        # Biển xe máy 2 dòng thường cao hơn → aspect nhỏ 0.5 – 1.2
        moto_aspect = 0.45 < aspect < 1.4

        # Biển xe máy 1 dòng: 1.2 – 2.3
        moto_one_line = 1.1 < aspect < 2.5

        # Biển ô tô: 2.8 – 6.5
        car_aspect = 3.0 < aspect < 6.8

        good_aspect = moto_aspect or moto_one_line or car_aspect
        if not good_aspect:
            return -1

        # Vị trí theo chiều dọc: cho phép từ 10% đến 90% chiều cao ảnh
        # (biển hơi cao lên trên vẫn được)
        if y < img_h * 0.10 or y > img_h * 0.90:
            return -1

        # ---------- Thêm 1: Mật độ pixel trắng trong threshold ----------
        patch_thresh = thresh_img[y:y + h, x:x + w]
        white_ratio = cv2.countNonZero(patch_thresh) / float(w * h + 1e-6)

        # Biển số sau tiền xử lý thường có nền trắng + chữ đen:
        # tỉ lệ trắng ~ 0.25 -> 0.90
        if white_ratio < 0.20 or white_ratio > 0.95:
            return -1

        # ---------- Thêm 2: Mật độ biên trong ảnh Canny ----------
        patch_edges = edges[y:y + h, x:x + w]
        edge_ratio = cv2.countNonZero(patch_edges) / float(w * h + 1e-6)

        # Nếu gần như không có biên → thường là mảng nền trơn
        if edge_ratio < 0.015:
            return -1

        solidity = area / (w * h + 1e-6)

        # Score tổng hợp: ưu tiên solidity + aspect + edge density
        score = (
            solidity * 0.5 +
            min(aspect, 6.0) * 0.3 +
            edge_ratio * 0.2
        )

        return score

    # ============================================================
    #       LỌC BIÊN + MORPHOLOGY ĐỂ TÁCH BIỂN
    # ============================================================
    def _morphology_plate(self, edges):
        """
        Nhận ảnh Canny edges, thực hiện morphology để nối các nét
        ký tự lại thành khối biển số.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mor = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mor

    # ============================================================
    #                   CROP XOAY CHUẨN
    # ============================================================
    def _crop_rotated_plate(self, gray, contour):

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.float32(box)

        W, H = rect[1]

        # Nếu bị đảo chiều thì đảo lại
        if W < H:
            W, H = H, W

        dst = np.array([
            [0, H - 1],
            [0, 0],
            [W - 1, 0],
            [W - 1, H - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(box, dst)
        warped = cv2.warpPerspective(gray, M, (int(W), int(H)))

        # Cắt bớt viền đen nếu kích thước đủ lớn
        if warped.shape[0] > 12 and warped.shape[1] > 12:
            warped = warped[5:-5, 5:-5]

        return warped

    # ============================================================
    #                   HÀM PHÁT HIỆN CHÍNH
    # ============================================================
    def detect(self, gray, threshold):

        H, W = gray.shape[:2]

        # Canny cho toàn ảnh (dùng cả cho morphology và chấm điểm)
        edges = cv2.Canny(gray, 70, 180)

        # Morphology để nối ký tự
        enhanced = self._morphology_plate(edges)

        # Kết hợp với ảnh threshold của Person1
        combined = cv2.bitwise_or(threshold, enhanced)

        cnts = cv2.findContours(combined.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        best_score = -1
        best_cnt = None

        for c in cnts:
            s = self._score_contour(c, H, W, edges, threshold)
            if s > best_score:
                best_score = s
                best_cnt = c

        if best_cnt is None or best_score < 0.15:
            raise ValueError("Không tìm được contour biển số hợp lệ!")

        self.best_candidate = best_cnt

        x, y, w, h = cv2.boundingRect(best_cnt)
        area = cv2.contourArea(best_cnt)
        aspect = w / float(h)
        solidity = area / (w * h + 1e-6)

        self.plate_info = {
            "x": x, "y": y, "w": w, "h": h,
            "aspectRatio": aspect,
            "solidity": solidity,
            "score": best_score,
            "type": "unknown"
        }

        cropped = self._crop_rotated_plate(gray, best_cnt)
        return cropped, self.plate_info


# ===============================
#           TEST
# ===============================
if __name__ == '__main__':
    import sys
    from person1_preprocessing import ImagePreprocessor

    if len(sys.argv) < 2:
        print("Usage: python person2_detection.py <image_path>")
        exit()

    img = cv2.imread(sys.argv[1])
    img = imutils.resize(img, width=600)

    pre = ImagePreprocessor()
    gray, thresh, _ = pre.process(img)

    det = LicensePlateDetector()
    plate, info = det.detect(gray, thresh)

    print(info)
    cv2.imshow("Plate", plate)
    cv2.waitKey(0)
