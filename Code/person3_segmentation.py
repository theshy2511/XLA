"""
person3_segmentation.py
Module tách ký tự - NGƯỜI 3 (Phần A)
Nhiệm vụ: Character Segmentation
"""
import cv2
import numpy as np


class CharacterSegmenter:
    """
    Tách ký tự từ biển số

    Input: cropped_plate (grayscale)
    Output: (segmented_visual, characters_list)
    """

    def __init__(self):
        self.characters = []

    def segment(self, plate_image):
        """
        Tách ký tự từ biển số

        Args:
            plate_image: Ảnh biển số đã crop (grayscale)

        Returns:
            tuple: (segmented_visual, characters)
            - segmented_visual: Ảnh có vẽ bounding box mỗi ký tự
            - characters: List[dict] với keys: 'image', 'position', 'width'
        """
        # 1. Preprocessing
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()

        # 2. Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Remove border noise
        h, w = binary.shape
        binary[0:5, :] = 0
        binary[h - 5:h, :] = 0
        binary[:, 0:5] = 0
        binary[:, w - 5:w] = 0

        # 4. Vertical projection
        vertical_sum = np.sum(binary, axis=0)

        # 5. Tìm vị trí tách (thung lũng)
        threshold = np.mean(vertical_sum) * 0.2
        in_char = False
        char_positions = []
        start = 0

        for i, val in enumerate(vertical_sum):
            if val > threshold and not in_char:
                start = i
                in_char = True
            elif val <= threshold and in_char:
                if i - start > 5:  # Bỏ qua vùng quá nhỏ (noise)
                    char_positions.append((start, i))
                in_char = False

        # Handle ký tự cuối
        if in_char:
            char_positions.append((start, len(vertical_sum)))

        # 6. Crop từng ký tự
        self.characters = []
        visual_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for idx, (x_start, x_end) in enumerate(char_positions):
            # Vẽ hình chữ nhật trên ảnh visual
            cv2.rectangle(visual_image, (x_start, 0), (x_end, h), (0, 255, 0), 2)
            cv2.putText(visual_image, str(idx + 1), (x_start + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Crop ký tự
            char_img = binary[:, x_start:x_end]

            # Tìm bounding box chặt hơn
            coords = cv2.findNonZero(char_img)
            if coords is not None:
                x, y, w_char, h_char = cv2.boundingRect(coords)
                char_img = char_img[y:y + h_char, x:x + w_char]

            self.characters.append({
                'image': char_img.copy(),
                'position': (x_start, x_end),
                'width': x_end - x_start,
                'index': idx
            })

        return visual_image, self.characters


# Test module
if __name__ == '__main__':
    import sys
    from person1_preprocessing import ImagePreprocessor
    from person2_detection import LicensePlateDetector

    if len(sys.argv) < 2:
        print("Usage: python person3_segmentation.py <image_path>")
        sys.exit(1)

    # Pipeline test
    img = cv2.imread(sys.argv[1])
    import imutils

    img = imutils.resize(img, width=600)

    # Bước 1: Preprocessing
    preprocessor = ImagePreprocessor()
    gray, thresh, _ = preprocessor.process(img)

    # Bước 2: Detection
    detector = LicensePlateDetector()
    plate, info = detector.detect(gray, thresh)

    # Bước 3: Segmentation
    segmenter = CharacterSegmenter()
    visual, characters = segmenter.segment(plate)

    print(f"✓ Tách ký tự thành công!")
    print(f"  - Số ký tự: {len(characters)}")
    for i, char in enumerate(characters):
        print(f"  - Ký tự {i + 1}: Position={char['position']}, Width={char['width']}")

    cv2.imshow('Segmented Characters', visual)
    cv2.waitKey(0)
