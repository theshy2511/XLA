"""
config.py
Cấu hình hệ thống
"""

# Tesseract path (Windows)
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Màu sắc giao diện
COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_medium': '#16213e',
    'bg_light': '#0f3460',
    'primary': '#00d4ff',
    'success': '#2ecc71',
    'danger': '#e74c3c',
    'warning': '#f39c12',
    'text_light': '#a8dadc',
    'text_dark': '#e0e0e0'
}

# Kích thước ảnh
IMAGE_WIDTH = 600
THUMBNAIL_SIZE = (550, 300)

# Thông số phát hiện biển số
PLATE_DETECTION = {
    'moto': {
        'aspect_ratio_min': 0.7,
        'aspect_ratio_max': 1.8,
        'width_min': 40,
        'height_min': 40,
        'solidity_min': 0.3
    },
    'car': {
        'aspect_ratio_min': 2.5,
        'aspect_ratio_max': 6.0,
        'width_min': 80,
        'height_min': 20,
        'solidity_min': 0.3
    }
}

# OCR Config
OCR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."
OCR_CONFIG = '--oem 3 --psm 6 -c tessedit_char_whitelist='
