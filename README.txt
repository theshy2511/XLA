===============================================
HỆ THỐNG NHẬN DẠNG BIỂN SỐ XE
===============================================

1. CÀI ĐẶT:
   pip install opencv-python numpy pillow pytesseract imutils
và cài Tesseract OCR

2. CHẠY CHƯƠNG TRÌNH:
   python code/main.py

3. TEST TỪNG PHẦN:
   python code/person1_preprocessing.py test/oto_1.jpg
   python code/person2_detection.py test/oto_1.jpg
   python code/person3_segmentation.py test/oto_1.jpg

4. CẤU TRÚC:
   - code/: Chứa tất cả file code
   - test/: Chứa ảnh để test

