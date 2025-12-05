"""
main.py
File chính - GUI sử dụng pipeline modular
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import *
from PIL import ImageTk, Image
from datetime import datetime
import cv2

from config import COLORS
from pipeline import LicensePlatePipeline
from gui_components import create_labeled_frame

class LicensePlateApp:
    """Ứng dụng nhận dạng biển số - Sử dụng pipeline modular"""

    def __init__(self, root):
        self.root = root
        self.root.title('Hệ Thống Nhận Dạng Biển Số Xe')
        self.root.geometry('1600x850')
        self.root.configure(background=COLORS['bg_dark'])

        self.file_path = None
        self.pipeline = LicensePlatePipeline()
        self.result_data = None
        self.display_mode = 'result'

        # Tạo giao diện
        self._create_header()
        self._create_main_controls()
        self._create_tab_bar()
        self._create_display_area()
        self._create_status_bar()

    def _create_header(self):
        """Header"""
        header = Frame(self.root, bg=COLORS['bg_medium'], height=65)
        header.pack(fill=X, side=TOP)
        header.pack_propagate(False)

        Label(header, text='🚗 HỆ THỐNG NHẬN DẠNG BIỂN SỐ XE',
              bg=COLORS['bg_medium'], fg=COLORS['primary'],
              font=('Arial', 22, 'bold')).pack(pady=8)


    def _create_main_controls(self):
        """3 nút chính"""
        control = Frame(self.root, bg=COLORS['bg_medium'], height=85)
        control.pack(fill=X, side=TOP)
        control.pack_propagate(False)

        btn_frame = Frame(control, bg=COLORS['bg_medium'])
        btn_frame.pack(pady=15)

        self.upload_btn = Button(
            btn_frame, text='📁 THÊM ẢNH',
            command=self.upload_image,
            bg='#3498db', fg='white', font=('Arial', 13, 'bold'),
            padx=30, pady=12, cursor='hand2', relief=RAISED, bd=3,
            width=18
        )
        self.upload_btn.grid(row=0, column=0, padx=10)

        self.recognize_btn = Button(
            btn_frame, text='🔍 NHẬN DẠNG',
            command=self.recognize_plate,
            bg='#27ae60', fg='white', font=('Arial', 13, 'bold'),
            padx=30, pady=12, cursor='hand2', state=DISABLED, relief=RAISED, bd=3,
            width=18
        )
        self.recognize_btn.grid(row=0, column=1, padx=10)

        self.reset_btn = Button(
            btn_frame, text='🔄 LÀM MỚI',
            command=self.reset_app,
            bg='#e74c3c', fg='white', font=('Arial', 13, 'bold'),
            padx=30, pady=12, cursor='hand2', relief=RAISED, bd=3,
            width=18
        )
        self.reset_btn.grid(row=0, column=2, padx=10)

    def _create_tab_bar(self):
        """Tab bar"""
        tab_container = Frame(self.root, bg='#2c3e50', height=70)
        tab_container.pack(fill=X, side=TOP)
        tab_container.pack_propagate(False)

        tabs = Frame(tab_container, bg='#2c3e50')
        tabs.pack(expand=True)

        # Tab 1: KẾT QUẢ
        self.result_tab = Frame(tabs, bg='#9b59b6', cursor='hand2',
                                relief=RAISED, bd=0,
                                highlightthickness=3, highlightbackground='#8e44ad')
        self.result_tab.pack(side=LEFT, padx=3)
        self.result_tab.bind('<Button-1>', lambda e: self.switch_mode('result'))

        result_content = Frame(self.result_tab, bg='#9b59b6')
        result_content.pack(padx=40, pady=15)
        result_content.bind('<Button-1>', lambda e: self.switch_mode('result'))

        Label(result_content, text='📊', bg='#9b59b6',
              font=('Arial', 22)).pack(side=LEFT, padx=5)
        Label(result_content, text='KẾT QUẢ NHẬN DẠNG', bg='#9b59b6', fg='white',
              font=('Arial', 15, 'bold')).pack(side=LEFT, padx=5)

        for widget in result_content.winfo_children():
            widget.bind('<Button-1>', lambda e: self.switch_mode('result'))

        # Tab 2: XỬ LÝ
        self.process_tab = Frame(tabs, bg='#7f8c8d', cursor='hand2',
                                relief=FLAT, bd=0,
                                highlightthickness=2, highlightbackground='#95a5a6')
        self.process_tab.pack(side=LEFT, padx=3)
        self.process_tab.bind('<Button-1>', lambda e: self.switch_mode('process'))

        process_content = Frame(self.process_tab, bg='#7f8c8d')
        process_content.pack(padx=40, pady=15)
        process_content.bind('<Button-1>', lambda e: self.switch_mode('process'))

        Label(process_content, text='⚙️', bg='#7f8c8d',
              font=('Arial', 22)).pack(side=LEFT, padx=5)
        Label(process_content, text='CÁC BƯỚC XỬ LÝ', bg='#7f8c8d', fg='white',
              font=('Arial', 15, 'bold')).pack(side=LEFT, padx=5)

        for widget in process_content.winfo_children():
            widget.bind('<Button-1>', lambda e: self.switch_mode('process'))

    def _create_display_area(self):
        """Vùng hiển thị"""
        self.display_container = Frame(self.root, bg=COLORS['bg_dark'])
        self.display_container.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.display_container.grid_rowconfigure(0, weight=1)
        for i in range(5):
            self.display_container.grid_columnconfigure(i, weight=1)

        self._create_result_frames()
        self._create_process_frames()
        self.switch_mode('result')

    def _create_result_frames(self):
        """3 frames kết quả"""
        self.result_frames = []

        # Ảnh gốc
        frame1 = create_labeled_frame(self.display_container, '📷 1. ẢNH GỐC')
        img_container1 = Frame(frame1, bg=COLORS['bg_light'], width=470, height=540)
        img_container1.pack(expand=True, fill=BOTH, padx=10, pady=10)
        img_container1.pack_propagate(False)

        self.original_img = Label(img_container1, bg=COLORS['bg_light'])
        self.original_img.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.result_frames.append(frame1)

        # Biển số
        frame2 = create_labeled_frame(self.display_container, '🎯 2. BIỂN SỐ PHÁT HIỆN')
        img_container2 = Frame(frame2, bg=COLORS['bg_light'], width=470, height=540)
        img_container2.pack(expand=True, fill=BOTH, padx=10, pady=10)
        img_container2.pack_propagate(False)

        self.plate_img = Label(img_container2, bg=COLORS['bg_light'])
        self.plate_img.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.result_frames.append(frame2)

        # ✅ THAY ĐỔI: BỎ char_count_label
        # Kết quả
        frame3 = create_labeled_frame(self.display_container, '✅ 3. KẾT QUẢ NHẬN DẠNG', COLORS['success'])
        result_container = Frame(frame3, bg=COLORS['bg_light'], width=470, height=540)
        result_container.pack(expand=True, fill=BOTH, padx=10, pady=10)
        result_container.pack_propagate(False)

        content = Frame(result_container, bg=COLORS['bg_light'])
        content.place(relx=0.5, rely=0.5, anchor=CENTER)

        Label(content, text='Biển số xe:', bg=COLORS['bg_light'],
              fg=COLORS['text_light'], font=('Arial', 14, 'bold')).pack(pady=15)

        self.result_text = Label(content, text='Chưa nhận dạng',
                                bg=COLORS['bg_light'], fg='#95a5a6',
                                font=('Arial', 44, 'bold'), wraplength=440)
        self.result_text.pack(pady=35)

        self.timestamp = Label(content, text='', bg=COLORS['bg_light'],
                              fg='#95a5a6', font=('Arial', 11))
        self.timestamp.pack(pady=15)

        # ✅ ĐÃ XÓA: self.char_count_label

        self.result_frames.append(frame3)

    def _create_process_frames(self):
        """5 frames xử lý"""
        self.process_frames = []
        self.processing_imgs = []

        steps = [
            ('BLACKHAT', '#ff6b6b', 'Tiền xử lý'),
            ('SOBEL', '#ffd93d', 'Gradient X'),
            ('THRESHOLD', '#6bcf7f', 'Otsu + Morphology'),
            ('BẮT BIỂN SỐ', '#00d2ff', 'Character Boxes'),
            ('OCR INPUT', '#a29bfe', 'Binary Image')
        ]

        for title, color, desc in steps:
            frame = create_labeled_frame(self.display_container, title, color)

            Label(frame, text=desc, bg=COLORS['bg_light'],
                  fg='#95a5a6', font=('Arial', 9, 'italic')).pack(pady=3)

            img_container = Frame(frame, bg=COLORS['bg_light'], width=290, height=510)
            img_container.pack(expand=True, fill=BOTH, padx=8, pady=8)
            img_container.pack_propagate(False)

            img_label = Label(img_container, bg=COLORS['bg_light'])
            img_label.place(relx=0.5, rely=0.5, anchor=CENTER)

            self.process_frames.append(frame)
            self.processing_imgs.append(img_label)

    def switch_mode(self, mode):
        """Chuyển tab"""
        self.display_mode = mode

        for frame in self.result_frames:
            frame.grid_forget()
        for frame in self.process_frames:
            frame.grid_forget()

        if mode == 'result':
            for col, frame in enumerate(self.result_frames):
                frame.grid(row=0, column=col, sticky='nsew', padx=5, pady=5)

            self.result_tab.config(bg='#9b59b6', relief=RAISED,
                                   highlightthickness=3, highlightbackground='#8e44ad')
            for widget in self.result_tab.winfo_children():
                if hasattr(widget, 'config'):
                    widget.config(bg='#9b59b6')
                for child in widget.winfo_children():
                    if hasattr(child, 'config'):
                        child.config(bg='#9b59b6')

            self.process_tab.config(bg='#7f8c8d', relief=FLAT,
                                    highlightthickness=2, highlightbackground='#95a5a6')
            for widget in self.process_tab.winfo_children():
                if hasattr(widget, 'config'):
                    widget.config(bg='#7f8c8d')
                for child in widget.winfo_children():
                    if hasattr(child, 'config'):
                        child.config(bg='#7f8c8d')

        elif mode == 'process':
            for col, frame in enumerate(self.process_frames):
                frame.grid(row=0, column=col, sticky='nsew', padx=4, pady=5)

            self.process_tab.config(bg='#e67e22', relief=RAISED,
                                    highlightthickness=3, highlightbackground='#d35400')
            for widget in self.process_tab.winfo_children():
                if hasattr(widget, 'config'):
                    widget.config(bg='#e67e22')
                for child in widget.winfo_children():
                    if hasattr(child, 'config'):
                        child.config(bg='#e67e22')

            self.result_tab.config(bg='#7f8c8d', relief=FLAT,
                                   highlightthickness=2, highlightbackground='#95a5a6')
            for widget in self.result_tab.winfo_children():
                if hasattr(widget, 'config'):
                    widget.config(bg='#7f8c8d')
                for child in widget.winfo_children():
                    if hasattr(child, 'config'):
                        child.config(bg='#7f8c8d')

    def _create_status_bar(self):
        """Status bar"""
        status = Frame(self.root, bg=COLORS['bg_medium'], height=45)
        status.pack(fill=X, side=BOTTOM)
        status.pack_propagate(False)

        self.status_label = Label(status, text='Trạng thái: Sẵn sàng',
                                 bg=COLORS['bg_medium'], fg=COLORS['text_light'],
                                 font=('Arial', 11))
        self.status_label.pack(side=LEFT, padx=20, pady=12)

        self.progress = ttk.Progressbar(status, mode='indeterminate', length=300)
        self.progress.pack(side=RIGHT, padx=20)

    def upload_image(self):
        """Upload ảnh"""
        try:
            self.file_path = filedialog.askopenfilename(
                title='Chọn ảnh biển số xe',
                filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp')]
            )

            if self.file_path:
                img = Image.open(self.file_path)
                photo = self._resize_keep_ratio(img, 460, 530)

                self.original_img.configure(image=photo)
                self.original_img.image = photo

                self.recognize_btn.config(state=NORMAL)

                filename = self.file_path.split("/")[-1].split("\\")[-1]
                self.status_label.config(text=f'✓ Đã tải: {filename}')

        except Exception as e:
            messagebox.showerror('Lỗi', f'Không thể mở ảnh:\n{str(e)}')

    def recognize_plate(self):
        """Nhận dạng - SỬ DỤNG PIPELINE"""
        if not self.file_path:
            messagebox.showwarning('Cảnh báo', 'Vui lòng chọn ảnh trước!')
            return

        self.progress.start()
        self.status_label.config(text='⏳ Đang xử lý pipeline...')
        self.root.update()

        try:
            # CHẠY PIPELINE
            self.result_data = self.pipeline.process(self.file_path)

            # Hiển thị các bước xử lý
            steps = self.result_data['processing_steps']
            self._display_cv_image(steps['blackhat'], self.processing_imgs[0], True, 280, 500)
            self._display_cv_image(steps['sobel'], self.processing_imgs[1], True, 280, 500)
            self._display_cv_image(steps['threshold'], self.processing_imgs[2], True, 280, 500)

            self._display_cv_image(steps['detection'], self.processing_imgs[3], False, 280, 500)
            binary_img = self.pipeline.recognizer.get_binary_image()
            self._display_cv_image(binary_img, self.processing_imgs[4], True, 280, 500)

            self._display_cv_image(self.result_data['plate_image'], self.plate_img, True, 460, 530)

            # Hiển thị kết quả text
            text = self.result_data['text']
            if text and text != "Không nhận dạng được":
                self.result_text.config(text=text, fg='#2ecc71')
                self.status_label.config(text=f'✓ Nhận dạng thành công: {text}')
            else:
                self.result_text.config(text='Không nhận dạng được', fg='#e74c3c')
                self.status_label.config(text='⚠ Không nhận dạng được biển số')

            # Timestamp
            now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            self.timestamp.config(text=f'Thời gian: {now}')

            self.progress.stop()

        except Exception as e:
            self.progress.stop()
            self.status_label.config(text=f'❌ Lỗi: {str(e)}')
            messagebox.showerror('Lỗi xử lý', f'Chi tiết lỗi:\n{str(e)}')

    def reset_app(self):
        """Reset toàn bộ"""
        self.file_path = None
        self.result_data = None

        self.original_img.config(image='')
        self.plate_img.config(image='')

        for img_label in self.processing_imgs:
            img_label.config(image='')

        self.result_text.config(text='Chưa nhận dạng', fg='#95a5a6')
        self.timestamp.config(text='')

        self.recognize_btn.config(state=DISABLED)
        self.status_label.config(text='Trạng thái: Sẵn sàng')

        self.switch_mode('result')

    def _resize_keep_ratio(self, pil_image, max_width, max_height):
        """Resize giữ tỷ lệ"""
        pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil_image)

    def _display_cv_image(self, cv_image, label_widget, is_gray, max_w, max_h):
        if cv_image is None:
            return

        if is_gray and len(cv_image.shape) == 2:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        elif len(cv_image.shape) == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(cv_image)
        pil_image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_image)

        label_widget.configure(image=photo)
        label_widget.image = photo

# Main
if __name__ == '__main__':
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()