"""
gui_components.py
Các component tái sử dụng cho giao diện
"""

import tkinter as tk
from tkinter import Frame, Label, LabelFrame, Canvas, Text, ttk
from PIL import ImageTk, Image
import cv2
import numpy as np
from config import COLORS


# ============================================
# SCROLLABLE FRAME
# ============================================
class ScrollableFrame(Frame):
    """Frame có thể cuộn được"""

    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)

        # Tạo canvas và scrollbar
        self.canvas = Canvas(self, bg=COLORS['bg_dark'], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas, bg=COLORS['bg_dark'])

        # Bind events
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack canvas và scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Hỗ trợ cuộn bằng chuột
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        """Xử lý sự kiện cuộn chuột"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


# ============================================
# IMAGE DISPLAY HELPER
# ============================================
class ImageDisplay:
    """Helper class để hiển thị ảnh OpenCV trong Tkinter"""

    # Thumbnail size mặc định cho grid layout
    GRID_THUMBNAIL_SIZE = (350, 250)
    LARGE_THUMBNAIL_SIZE = (550, 400)

    @staticmethod
    def cv_to_tk(cv_image, is_gray=False, size=None):
        """
        Chuyển ảnh OpenCV sang Tkinter PhotoImage

        Args:
            cv_image: Ảnh OpenCV (numpy array)
            is_gray: True nếu ảnh grayscale
            size: Tuple (width, height) cho thumbnail, None = sử dụng mặc định

        Returns:
            ImageTk.PhotoImage object
        """
        # Chuyển đổi color space
        if is_gray and len(cv_image.shape) == 2:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        elif len(cv_image.shape) == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Chuyển sang PIL Image
        pil_image = Image.fromarray(cv_image)

        # Resize nếu cần
        if size is None:
            size = ImageDisplay.GRID_THUMBNAIL_SIZE
        pil_image.thumbnail(size, Image.Resampling.LANCZOS)

        # Chuyển sang PhotoImage
        return ImageTk.PhotoImage(pil_image)

    @staticmethod
    def display_image(cv_image, label_widget, is_gray=False, size=None):
        """
        Hiển thị ảnh OpenCV lên Label widget

        Args:
            cv_image: Ảnh OpenCV
            label_widget: Tkinter Label widget
            is_gray: True nếu ảnh grayscale
            size: Tuple (width, height) cho thumbnail
        """
        photo = ImageDisplay.cv_to_tk(cv_image, is_gray, size)
        label_widget.configure(image=photo)
        label_widget.image = photo  # Giữ reference để tránh garbage collection

    @staticmethod
    def clear_image(label_widget):
        """Xóa ảnh khỏi label"""
        label_widget.configure(image='')
        label_widget.image = None


# ============================================
# LABELED FRAME FACTORY
# ============================================
def create_labeled_frame(parent, title, title_color=None, **kwargs):
    """
    Tạo LabelFrame với style thống nhất

    Args:
        parent: Widget cha
        title: Tiêu đề của frame
        title_color: Màu tiêu đề (None = dùng primary color)
        **kwargs: Các tham số bổ sung cho LabelFrame

    Returns:
        LabelFrame widget
    """
    if title_color is None:
        title_color = COLORS['primary']

    # Merge kwargs với default settings
    default_settings = {
        'bg': COLORS['bg_light'],
        'fg': title_color,
        'font': ('Arial', 11, 'bold'),
        'bd': 3,
        'relief': tk.RIDGE
    }
    default_settings.update(kwargs)

    frame = LabelFrame(parent, text=title, **default_settings)
    return frame


# ============================================
# STYLED BUTTON FACTORY
# ============================================
def create_styled_button(parent, text, command=None, button_type='primary', **kwargs):
    """
    Tạo Button với style định sẵn

    Args:
        parent: Widget cha
        text: Text của button
        command: Callback function
        button_type: 'primary', 'success', 'danger', 'warning', 'info'
        **kwargs: Các tham số bổ sung

    Returns:
        Button widget
    """
    # Color schemes
    color_map = {
        'primary': '#3498db',
        'success': '#27ae60',
        'danger': '#e74c3c',
        'warning': '#f39c12',
        'info': '#9b59b6',
        'default': '#95a5a6'
    }

    bg_color = color_map.get(button_type, color_map['default'])

    # Default settings
    default_settings = {
        'bg': bg_color,
        'fg': 'white',
        'font': ('Arial', 11, 'bold'),
        'padx': 18,
        'pady': 8,
        'cursor': 'hand2',
        'relief': tk.RAISED,
        'bd': 2,
        'activebackground': bg_color,
        'activeforeground': 'white'
    }
    default_settings.update(kwargs)

    if command:
        default_settings['command'] = command

    button = tk.Button(parent, text=text, **default_settings)
    return button


# ============================================
# INFO TEXT DISPLAY
# ============================================
class InfoTextDisplay:
    """Helper class để hiển thị thông tin dạng text"""

    @staticmethod
    def create_info_text(parent, height=10, width=40):
        """
        Tạo Text widget để hiển thị thông tin

        Returns:
            Text widget
        """
        text_widget = Text(
            parent,
            bg=COLORS['bg_medium'],
            fg=COLORS['text_dark'],
            font=('Courier', 9),
            height=height,
            width=width,
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        return text_widget

    @staticmethod
    def display_plate_info(text_widget, info_dict):
        """
        Hiển thị thông tin phát hiện biển số

        Args:
            text_widget: Text widget
            info_dict: Dictionary chứa thông tin (từ LicensePlateProcessor)
        """
        text_widget.delete(1.0, tk.END)

        info_str = "✓ PHÁT HIỆN THÀNH CÔNG!\n"
        info_str += "=" * 35 + "\n\n"
        info_str += f"Loại xe:       {info_dict.get('type', 'N/A')}\n"
        info_str += f"Vị trí (x,y):  ({info_dict.get('x', 0)}, {info_dict.get('y', 0)})\n"
        info_str += f"Kích thước:    {info_dict.get('w', 0)} x {info_dict.get('h', 0)} px\n"
        info_str += f"Tỉ lệ W/H:     {info_dict.get('aspectRatio', 0):.2f}\n"
        info_str += f"Solidity:      {info_dict.get('solidity', 0):.3f}\n"
        info_str += f"Diện tích:     {info_dict.get('area', 0):.0f} px²\n"

        text_widget.insert(tk.END, info_str)

    @staticmethod
    def display_error(text_widget, error_message):
        """Hiển thị thông báo lỗi"""
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"❌ LỖI:\n{error_message}")


# ============================================
# PROGRESS INDICATOR
# ============================================
class ProgressIndicator:
    """Helper class để quản lý progress bar"""

    @staticmethod
    def create_progress_bar(parent, length=400):
        """Tạo progress bar"""
        return ttk.Progressbar(parent, mode='indeterminate', length=length)

    @staticmethod
    def start(progress_bar, root=None):
        """Bắt đầu progress"""
        progress_bar.start()
        if root:
            root.update()

    @staticmethod
    def stop(progress_bar):
        """Dừng progress"""
        progress_bar.stop()
