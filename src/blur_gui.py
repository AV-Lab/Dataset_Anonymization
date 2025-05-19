import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRect, QPoint, QSize
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QSlider,
    QGroupBox,
    QMessageBox,
)

"""Image Blur GUI  ░ v3-fix ░
--------------------------------------------------
* Drag–drop an image.
* **Start** → click centre → resize rectangle → **Finish** to blur.
* **Undo** undoes the most recent blur.
* **Save** overwrites the *original* file automatically.
* The picture rescales when you resize the main window.
"""

# -----------------------------------------------------------------------------
# Helpers: OpenCV ↔ Qt
# -----------------------------------------------------------------------------

def cv2_to_qpix(img: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))


def qpix_to_cv2(pix: QPixmap) -> np.ndarray:
    qimg = pix.toImage().convertToFormat(QImage.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    ptr.setsize(h * w * 3)
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# -----------------------------------------------------------------------------
# ImageLabel
# -----------------------------------------------------------------------------
class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumSize(200, 200)

        self._orig_pixmap: QPixmap | None = None  # current edited image
        self._undo_pixmap: QPixmap | None = None  # one-step undo buffer
        self._center_point: QPoint | None = None  # selection centre (label coords)
        self._selecting = False
        self.rect_size = (100, 100)  # (w, h) in label pixels
        self._file_path: str | None = None       # original file location

    # ---------- Drag & Drop ----------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Error", "Unsupported image file.")
            return
        self._file_path = path
        self._orig_pixmap = cv2_to_qpix(img)
        self._undo_pixmap = None
        self._center_point = None
        self.setPixmap(self._scaled_pixmap())

    # ---------- Mouse ----------
    def mousePressEvent(self, event):
        if self._selecting and event.button() == Qt.LeftButton and self._orig_pixmap:
            self._center_point = event.pos()
            self.update()

    # ---------- Painting preview rectangle ----------
    def paintEvent(self, event):
        super().paintEvent(event)
        if self._center_point and self._orig_pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(Qt.red, 2))
            w, h = self.rect_size
            top_left = QPoint(self._center_point.x() - w // 2, self._center_point.y() - h // 2)
            painter.drawRect(QRect(top_left, QSize(w, h)))

    # ---------- Resize ----------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._orig_pixmap:
            self.setPixmap(self._scaled_pixmap())

    # ---------- Helpers ----------
    def _scaled_pixmap(self) -> QPixmap:
        if not self._orig_pixmap:
            return QPixmap()
        return self._orig_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def set_selecting(self, state: bool):
        self._selecting = state
        if not state:
            self.update()

    def set_rect_size(self, w: int, h: int):
        self.rect_size = (w, h)
        self.update()

    # ---------- Blur ----------
    def apply_blur(self) -> bool:
        if not (self._orig_pixmap and self._center_point):
            return False

        # store snapshot for undo
        self._undo_pixmap = self._orig_pixmap.copy()

        img = qpix_to_cv2(self._orig_pixmap)
        h_img, w_img = img.shape[:2]
        disp_pix = self._scaled_pixmap()
        disp_w, disp_h = disp_pix.width(), disp_pix.height()
        off_x = (self.width() - disp_w) // 2
        off_y = (self.height() - disp_h) // 2
        rel_x = self._center_point.x() - off_x
        rel_y = self._center_point.y() - off_y
        if not (0 <= rel_x < disp_w and 0 <= rel_y < disp_h):
            return False
        scale = disp_w / w_img
        cx, cy = int(rel_x / scale), int(rel_y / scale)
        rw, rh = int(self.rect_size[0] / scale), int(self.rect_size[1] / scale)
        x1, y1 = max(cx - rw // 2, 0), max(cy - rh // 2, 0)
        x2, y2 = min(cx + rw // 2, w_img), min(cy + rh // 2, h_img)
        if x2 <= x1 or y2 <= y1:
            return False
        img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (0, 0), sigmaX=7, sigmaY=7)
        self._orig_pixmap = cv2_to_qpix(img)
        self._center_point = None
        self.setPixmap(self._scaled_pixmap())
        self.update()
        return True

    # ---------- Undo ----------
    def undo(self):
        if self._undo_pixmap is None:
            QMessageBox.information(self, "Undo", "Nothing to undo.")
            return
        self._orig_pixmap = self._undo_pixmap
        self._undo_pixmap = None
        self.setPixmap(self._scaled_pixmap())
        self.update()

    # ---------- Save ----------
    def save_over_original(self) -> bool:
        if not (self._orig_pixmap and self._file_path):
            return False
        img = qpix_to_cv2(self._orig_pixmap)
        ok = cv2.imwrite(self._file_path, img)
        if not ok:
            QMessageBox.warning(self, "Error", "Failed to write the file.")
        return ok


# -----------------------------------------------------------------------------
# Main Window
# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Blur GUI")
        self.resize(900, 700)

        self.image_label = ImageLabel()

        # -------- Buttons --------
        btn_start = QPushButton("Start")
        btn_finish = QPushButton("Finish")
        btn_undo = QPushButton("Undo")
        btn_save = QPushButton("Save")

        btn_start.clicked.connect(lambda: self.image_label.set_selecting(True))
        btn_finish.clicked.connect(self.finish_clicked)
        btn_undo.clicked.connect(self.image_label.undo)
        btn_save.clicked.connect(self.save_clicked)

        # -------- Sliders --------
        self.slider_w = QSlider(Qt.Horizontal)
        self.slider_h = QSlider(Qt.Horizontal)
        for s in (self.slider_w, self.slider_h):
            s.setRange(20, 500)
            s.setValue(100)
            s.valueChanged.connect(self.update_size)

        size_box = QGroupBox("Rectangle Size")
        v_size = QVBoxLayout(size_box)
        v_size.addWidget(QLabel("Width"))
        v_size.addWidget(self.slider_w)
        v_size.addWidget(QLabel("Height"))
        v_size.addWidget(self.slider_h)

        btn_row = QHBoxLayout()
        btn_row.addWidget(btn_start)
        btn_row.addWidget(btn_finish)
        btn_row.addWidget(btn_undo)
        btn_row.addWidget(btn_save)

        main = QVBoxLayout()
        main.addWidget(self.image_label, 1)
        main.addLayout(btn_row)
        main.addWidget(size_box)

        central = QWidget()
        central.setLayout(main)
        self.setCentralWidget(central)

    # ---------- Slots ----------
    def update_size(self):
        """Sync slider values to rectangle size."""
        self.image_label.set_rect_size(self.slider_w.value(), self.slider_h.value())

    def finish_clicked(self):
        if not self.image_label.apply_blur():
            QMessageBox.information(self, "Info", "Click **Start**, then choose a point inside the image, then **Finish**.")
        self.image_label.set_selecting(False)

    def save_clicked(self):
        if not self.image_label.save_over_original():
            QMessageBox.warning(self, "Error", "No image loaded or unable to save.")
        else:
            QMessageBox.information(self, "Saved", "Changes written to original file.")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        win = MainWindow()
        win.show()
        sys.exit(app.exec_())
    except Exception as e:
        # Print to console so user sees why it crashed
        print("Unhandled exception:", e)
