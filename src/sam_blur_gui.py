import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QWheelEvent, QMouseEvent, QMovie
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QComboBox,
    QWidget, QPushButton, QMessageBox, QLabel as QtLabel, QDialog, QListWidget
)

BASE_DIR = Path("../data/")
ORIGINAL_DIR = BASE_DIR / "raw"
SEMI_BLURRED_DIR = BASE_DIR / "verify"
FINAL_DIR = BASE_DIR / "final"

parser = argparse.ArgumentParser(
    description="Final Blur GUI — blur images via clicks or rectangular selections"
)
parser.add_argument(
    "--disable-sam",
    action="store_true",
    help="Disable MobileSAM segmentation (clicks will be ignored)"
)
args = parser.parse_args()

# 2) Only build a SAMSegmenter if not disabled
segmenter = None
if not args.disable_sam:
    import torch
    from mobile_sam import sam_model_registry, SamPredictor

    class SAMSegmenter:
        def __init__(self, checkpoint_path="MobileSAM/weights/mobile_sam.pt", model_type="vit_t"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(self.device)
            self.predictor = SamPredictor(sam)

        def get_mask_from_click(self, image: np.ndarray, click_point: tuple[int, int]) -> np.ndarray:
            self.predictor.set_image(image)
            input_point = np.array([click_point])
            input_label = np.array([1])
            masks, scores, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            smallest_idx = np.argmin([np.sum(m) for m in masks])
            return masks[smallest_idx].astype(np.uint8)

    segmenter = SAMSegmenter()
else:
    print("⚠️  MobileSAM has been disabled; click-to-segment is turned off.")

def cv2_to_qpix(img: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self._orig_img = None
        self._orig_pixmap = None
        self._file_path = None
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._dragging = False
        self._drag_start = QPointF()
        self._drawing_rect = False
        self._rect_start = None
        self._rect_end = None

        # Undo/Redo stacks
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []

    def _push_undo(self):
        if self._orig_img is not None:
            self._undo_stack.append(self._orig_img.copy())

    def apply_image(self, new_img: np.ndarray):
        self._push_undo()
        self._redo_stack.clear()
        self._orig_img = new_img
        self._orig_pixmap = cv2_to_qpix(new_img)
        self.update()

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self._orig_img.copy())
        img = self._undo_stack.pop()
        self._orig_img = img
        self._orig_pixmap = cv2_to_qpix(img)
        self.update()

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._orig_img.copy())
        img = self._redo_stack.pop()
        self._orig_img = img
        self._orig_pixmap = cv2_to_qpix(img)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            # panning
            self._dragging = True
            self._drag_start = event.pos()

        elif event.button() == Qt.LeftButton and self._orig_pixmap:
            # rectangular blur (Shift+click-drag) still works
            if QApplication.keyboardModifiers() == Qt.ShiftModifier:
                self._drawing_rect = True
                self._rect_start = event.pos()
                self._rect_end = event.pos()

            # otherwise: click-to-segment
            else:
                # if SAM is disabled, do nothing on plain clicks
                if segmenter is None:
                    return

                click_pos = event.pos()
                img = self._orig_img.copy()
                scaled = self._scaled_pixmap()
                disp_w, disp_h = scaled.width(), scaled.height()
                img_x = (self.width() - disp_w) / 2 + self._pan_offset.x()
                img_y = (self.height() - disp_h) / 2 + self._pan_offset.y()
                rel_x = (click_pos.x() - img_x) / disp_w
                rel_y = (click_pos.y() - img_y) / disp_h
                if not (0 <= rel_x <= 1 and 0 <= rel_y <= 1):
                    return
                cx = int(rel_x * img.shape[1])
                cy = int(rel_y * img.shape[0])

                # now safe to call into MobileSAM
                mask = segmenter.get_mask_from_click(img, (cx, cy))
                blurred = cv2.GaussianBlur(img, (21, 21), sigmaX=15)
                new_img = img.copy()
                new_img[mask == 1] = blurred[mask == 1]
                self.apply_image(new_img)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            delta = event.pos() - self._drag_start
            self._pan_offset += delta
            self._drag_start = event.pos()
            self.update()
        elif self._drawing_rect:
            self._rect_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging = False
        if self._drawing_rect:
            self._drawing_rect = False
            if self._rect_start and self._rect_end:
                x1, y1 = self._rect_start.x(), self._rect_start.y()
                x2, y2 = self._rect_end.x(), self._rect_end.y()
                rect = QRectF(QPointF(min(x1, x2), min(y1, y2)), QPointF(max(x1, x2), max(y1, y2)))
                img = self._orig_img.copy()
                scaled = self._scaled_pixmap()
                disp_w, disp_h = scaled.width(), scaled.height()
                img_x = (self.width() - disp_w) / 2 + self._pan_offset.x()
                img_y = (self.height() - disp_h) / 2 + self._pan_offset.y()
                rel_x1 = (rect.left() - img_x) / disp_w
                rel_y1 = (rect.top() - img_y) / disp_h
                rel_x2 = (rect.right() - img_x) / disp_w
                rel_y2 = (rect.bottom() - img_y) / disp_h
                h, w = img.shape[:2]
                ix1 = int(np.clip(rel_x1 * w, 0, w))
                iy1 = int(np.clip(rel_y1 * h, 0, h))
                ix2 = int(np.clip(rel_x2 * w, 0, w))
                iy2 = int(np.clip(rel_y2 * h, 0, h))
                if ix2 > ix1 and iy2 > iy1:
                    new_img = img.copy()
                    new_img[iy1:iy2, ix1:ix2] = cv2.GaussianBlur(img[iy1:iy2, ix1:ix2], (21, 21), sigmaX=15)
                    self.apply_image(new_img)
            self.update()

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom = max(0.1, min(10.0, self._zoom * factor))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._orig_pixmap:
            return
        painter = QPainter(self)
        scaled = self._scaled_pixmap()
        rect = QRectF((self.width() - scaled.width()) / 2 + self._pan_offset.x(), (self.height() - scaled.height()) / 2 + self._pan_offset.y(), scaled.width(), scaled.height())
        painter.drawPixmap(rect.topLeft(), scaled)
        if self._drawing_rect and self._rect_start and self._rect_end:
            painter.setPen(QPen(Qt.red, 2, Qt.DashLine))
            painter.drawRect(QRectF(self._rect_start, self._rect_end))

    def _scaled_pixmap(self) -> QPixmap:
        return self._orig_pixmap.scaled(self.size() * self._zoom, Qt.KeepAspectRatio, Qt.SmoothTransformation) if self._orig_pixmap else QPixmap()

    def load_image(self, img: np.ndarray, file_path: str):
        self._orig_img = img.copy()
        self._orig_pixmap = cv2_to_qpix(img)
        self._file_path = file_path
        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.update()

    def refresh(self):
        if self._file_path:
            img = cv2.imread(self._file_path)
            if img is not None:
                self.load_image(img, self._file_path)

    def use_original(self, original_path):
        img = cv2.imread(str(original_path))
        if img is not None:
            self.load_image(img, str(original_path))

    def save_to(self, save_path: Path):
        if self._orig_img is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), self._orig_img)

class ChecklistDialog(QDialog):
    def __init__(self, parent, image_paths, current_folder):
        super().__init__(parent)
        self.setWindowTitle("Image Checklist")
        self.resize(400, 600)
        layout = QVBoxLayout(self)
        self.loading_label = QtLabel("Loading checklist...")
        self.loading_gif = QtLabel()
        self.movie = QMovie("loading.gif")
        self.loading_gif.setMovie(self.movie)
        self.movie.start()
        layout.addWidget(self.loading_label)
        layout.addWidget(self.loading_gif)
        QApplication.processEvents()
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        last_saved_idx = parent.last_saved_index()
        for i, img_path in enumerate(image_paths):
            rel_path = img_path.relative_to(SEMI_BLURRED_DIR)
            final_path = FINAL_DIR / rel_path
            status = "✅" if final_path.exists() else "❌"
            self.list_widget.addItem(f"{status}  {rel_path}")
        layout.removeWidget(self.loading_label)
        layout.removeWidget(self.loading_gif)
        self.loading_label.deleteLater()
        self.loading_gif.deleteLater()
        self.list_widget.itemClicked.connect(lambda item: parent.jump_to_image(item.text()))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Final Blur GUI")
        self.resize(1000, 800)
        self.image_label = ImageLabel()
        self.combo = QComboBox()
        self.combo.addItems(sorted([p.name for p in SEMI_BLURRED_DIR.iterdir() if p.is_dir()]))
        self.combo.currentTextChanged.connect(self.load_selected_folder)
        self.semi_paths = []
        self.current_index = 0
        self.current_subfolder = self.combo.currentText()
        self.status_label = QtLabel("Status: ❌ Not Saved")
        self.count_label = QtLabel("Saved: 0 / 0")
        btn_refresh = QPushButton("Refresh")
        btn_original = QPushButton("Use Original")
        btn_save_next = QPushButton("Save + Next")
        btn_undo = QPushButton("Undo")
        btn_redo = QPushButton("Redo")
        btn_last_saved = QPushButton("Go to Last Saved")
        btn_checklist = QPushButton("Checklist")
        btn_refresh.clicked.connect(self.refresh)
        btn_original.clicked.connect(self.use_original)
        btn_save_next.clicked.connect(self.save_and_next)
        btn_undo.clicked.connect(self.image_label.undo)
        btn_redo.clicked.connect(self.image_label.redo)
        btn_last_saved.clicked.connect(self.goto_last_saved)
        btn_checklist.clicked.connect(self.show_checklist_dialog)
        main_layout = QVBoxLayout()
        btn_row = QHBoxLayout()
        btn_row.addWidget(QtLabel("Subfolder:"))
        btn_row.addWidget(self.combo)
        btn_row.addWidget(btn_refresh)
        btn_row.addWidget(btn_original)
        btn_row.addWidget(btn_undo)
        btn_row.addWidget(btn_redo)
        btn_row.addWidget(btn_save_next)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status_label)
        status_row.addSpacing(20)
        status_row.addWidget(self.count_label)
        status_row.addStretch()
        status_row.addWidget(btn_last_saved)
        status_row.addWidget(btn_checklist)
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(btn_row)
        main_layout.addLayout(status_row)
        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.load_selected_folder(self.current_subfolder)

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.image_label.undo()
        elif event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Y:
            self.image_label.redo()
        elif event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_Left:
            self.prev_image()

    def load_selected_folder(self, folder_name):
        self.current_subfolder = folder_name
        folder_path = SEMI_BLURRED_DIR / folder_name
        self.semi_paths = sorted(list(folder_path.rglob("*.jpg")) + list(folder_path.rglob("*.png")))
        self.current_index = 0
        if self.semi_paths:
            self.load_image_at(self.current_index)

    def load_image_at(self, index):
        if 0 <= index < len(self.semi_paths):
            path = self.semi_paths[index]
            img = cv2.imread(str(path))
            if img is not None:
                self.image_label.load_image(img, str(path))
                self.current_img_path = path
                self.update_status_label()

    def update_status_label(self):
        rel_path = self.current_img_path.relative_to(SEMI_BLURRED_DIR)
        final_path = FINAL_DIR / rel_path
        if final_path.exists():
            self.status_label.setText("Status: ✅ Saved")
        else:
            self.status_label.setText("Status: ❌ Not Saved")
        saved = sum((FINAL_DIR / p.relative_to(SEMI_BLURRED_DIR)).exists() for p in self.semi_paths)
        self.count_label.setText(f"Saved: {saved} / {len(self.semi_paths)}")

    def refresh(self):
        self.image_label.refresh()
        self.update_status_label()

    def use_original(self):
        # get the verify path relative to SEMI_BLURRED_DIR, e.g. "subdir/0123.png"
        rel_path = self.current_img_path.relative_to(SEMI_BLURRED_DIR)
        stem = rel_path.stem   # e.g. "0123"
        suffix = rel_path.suffix

        # parse out the integer frame number
        try:
            frame_num = int(stem)
        except ValueError:
            QMessageBox.warning(self, "Error", f"Cannot parse frame number from '{stem}'")
            return

        # look in the same subfolder under raw/
        search_dir = ORIGINAL_DIR / rel_path.parent
        if not search_dir.exists():
            QMessageBox.warning(self, "Not Found", f"Raw folder not found:\n{search_dir}")
            return

        # scan for any file whose stem, when int()-ed, == frame_num
        for cand in search_dir.glob(f"*{suffix}"):
            try:
                if int(cand.stem) == frame_num:
                    # found it!
                    self.image_label.use_original(cand)
                    return
            except ValueError:
                continue

        # if we get here, no match
        QMessageBox.warning(
            self,
            "Not Found",
            f"No raw file matching frame {frame_num} in\n{search_dir}"
        )

    def save_and_next(self):
        rel_path = self.current_img_path.relative_to(SEMI_BLURRED_DIR)
        final_path = FINAL_DIR / rel_path
        self.image_label.save_to(final_path)
        self.update_status_label()
        self.next_image()

    def next_image(self):
        self.current_index += 1
        if self.current_index >= len(self.semi_paths):
            QMessageBox.information(self, "Done", "All images processed.")
        else:
            self.load_image_at(self.current_index)

    def prev_image(self):
        self.current_index -= 1
        if self.current_index < 0:
            self.current_index = 0
        self.load_image_at(self.current_index)

    def goto_last_saved(self):
        for i in reversed(range(len(self.semi_paths))):
            rel = self.semi_paths[i].relative_to(SEMI_BLURRED_DIR)
            if (FINAL_DIR / rel).exists():
                self.current_index = i
                self.load_image_at(i)
                return
        QMessageBox.information(self, "Info", "No saved images found in this folder.")

    def last_saved_index(self):
        for i in reversed(range(len(self.semi_paths))):
            rel = self.semi_paths[i].relative_to(SEMI_BLURRED_DIR)
            if (FINAL_DIR / rel).exists():
                return i
        return -1

    def jump_to_image(self, text):
        name = text.split(maxsplit=1)[1].strip()
        for i, p in enumerate(self.semi_paths):
            if str(p.relative_to(SEMI_BLURRED_DIR)) == name:
                self.current_index = i
                self.load_image_at(i)
                return

    def show_checklist_dialog(self):
        dialog = ChecklistDialog(self, self.semi_paths, self.current_subfolder)
        dialog.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())