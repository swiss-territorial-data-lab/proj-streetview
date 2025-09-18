"""
COCO_validator.py

A PyQt-based GUI tool for validating and editing COCO-format polygon annotations.
Allows users to visually inspect, move, and scale polygons, and mark annotations as valid or removed.
Validation status is cached and can be exported to a new COCO file.
"""

import sys
import os
import json
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QSpinBox, QGraphicsView, QGraphicsScene, QGraphicsPolygonItem, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QPolygonF, QPen, QColor
from PyQt5.QtCore import Qt, QPointF

def polygon_to_mask(polygon, height, width):
    """
    Convert COCO polygon segmentation to a binary mask, clipping coordinates within image bounds.

    Args:
        polygon (list): List of polygons (each a list of x,y coordinates).
        height (int): Image height.
        width (int): Image width.

    Returns:
        np.ndarray: Binary mask of shape (height, width).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygon:
        return mask
    pts = []
    for seg in polygon:
        arr = np.array(seg, dtype=np.float32).reshape(-1, 2)
        # Clip x to [0, width-1], y to [0, height-1]
        arr[:, 0] = np.clip(arr[:, 0], 0, width - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, height - 1)
        pts.append(arr.astype(np.int32))
    cv2.fillPoly(mask, pts, 1)
    return mask

class MaskPolygonItem(QGraphicsPolygonItem):
    """
    QGraphicsPolygonItem subclass for displaying and interacting with annotation polygons.
    Allows moving and selecting the polygon in the scene.
    """
    def __init__(self, polygon, parent=None):
        super().__init__(polygon, parent)
        self.setFlag(QGraphicsPolygonItem.ItemIsMovable, True)
        self.setFlag(QGraphicsPolygonItem.ItemIsSelectable, True)
        self.setPen(QPen(QColor(0, 255, 0), 2))
        self.setBrush(QColor(0, 255, 0, 80))
        self.ori_pos = polygon  # Original polygon position

class COCOValidatorWindow(QMainWindow):
    """
    Main window for the COCO annotation validator GUI.
    Handles loading, displaying, editing, and exporting COCO annotations.
    """
    def __init__(self):
        super().__init__()
        # Handle both script and PyInstaller executable paths
        if getattr(sys, 'frozen', False):
            # Running as PyInstaller executable
            base_path = os.path.dirname(sys.executable)
        else:
            # Running as script
            base_path = os.path.dirname(os.path.realpath(__file__))
        self.CACHE_FILE = os.path.join(base_path, '.coco_validator_cache.json')

        self.setWindowTitle('COCO Annotation Validator (PyQt)')
        self.resize(1200, 900)
        self.coco_data = None
        self.image_dir = ''
        self.annotations = []
        self.images = []
        self.current_idx = 0
        self.buffer_size = 200
        self.validation_status = {}  # annotation_id: {'validated': True/False, 'removed': True/False}
        self.current_patch = None
        self.current_mask = None
        self.current_ann = None
        self.current_img_meta = None
        self.last_coco_path = ''
        self.last_img_dir = ''
        self.last_buffer_size = 200
        self._polygon_moved = False  # Track if polygon was moved

        self.init_ui()
        self.load_cache()

    def init_ui(self):
        """
        Initialize the user interface: layouts, widgets, and their connections.
        """
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Left: Image and mask display
        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        main_layout.addWidget(self.graphics_view, 3)

        # Right: Controls (vertical column)
        right_panel = QVBoxLayout()
        self.info_label = QLabel('No annotation loaded.')
        right_panel.addWidget(self.info_label)

        # Go to toolbar (annotation index navigation)
        goto_layout = QHBoxLayout()
        goto_layout.addWidget(QLabel("Go to:"))
        self.index_spinbox = QSpinBox()
        self.index_spinbox.setMinimum(1)
        self.index_spinbox.setMaximum(1)  # Will be updated when COCO is loaded
        self.index_spinbox.valueChanged.connect(self.go_to_annotation)
        goto_layout.addWidget(self.index_spinbox)
        right_panel.addLayout(goto_layout)

        # Zoom controls (buffer size adjustment)
        zoom_btns = QHBoxLayout()
        self.zoom_out_btn = QPushButton('Zoom +')
        self.zoom_out_btn.clicked.connect(self.decrease_buffer)
        zoom_btns.addWidget(self.zoom_out_btn)
        self.zoom_in_btn = QPushButton('Zoom -')
        self.zoom_in_btn.clicked.connect(self.increase_buffer)
        zoom_btns.addWidget(self.zoom_in_btn)
        right_panel.addLayout(zoom_btns)

        right_panel.addStretch()  # Adaptive space after zoom

        # Keep/Remove group (validation controls)
        keep_remove_group = QVBoxLayout()
        self.keep_btn = QPushButton('Keep')
        self.keep_btn.clicked.connect(self.keep_current)
        keep_remove_group.addWidget(self.keep_btn)
        self.remove_btn = QPushButton('Remove')
        self.remove_btn.clicked.connect(self.remove_current)
        keep_remove_group.addWidget(self.remove_btn)
        right_panel.addLayout(keep_remove_group)

        right_panel.addStretch()  # Adaptive space after keep/remove

        # Navigation group (annotation navigation)
        nav_group = QVBoxLayout()
        self.prev_btn = QPushButton('Prev')
        self.prev_btn.clicked.connect(self.prev_annotation)
        nav_group.addWidget(self.prev_btn)
        self.next_btn = QPushButton('Next')
        self.next_btn.clicked.connect(self.next_annotation)
        nav_group.addWidget(self.next_btn)
        self.prev_unval_btn = QPushButton('Prev Unvalidated')
        self.prev_unval_btn.clicked.connect(self.prev_unvalidated)
        nav_group.addWidget(self.prev_unval_btn)
        self.next_unval_btn = QPushButton('Next Unvalidated')
        self.next_unval_btn.clicked.connect(self.next_unvalidated)
        nav_group.addWidget(self.next_unval_btn)
        right_panel.addLayout(nav_group)

        right_panel.addStretch(20)  # Adaptive space after navigation

        # Load and export buttons
        self.load_btn = QPushButton('Load COCO')
        self.load_btn.clicked.connect(self.load_coco)
        right_panel.addWidget(self.load_btn)

        self.export_btn = QPushButton('Export Validated')
        self.export_btn.clicked.connect(self.export_validated)
        right_panel.addWidget(self.export_btn)

        main_layout.addLayout(right_panel, 1)
        self.setCentralWidget(main_widget)

        # Keyboard shortcuts
        self.graphics_view.setFocusPolicy(Qt.StrongFocus)
        self.graphics_view.keyPressEvent = self.keyPressEvent

    def load_cache(self):
        """
        Load cached validation status and last-used paths from disk.
        Prompts user to select new paths if previous ones are missing.
        """
        if os.path.isfile(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, 'r') as f:
                    cache = json.load(f)
                self.last_coco_path = cache.get('last_coco_path', '')
                self.last_img_dir = cache.get('last_img_dir', '')
                self.last_buffer_size = cache.get('buffer_size', 200)
                self.validation_status = cache.get('validation_status', {})
                self.buffer_size = self.last_buffer_size
                
                # Check if paths exist
                if self.last_coco_path == None or not os.path.isfile(self.last_coco_path):
                    QMessageBox.warning(self, 'Path Error', 
                        f'COCO file not found at {self.last_coco_path}. Please select a new file.')
                    self.last_coco_path, _ = QFileDialog.getOpenFileName(self, 'Open COCO JSON', '', 'JSON Files (*.json)')
                if self.last_img_dir == None or not os.path.isdir(self.last_img_dir):
                    QMessageBox.warning(self, 'Path Error', 
                        f'Image directory not found at {self.last_img_dir}. Please select a new directory.')
                    self.last_img_dir = QFileDialog.getExistingDirectory(self, 'Select Image Directory')
                
                self.save_cache()
                # If both paths are valid, load automatically
                if self.last_coco_path and self.last_img_dir:
                    self.load_coco(self.last_coco_path, self.last_img_dir, resume=True)
            except Exception as e:
                QMessageBox.warning(self, 'Cache Error', f'Error loading cache: {str(e)}')
                self.last_coco_path = None
                self.last_img_dir = None

    def save_cache(self):
        """
        Save current validation status and paths to disk for later resumption.
        """
        cache = {
            'last_coco_path': self.last_coco_path,
            'last_img_dir': self.last_img_dir,
            'buffer_size': self.buffer_size,
            'validation_status': self.validation_status
        }
        with open(self.CACHE_FILE, 'w') as f:
            json.dump(cache, f)

    def load_coco(self, coco_path=None, img_dir=None, resume=False):
        """
        Load a COCO JSON file and associated image directory.

        Args:
            coco_path (str, optional): Path to COCO JSON file.
            img_dir (str, optional): Path to image directory.
            resume (bool): If True, resume at last validated annotation.
        """
        if not coco_path:
            coco_path, _ = QFileDialog.getOpenFileName(self, 'Open COCO JSON', '', 'JSON Files (*.json)')
            if not coco_path:
                return
        if not img_dir:
            img_dir = QFileDialog.getExistingDirectory(self, 'Select Image Directory')
            if not img_dir:
                return

        try:
            with open(coco_path, 'r') as f:
                self.coco_data = json.load(f)
            self.image_dir = img_dir
            self.last_coco_path = coco_path
            self.last_img_dir = img_dir
            self.annotations = sorted(self.coco_data.get('annotations', []), key=lambda ann: int(ann['id']))
            self.images = self.coco_data.get('images', [])
            # Build id-to-index mapping for fast annotation update
            self.ann_id_to_idx = {int(ann['id']): int(idx) for idx, ann in enumerate(self.coco_data['annotations'])}
            # Update spinbox maximum
            self.index_spinbox.setMaximum(len(self.annotations))
            # Resume at latest validated annotation
            if resume and self.validation_status:
                validated_ids = [int(aid) for aid, v in self.validation_status.items() if v.get('validated')]
                if validated_ids:
                    last_validated = max(validated_ids)
                    try:
                        self.current_idx = self.ann_id_to_idx[last_validated]
                    except Exception as e:
                        QMessageBox.warning(self, 'Cache Error', f'Error loading validation status: {str(e)}. Start from beginning.')
                        self.current_idx = 0
                else:
                    self.current_idx = 0
            else:
                self.current_idx = 0
            self.show_current()
            self.save_cache()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error loading COCO file: {str(e)}')
            self.last_coco_path = None
            self.last_img_dir = None

    def get_mask_and_patch(self, ann, buffer_size):
        """
        Given an annotation, extract the image patch and mask for the polygon, with a buffer around the object.

        Args:
            ann (dict): COCO annotation.
            buffer_size (int): Number of pixels to expand around the polygon.

        Returns:
            tuple: (patch, mask_patch, img_meta, (x_min, y_min))
                patch: Cropped image patch (np.ndarray)
                mask_patch: Cropped mask patch (np.ndarray)
                img_meta: Image metadata dict
                (x_min, y_min): Offset of the patch in the original image
        """
        img_meta = next((img for img in self.images if img['id'] == ann['image_id']), None)
        if img_meta is None:
            return None, None, None, None
        img_path = os.path.join(self.image_dir, img_meta['file_name'])
        if not os.path.isfile(img_path):
            return None, None, None, None
        img = cv2.imread(img_path)
        if img is None:
            return None, None, None, None
        h, w = img.shape[:2]
        segm = ann.get('segmentation', [])
        mask = polygon_to_mask(segm, h, w)
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return img, mask, None, img_meta
        x_min = max(0, np.min(x_indices) - buffer_size)
        x_max = min(w-1, np.max(x_indices) + buffer_size)
        y_min = max(0, np.min(y_indices) - buffer_size)
        y_max = min(h-1, np.max(y_indices) + buffer_size)
        patch = img[y_min:y_max, x_min:x_max].copy()
        mask_patch = mask[y_min:y_max, x_min:x_max]
        return patch, mask_patch, img_meta, (x_min, y_min)

    def show_current(self):
        """
        Display the current annotation's image patch and polygon in the scene.
        Updates the info panel with annotation details and validation status.
        """
        self.scene.clear()
        self._polygon_moved = False  # Track if polygon was moved
        if not self.annotations or self.current_idx < 0 or self.current_idx >= len(self.annotations):
            self.info_label.setText('No annotation loaded.')
            return
        ann = self.annotations[self.current_idx]
        patch, mask_patch, img_meta, offset = self.get_mask_and_patch(ann, self.buffer_size)
        self.current_patch = patch
        self.current_mask = mask_patch
        self.current_ann = ann
        self.current_img_meta = img_meta

        if patch is not None:
            # Always fit patch to canvas size
            view_w = self.graphics_view.viewport().width()
            view_h = self.graphics_view.viewport().height()
            h, w = patch.shape[:2]
            if view_w > 0 and view_h > 0:
                scale_x = view_w / w
                scale_y = view_h / h
                disp_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                disp_patch = cv2.resize(disp_patch, (view_w, view_h), interpolation=cv2.INTER_AREA)
                img_pil = QImage(disp_patch.data, view_w, view_h, 3 * view_w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(img_pil)
                self.scene.setSceneRect(0, 0, view_w, view_h)
                self.scene.addPixmap(pixmap)
                # Draw mask polygon
                segm = ann.get('segmentation', [])
                if segm and len(segm[0]) >= 6:
                    arr = np.array(segm[0], dtype=np.float32).reshape(-1, 2)
                    arr[:, 0] -= offset[0]
                    arr[:, 1] -= offset[1]
                    arr[:, 0] *= scale_x
                    arr[:, 1] *= scale_y
                    polygon = QPolygonF([QPointF(x, y) for x, y in arr])
                    self.mask_item = MaskPolygonItem(polygon)
                    self.scene.addItem(self.mask_item)
                # Store for coordinate conversion
                self._last_patch_shape = (h, w)
                self._last_scale = (scale_x, scale_y)
                self._last_offset = offset

        # Info panel: show annotation and validation status
        validated_ids = {aid for aid, v in self.validation_status.items() if v.get('validated') or v.get('removed')}
        removed_ids = {aid for aid, v in self.validation_status.items() if v.get('removed')}
        if str(self.current_ann['id']) in removed_ids:
            current_status = "Removed"
        elif str(self.current_ann['id']) in validated_ids:
            current_status = "Valid"
        else:
            current_status = "Unchecked"
        info_lines = [
            '',
            '',
            f'Annotation {self.current_idx+1}/{len(self.annotations)}',
            f'Annotation ID: {ann.get("id")}',
            f'Current status: {current_status}',
            '',
            f'Validated: {len(validated_ids)}',
            f'Removed: {len(removed_ids)}',
            '',
            f'Area: {ann.get("area")}',
            f'Bbox: {ann.get("bbox")}',
            '',
            f'Object ID: {ann.get("object_id")}',
            f'Image ID: {ann.get("image_id")}',
            f'Category ID: {ann.get("category_id")}',
            f'Image: {self.current_img_meta.get("file_name") if self.current_img_meta else "N/A"}'
        ]
        self.info_label.setText("\n".join(info_lines))


    def remove_current(self):
        """
        Mark the current annotation as removed and advance to the next annotation.
        """
        ann_id = str(self.annotations[self.current_idx]['id'])
        self.validation_status[ann_id] = {'validated': True, 'removed': True}
        self.save_cache()
        self.next_annotation()
        self.show_current()

    def prev_annotation(self):
        """
        Go to the previous annotation in the list.
        """
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current()

    def next_annotation(self):
        """
        Go to the next annotation in the list.
        """
        if self.current_idx < len(self.annotations) - 1:
            self.current_idx += 1
            self.show_current()

    def keep_current(self):
        """
        Mark the current annotation as validated (kept).
        If the polygon was moved, update the annotation's segmentation and bbox.
        """
        # Check if the polygon was moved
        if self.mask_item.pos() != QPointF(0, 0):
            self._polygon_moved = True

        if self._polygon_moved:
            # Save moved polygon to annotation and mark as kept
            polygon = self.mask_item.polygon()
            item_offset = self.mask_item.pos()
            scale_x, scale_y = getattr(self, '_last_scale', (1.0, 1.0))
            offset = getattr(self, '_last_offset', (0, 0))
            coords = []

            # Get image dimensions for clipping
            img_width = self.current_img_meta.get("width", None) if self.current_img_meta else None
            img_height = self.current_img_meta.get("height", None) if self.current_img_meta else None

            for i in range(polygon.count()):
                pt = polygon.at(i)
                x = float(pt.x() + item_offset.x()) / scale_x + offset[0]
                y = float(pt.y() + item_offset.y()) / scale_y + offset[1]
                # Clip coordinates to image bounds if dimensions are available
                if img_width is not None:
                    x = min(max(x, 0), img_width - 1)
                if img_height is not None:
                    y = min(max(y, 0), img_height - 1)
                coords.extend([int(round(x)), int(round(y))])
            self.current_ann['segmentation'] = [coords]
            # Update bbox
            xs = coords[::2]
            ys = coords[1::2]
            if xs and ys:
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                self.current_ann['bbox'] = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

        # Update self.coco_data['annotations'] with the changed annotation using id as index
        ann_id = str(self.current_ann['id'])

        self.validation_status[ann_id] = {'validated': True, 'removed': False}
        self.save_cache()
        self.next_annotation()
        return

            
    def keyPressEvent(self, event):
        """
        Handle keyboard shortcuts for navigation and validation.

        Arrow keys: next/prev annotation
        Delete/Backspace: remove annotation
        Enter: keep annotation
        +/-: scale polygon
        """
        key = event.key()
        # Arrow keys for navigation
        if key == Qt.Key_Right:
            self.next_annotation()
            return
        elif key == Qt.Key_Left:
            self.prev_annotation()
            return
        # Remove: Delete or Backspace
        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.remove_current()
            return
        # Keep: Enter
        elif key == Qt.Key_Return and hasattr(self, 'mask_item'):
            self.keep_current()

        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.scale_polygon(1.1)  # Scale up by 10%
            return
        elif key == Qt.Key_Minus or key == Qt.Key_Underscore:
            self.scale_polygon(0.9)  # Scale down by 10%
            return
        else:
            super().keyPressEvent(event)

    def export_validated(self):
        """
        Export all validated (kept) annotations to a new COCO JSON file.
        """
        if not self.coco_data:
            QMessageBox.warning(self, 'Export', 'No COCO data loaded.')
            return
        export_path, _ = QFileDialog.getSaveFileName(self, 'Export Validated', '', 'JSON Files (*.json)')
        if not export_path:
            return
        valid_ids = {aid for aid, v in self.validation_status.items() if v.get('validated') and not v.get('removed')}
        new_anns = [ann for ann in self.annotations if str(ann['id']) in valid_ids]
        new_coco = dict(self.coco_data)
        new_coco['annotations'] = new_anns
        try:
            with open(export_path, 'w') as f:
                json.dump(new_coco, f)
            QMessageBox.information(self, 'Export', f'Exported {len(new_anns)} annotations to {export_path}')
        except Exception as e:
            QMessageBox.critical(self, 'Export Error', str(e))

    def closeEvent(self, event):
        """
        Handle window close event: save cache and update COCO file with any changes.
        """
        self.save_cache()  # Save validation status before exit
        if self.last_coco_path and self.coco_data:
            try:
                with open(self.last_coco_path, 'w') as f:
                    json.dump(dict(self.coco_data), f)
                print(f"COCO file '{self.last_coco_path}' updated on exit.")
            except Exception as e:
                print(f"Failed to save COCO file on exit: {e}")
        else:
            print("Skipping COCO save on exit: no valid COCO data loaded.")
        event.accept()

    def scale_polygon(self, factor):
        """
        Scale the current polygon about its centroid by the given factor.

        Args:
            factor (float): Scaling factor (>1 to enlarge, <1 to shrink).
        """
        if hasattr(self, 'mask_item'):
            polygon = self.mask_item.polygon()
            centroid = QPointF(
                sum(p.x() for p in polygon) / polygon.count(),
                sum(p.y() for p in polygon) / polygon.count()
            )
            new_poly = QPolygonF([
                QPointF(
                    centroid.x() + (p.x() - centroid.x()) * factor,
                    centroid.y() + (p.y() - centroid.y()) * factor
                ) for p in polygon
            ])
            self.mask_item.setPolygon(new_poly)
            self._polygon_moved = True

    def next_unvalidated(self):
        """
        Go to the next unvalidated annotation in the list.
        Wraps around to the beginning if needed.
        """
        for idx in range(self.current_idx + 1, len(self.annotations)):
            ann_id = str(self.annotations[idx]['id'])
            v = self.validation_status.get(ann_id, {})
            if not (v.get('validated') or v.get('removed')):
                self.current_idx = idx
                self.show_current()
                return
        # If not found, wrap around
        for idx in range(0, self.current_idx):
            ann_id = str(self.annotations[idx]['id'])
            v = self.validation_status.get(ann_id, {})
            if not (v.get('validated') or v.get('removed')):
                self.current_idx = idx
                self.show_current()
                return

    def prev_unvalidated(self):
        """
        Go to the previous unvalidated annotation with largest id < current.
        """
        prev_idx = None
        prev_id = None
        for idx in range(0, self.current_idx):
            ann_id = str(self.annotations[idx]['id'])
            v = self.validation_status.get(ann_id, {})
            if not (v.get('validated') or v.get('removed')):
                if prev_id is None or int(ann_id) > prev_id:
                    prev_id = int(ann_id)
                    prev_idx = idx
        if prev_idx is not None:
            self.current_idx = prev_idx
            self.show_current()
            return

    def increase_buffer(self):
        """
        Increase the buffer size (zoom out) for the image patch display.
        """
        self.buffer_size = min(1000, self.buffer_size + 100)
        self.save_cache()
        self.show_current()

    def decrease_buffer(self):
        """
        Decrease the buffer size (zoom in) for the image patch display.
        """
        self.buffer_size = max(0, self.buffer_size - 100)
        self.save_cache()
        self.show_current()

    def go_to_annotation(self, index):
        """
        Jump to a specific annotation by index (1-based from spinbox).

        Args:
            index (int): 1-based annotation index.
        """
        if 1 <= index <= len(self.annotations):
            self.current_idx = index - 1
            self.show_current()

if __name__ == '__main__':
    # Entry point: start the PyQt application and show the validator window
    app = QApplication(sys.argv)
    win = COCOValidatorWindow()
    win.show()
    sys.exit(app.exec_())


