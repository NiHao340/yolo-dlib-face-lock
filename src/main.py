# -*- coding: utf-8 -*-
import sys
import cv2
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor

from dlib_face import DlibFace
from worker import FaceWorker
from config import DISPLAY_INTERVAL, DETECT_INTERVAL

class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸锁定（YOLO + dlib 稳定版）")
        self.resize(900, 600)

        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(255, 255, 255))
        self.setPalette(pal)

        self.video_label = QLabel("视频预览")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#eee;border:2px solid #ccc")

        btn_style = "background:#ffd700;padding:10px;font-size:16px;border-radius:8px;"

        self.btn_face = QPushButton("选择参考脸")
        self.btn_face.setStyleSheet(btn_style)
        self.btn_face.clicked.connect(self.load_face)

        self.btn_video = QPushButton("选择视频")
        self.btn_video.setStyleSheet(btn_style)
        self.btn_video.clicked.connect(self.load_video)

        self.btn_start = QPushButton("开始处理")
        self.btn_start.setStyleSheet(btn_style)
        self.btn_start.clicked.connect(self.start)

        self.save_check = QCheckBox("保存视频")
        self.save_check.setChecked(True)

        h = QHBoxLayout()
        for b in [self.btn_face, self.btn_video, self.btn_start, self.save_check]:
            h.addWidget(b)

        v = QVBoxLayout()
        v.addWidget(self.video_label)
        v.addLayout(h)

        w = QWidget()
        w.setLayout(v)
        self.setCentralWidget(w)

        self.ref_emb = None
        self.video_path = None
        self.cap = None
        self.yolo = None
        self.worker = None
        self.last_box = None
        self.frame_id = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.loop)

    def load_face(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择人脸")
        if not path:
            return

        img = cv2.imread(path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = DlibFace().detector(rgb, 1)

        if not dets:
            QMessageBox.warning(self, "错误", "未检测到人脸")
            return

        self.ref_emb = DlibFace().embedding(
            rgb,
            (dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom())
        )

        QMessageBox.information(self, "OK", "参考人脸加载成功")

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频")
        if not path:
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "错误", "视频无法打开（路径或编码问题）")
            return

        ret, frame = cap.read()
        cap.release()
        if not ret:
            QMessageBox.critical(self, "错误", "无法读取视频首帧")
            return

        self.show_frame(frame)
        self.video_path = path

    def start(self):
        if self.ref_emb is None or not self.video_path:
            QMessageBox.warning(self, "错误", "请先选人脸和视频")
            return

        model_path, _ = QFileDialog.getOpenFileName(self, "选择YOLO模型", "", "*.pt")
        if not model_path:
            QMessageBox.warning(self, "错误", "未选择 YOLO 模型")
            return

        self.yolo = YOLO(model_path)
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "视频打开失败")
            return

        self.worker = FaceWorker(self.ref_emb)
        self.worker.result.connect(self.update_box)
        self.worker.start()

        self.timer.start(30)

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.worker.stop()
            return

        self.frame_id += 1

        if self.frame_id % DETECT_INTERVAL == 0:
            results = self.yolo(frame, classes=[0], verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
            self.worker.set_data(frame.copy(), boxes)

        if self.last_box:
            x1, y1, x2, y2 = self.last_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if self.frame_id % DISPLAY_INTERVAL == 0:
            self.show_frame(frame)

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pix)

    def update_box(self, box):
        self.last_box = box

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop()
        e.accept()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    ui = MainUI()
    ui.show()
    sys.exit(app.exec_())
