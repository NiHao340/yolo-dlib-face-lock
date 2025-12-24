# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QCheckBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor

# 导入自定义模块
from dlib_face import DlibFace
from worker import FaceWorker
from config import DISPLAY_INTERVAL, DETECT_INTERVAL

class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸锁定跟踪系统（YOLO + dlib）")
        self.resize(1000, 700)

        # 设置背景颜色
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(245, 245, 245))
        self.setPalette(pal)

        # --- 核心修复：防止界面变大的视频展示区 ---
        self.video_label = QLabel("等待视频输入...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#2c3e50; border:2px solid #34495e; color:white; font-size:20px;")
        
        # 关键代码：设置尺寸策略，防止 QLabel 被图片撑大
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # --- 按钮样式 ---
        btn_style = "QPushButton { background:#ffd700; padding:12px; font-size:14px; border-radius:6px; font-weight:bold; } " \
                    "QPushButton:hover { background:#ffcc00; }"

        self.btn_face = QPushButton("1. 选择参考脸")
        self.btn_face.setStyleSheet(btn_style)
        self.btn_face.clicked.connect(self.load_face)

        self.btn_video = QPushButton("2. 选择视频文件")
        self.btn_video.setStyleSheet(btn_style)
        self.btn_video.clicked.connect(self.load_video)

        self.btn_cam = QPushButton("或: 开启摄像头")
        self.btn_cam.setStyleSheet("QPushButton { background:#3498db; color:white; padding:12px; border-radius:6px; font-weight:bold; }")
        self.btn_cam.clicked.connect(self.use_camera)

        self.btn_start = QPushButton("开始锁定检测")
        self.btn_start.setStyleSheet("QPushButton { background:#2ecc71; color:white; padding:12px; border-radius:6px; font-weight:bold; font-size:16px; }")
        self.btn_start.clicked.connect(self.start)

        self.save_check = QCheckBox("保存处理视频")
        self.save_check.setChecked(False)

        # --- 布局管理 ---
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.btn_face)
        h_layout.addWidget(self.btn_video)
        h_layout.addWidget(self.btn_cam)
        h_layout.addStretch()
        h_layout.addWidget(self.save_check)
        h_layout.addWidget(self.btn_start)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.video_label, stretch=1) # 视频区域占据主要空间
        v_layout.addLayout(h_layout)

        central_widget = QWidget()
        central_widget.setLayout(v_layout)
        self.setCentralWidget(central_widget)

        # --- 状态变量 ---
        self.ref_emb = None      # 基准人脸特征
        self.video_path = None   # 视频路径或摄像头ID
        self.cap = None          # 视频捕获对象
        self.yolo = None         # YOLO模型对象
        self.worker = None       # 后台比对线程
        self.last_box = None     # 最近一次匹配成功的坐标
        self.frame_id = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.loop)

    def load_face(self):
        """加载参考人脸图片并提取特征"""
        path, _ = QFileDialog.getOpenFileName(self, "选择一张清晰的人脸照片")
        if not path: return

        # 使用 numpy 读取以支持中文路径
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "错误", "图片读取失败，请检查路径。")
            return

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = DlibFace().detector(rgb, 1) # 使用 dlib 检测

        if not dets:
            QMessageBox.warning(self, "检测失败", "未在照片中识别出人脸。")
            return

        # 获取特征向量
        self.ref_emb = DlibFace().embedding(
            rgb, (dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom())
        )
        QMessageBox.information(self, "成功", "目标人脸特征已锁定。")

    def load_video(self):
        """选择视频文件"""
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "*.mp4 *.avi *.mkv")
        if path:
            self.video_path = path
            QMessageBox.information(self, "就绪", f"已选择视频: {path.split('/')[-1]}")

    def use_camera(self):
        """切换到实时摄像头模式"""
        self.video_path = 0 
        QMessageBox.information(self, "模式切换", "已切换至摄像头模式，点击“开始”即可实时检测。")

    def start(self):
        """初始化 YOLO 模型并开始运行循环"""
        if self.ref_emb is None:
            QMessageBox.warning(self, "提示", "请先选择参考人脸。")
            return
        if self.video_path is None:
            QMessageBox.warning(self, "提示", "请选择视频或开启摄像头。")
            return

        # 加载模型
        model_path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 模型", "models", "*.pt")
        if not model_path: return
        
        try:
            self.yolo = YOLO(model_path)
            self.cap = cv2.VideoCapture(self.video_path)
            
            # 开启 worker 线程进行人脸比对
            self.worker = FaceWorker(self.ref_emb)
            self.worker.result.connect(self.update_box)
            self.worker.start()

            self.timer.start(30) # 约 33 帧/秒
        except Exception as e:
            QMessageBox.critical(self, "启动失败", f"错误详情: {str(e)}")

    def loop(self):
        """主循环：读取帧 -> 检测 -> 绘制"""
        ret, frame = self.cap.read()
        if not ret:
            self.stop_all()
            return

        self.frame_id += 1

        # 按照配置的间隔进行 YOLO 检测
        if self.frame_id % DETECT_INTERVAL == 0:
            results = self.yolo(frame, classes=[0], verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
            self.worker.set_data(frame.copy(), boxes)

        # 绘制锁定框（绿色）
        if self.last_box:
            x1, y1, x2, y2 = map(int, self.last_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, "TARGET", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 按照显示间隔更新 UI
        if self.frame_id % DISPLAY_INTERVAL == 0:
            self.show_frame(frame)

    def show_frame(self, frame):
        """转换图像并在界面显示"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        
        # 缩放图像以适应 Label，保持比例
        pix = QPixmap.fromImage(img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def update_box(self, box):
        """接收 worker 线程返回的比对结果"""
        self.last_box = box

    def stop_all(self):
        self.timer.stop()
        if self.worker: self.worker.stop()
        if self.cap: self.cap.release()
        self.video_label.setText("视频处理已结束")

    def closeEvent(self, e):
        self.stop_all()
        e.accept()

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    ui = MainUI()
    ui.show()
    sys.exit(app.exec_())

