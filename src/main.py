# -*- coding: utf-8 -*-
import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QInputDialog,
    QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QCheckBox, QSizePolicy, QDialog, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor

# 导入项目中已有的自定义模块
from dlib_face import DlibFace
from worker import FaceWorker
from config import DISPLAY_INTERVAL, DETECT_INTERVAL

# ==========================================================
# 1. 登录对话框类：处理密码解锁
# ==========================================================
class LoginDialog(QDialog):
    def __init__(self, pwd_path="data/pwd.txt"):
        super().__init__()
        self.pwd_path = pwd_path
        self.setWindowTitle("安全验证 - 系统锁定中")
        self.setFixedSize(350, 180)
        self.setStyleSheet("background-color: #f5f5f5;")

        # 确保数据目录存在并初始化默认密码
        if not os.path.exists("data"): os.makedirs("data")
        if not os.path.exists(self.pwd_path):
            with open(self.pwd_path, "w") as f: f.write("123456") # 默认初始密码

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("请输入解锁密码")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)

        self.pwd_input = QLineEdit()
        self.pwd_input.setPlaceholderText("初始密码为 123456")
        self.pwd_input.setEchoMode(QLineEdit.Password)
        self.pwd_input.setStyleSheet("padding: 8px; border: 1px solid #ccc; border-radius: 4px;")

        self.btn_login = QPushButton("验证并进入")
        self.btn_login.setStyleSheet("background: #2ecc71; color: white; padding: 10px; font-weight: bold;")
        self.btn_login.clicked.connect(self.check_password)

        layout.addWidget(title)
        layout.addWidget(self.pwd_input)
        layout.addWidget(self.btn_login)
        self.setLayout(layout)

    def check_password(self):
        # 读取存储的密码
        with open(self.pwd_path, "r") as f:
            saved_pwd = f.read().strip()
        
        if self.pwd_input.text() == saved_pwd:
            self.accept() # 关闭对话框并返回成功状态
        else:
            QMessageBox.warning(self, "错误", "解锁密码不正确，请重新输入")

# ==========================================================
# 2. 主界面类：包含人脸识别、摄像头、密码修改逻辑
# ==========================================================
class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸追踪锁定系统 (YOLOv8 + dlib)")
        self.resize(1100, 750)

        # 视频渲染区域
        self.video_label = QLabel("等待视频流加载...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#1a1a1a; color:#555; border:2px solid #333;")
        # 核心修复：防止图片撑大导致窗口无限变大
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # UI 按钮布局
        btn_style = "QPushButton { background:#ffd700; padding:12px; font-weight:bold; border-radius:5px; } " \
                    "QPushButton:hover { background:#ffcc00; }"

        self.btn_face = QPushButton("1. 选择参考脸")
        self.btn_face.setStyleSheet(btn_style)
        self.btn_face.clicked.connect(self.load_face)

        self.btn_video = QPushButton("2. 选择视频")
        self.btn_video.setStyleSheet(btn_style)
        self.btn_video.clicked.connect(self.load_video)

        self.btn_cam = QPushButton("或：开启摄像头")
        self.btn_cam.setStyleSheet("QPushButton { background:#3498db; color:white; padding:12px; border-radius:5px; font-weight:bold; }")
        self.btn_cam.clicked.connect(self.use_camera)

        self.btn_start = QPushButton("开始锁定检测")
        self.btn_start.setStyleSheet("QPushButton { background:#2ecc71; color:white; padding:12px; border-radius:5px; font-weight:bold; }")
        self.btn_start.clicked.connect(self.start)

        # 密码管理按钮
        self.btn_set_pwd = QPushButton("修改密码")
        self.btn_set_pwd.setStyleSheet("background:#95a5a6; color:white; padding:10px; border-radius:5px;")
        self.btn_set_pwd.clicked.connect(self.change_password)

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.btn_face)
        h_layout.addWidget(self.btn_video)
        h_layout.addWidget(self.btn_cam)
        h_layout.addStretch()
        h_layout.addWidget(self.btn_set_pwd)
        h_layout.addWidget(self.btn_start)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.video_label, stretch=1)
        v_layout.addLayout(h_layout)

        container = QWidget()
        container.setLayout(v_layout)
        self.setCentralWidget(container)

        # 变量初始化
        self.ref_emb = None
        self.video_source = None
        self.cap = None
        self.yolo = None
        self.worker = None
        self.last_box = None
        self.frame_id = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.loop)

    def change_password(self):
        """修改进入密码的功能"""
        new_pwd, ok = QInputDialog.getText(self, "修改安全密码", "请输入新密码:", QLineEdit.Password)
        if ok and new_pwd:
            confirm, ok2 = QInputDialog.getText(self, "确认密码", "请再次输入新密码:", QLineEdit.Password)
            if ok2 and new_pwd == confirm:
                with open("data/pwd.txt", "w") as f:
                    f.write(new_pwd)
                QMessageBox.information(self, "成功", "密码已更新，下次启动生效")
            else:
                QMessageBox.warning(self, "失败", "两次密码输入不一致")

    def load_face(self):
        """提取参考人脸特征"""
        path, _ = QFileDialog.getOpenFileName(self, "选择一张目标照片")
        if not path: return
        
        # 兼容中文路径的读取
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: return

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = DlibFace().detector(rgb, 1)
        if not dets:
            QMessageBox.warning(self, "错误", "未能在该照片中识别到人脸")
            return
        
        self.ref_emb = DlibFace().embedding(rgb, (dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom()))
        QMessageBox.information(self, "锁定成功", "已成功提取目标人脸特征")

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择本地视频文件")
        if path:
            self.video_source = path
            QMessageBox.information(self, "就绪", f"已选择视频：{os.path.basename(path)}")

    def use_camera(self):
        """切换至摄像头模式"""
        self.video_source = 0
        QMessageBox.information(self, "模式切换", "已成功切换至实时摄像头")

    def start(self):
        """启动 YOLO 线程与主循环"""
        if self.ref_emb is None:
            QMessageBox.warning(self, "提示", "请先提取参考人脸特征")
            return
        if self.video_source is None:
            QMessageBox.warning(self, "提示", "请选择视频或开启摄像头")
            return

        model_path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 权重", "models", "*.pt")
        if not model_path: return

        self.yolo = YOLO(model_path)
        self.cap = cv2.VideoCapture(self.video_source)
        
        # 开启异步人脸比对线程
        self.worker = FaceWorker(self.ref_emb)
        self.worker.result.connect(self.update_box)
        self.worker.start()
        
        self.timer.start(30)

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_all()
            return

        self.frame_id += 1
        # 根据 config 配置的频率进行 YOLO 检测
        if self.frame_id % DETECT_INTERVAL == 0:
            results = self.yolo(frame, classes=[0], verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
            self.worker.set_data(frame.copy(), boxes)

        # 渲染绿色锁定框
        if self.last_box:
            x1, y1, x2, y2 = map(int, self.last_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, "LOCKED TARGET", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 根据配置频率刷新 UI
        if self.frame_id % DISPLAY_INTERVAL == 0:
            self.show_frame(frame)

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        # 适应 Label 大小且不破坏比例
        pix = QPixmap.fromImage(qt_img).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def update_box(self, box):
        self.last_box = box

    def stop_all(self):
        self.timer.stop()
        if self.cap: self.cap.release()
        if self.worker: self.worker.stop()
        self.video_label.setText("任务已终止")

    def closeEvent(self, e):
        self.stop_all()
        e.accept()

# ==========================================================
# 3. 程序入口逻辑
# ==========================================================
if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    
    # 步骤1：显示登录弹窗
    login = LoginDialog()
    if login.exec_() == QDialog.Accepted:
        # 步骤2：登录成功才进入主界面
        ui = MainUI()
        ui.show()
        sys.exit(app.exec_())
    else:
        # 验证取消或失败则退出
        sys.exit(0)

