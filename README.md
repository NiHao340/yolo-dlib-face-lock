# yolo-dlib-face-lock

YOLO + dlib 的人脸识别 / 人脸解锁系统（PyQt5 稳定版）

---

## 📌 项目简介

本项目基于 **YOLOv8 + dlib** 实现实时人脸检测与人脸识别，并通过 **PyQt5** 构建桌面图形界面，
可用于人脸身份验证、人脸解锁等场景。

项目特点：

- 🚀 YOLOv8 实时人脸检测
- 🧠 dlib ResNet 人脸特征提取
- 🖥 PyQt5 图形界面，操作直观
- 📂 模块化代码结构，便于扩展

---

## 🧱 项目结构

```text
yolo-dlib-face-lock/
├── README.md              # 项目说明（本文件）
├── requirements.txt       # 依赖环境
│
├── src/                   # 核心代码
│   ├── main.py            # 程序入口
│   ├── face_detect.py     # 人脸检测（YOLO）
│   ├── face_recognize.py  # 人脸识别（dlib）
│   └── ui.py              # PyQt 界面
│
├── models/                # 模型文件（不直接上传大模型）
│   ├── README.md          # 模型下载说明
│   └── yolov8n.pt
│
└── data/                  # 测试数据 / 用户数据
