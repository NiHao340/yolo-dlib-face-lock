# yolo-dlib-face-lock

YOLO + dlib 的人脸识别 / 人脸解锁系统（PyQt5 稳定版）

---

## 📌 项目简介

本项目基于 **YOLOv8 + dlib** 实现实时人脸锁定与人脸识别，并通过 **PyQt5** 构建桌面图形界面，
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
│   ├── main.py            # 程序主入口 (PyQt5 界面启动)
│   ├── dlib_face.py       # 人脸识别逻辑 (dlib 特征提取)
│   ├── config.py          # 项目配置文件 (路径、阈值等)
│   └── worker.py          # 多线程处理逻辑 (解耦界面与算法)
│
├── models/                # 模型文件
│   ├── README.md          # 模型下载说明
│   └── yolov8n.pt         # YOLOv8 权重文件
│
└── data/                  # 测试数据 / 用户人脸库数据
```

---

## 🧪 运行环境

- Python 3.8+
- Windows
- 推荐使用虚拟环境（venv / conda）

---

## 📦 安装依赖
- pip install -r requirements.txt

---

## 📥 模型下载说明

- 由于模型文件体积较大，不直接上传到 GitHub。
- 请按照以下说明下载并放入 models/ 目录：
- dlib 人脸识别模型
- shape_predictor_68_face_landmarks.dat
- dlib_face_recognition_resnet_model_v1.dat
- 👉 具体下载地址与说明见：
- models/README.md

---

## ▶️ 运行项目
- python src/main.py
- 启动后即可通过图形界面进行人脸识别操作。

---

## ⚠️ 注意事项
- 首次运行请确认模型路径正确
- 模型文件未放置会导致程序启动失败

---

## 📄 免责声明
- 本项目仅用于学习与研究目的，请勿用于任何非法或侵犯隐私的场景。
