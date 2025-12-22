## dlib 预训练模型下载

本项目使用 dlib 官方提供的人脸模型，由于文件较大，未直接上传至 GitHub。

请手动下载并放置到 `models/` 目录下：

### 1. 人脸关键点模型
- 文件名：`shape_predictor_68_face_landmarks.dat`
- 下载地址：http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

### 2. 人脸识别模型
- 文件名：`dlib_face_recognition_resnet_model_v1.dat`
- 下载地址：http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

下载后请解压，并确保目录结构如下：

```text
models/
├── shape_predictor_68_face_landmarks.dat
└── dlib_face_recognition_resnet_model_v1.dat
