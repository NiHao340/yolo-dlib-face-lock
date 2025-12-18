# -*- coding: utf-8 -*-
import dlib
import numpy as np
from config import DLIB_PREDICTOR, DLIB_RECOG_MODEL

class DlibFace:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.detector = dlib.get_frontal_face_detector()
            cls._instance.sp = dlib.shape_predictor(DLIB_PREDICTOR)
            cls._instance.model = dlib.face_recognition_model_v1(DLIB_RECOG_MODEL)
        return cls._instance

    def embedding(self, rgb, box):
        x1, y1, x2, y2 = box
        rect = dlib.rectangle(x1, y1, x2, y2)
        shape = self.sp(rgb, rect)
        return np.array(self.model.compute_face_descriptor(rgb, shape))
