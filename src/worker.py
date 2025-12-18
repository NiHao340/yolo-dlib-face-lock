# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from dlib_face import DlibFace
from config import FACE_THRESHOLD

class FaceWorker(QThread):
    result = pyqtSignal(object)

    def __init__(self, ref_emb):
        super().__init__()
        self.ref_emb = ref_emb
        self.frame = None
        self.boxes = None
        self.running = True

    def set_data(self, frame, boxes):
        self.frame = frame
        self.boxes = boxes

    def run(self):
        dlib_face = DlibFace()
        while self.running:
            if self.frame is None or self.boxes is None:
                time.sleep(0.005)
                continue

            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            target_box = None

            for box in self.boxes:
                x1, y1, x2, y2 = map(int, box)
                try:
                    emb = dlib_face.embedding(rgb, (x1, y1, x2, y2))
                    if np.linalg.norm(emb - self.ref_emb) < FACE_THRESHOLD:
                        target_box = (x1, y1, x2, y2)
                        break
                except:
                    continue

            self.result.emit(target_box)
            self.frame = None

    def stop(self):
        self.running = False
