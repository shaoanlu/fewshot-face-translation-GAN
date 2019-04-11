import numpy as np
import tensorflow as tf
from keras import backend as K

from .s3fd.s3fd_detector import S3FD
from .landmarks_detector import FANLandmarksDetector

class BaseFaceDetector():
    def __init(self):
        pass
    
    def detect_face(self):
        raise NotImplementedError
    
class S3FaceDetector(BaseFaceDetector):
    def __init__(self, weights_path="./detector/s3fd/s3fd_keras_weights.h5"):
        self.face_detector = S3FD(weights_path)
        
    def detect_face(self, image):
        # output bbox coordinate: y0 (left), x0 (top), y1 (right), x1 (bottom)
        return self.face_detector.detect_face(image)
        
    def batch_detect_face(self, image):
        raise NotImplementedError
    
class FaceAlignmentDetector(BaseFaceDetector):
    def __init__(self, 
                 fd_weights_path="./detector/s3fd/s3fd_keras_weights.h5", 
                 lmd_weights_path="./detector/FAN/2DFAN-4_keras.h5",
                 fd_type="s3fd"):
        self.fd_type = fd_type.lower()
        if fd_type.lower() == "s3fd":
            self.fd = S3FaceDetector(fd_weights_path)
        elif fd_type.lower() == "mtcnn":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown face detector {face_detector}.")
        self.lmd = FANLandmarksDetector(lmd_weights_path)
        
    def detect_face(self, image, with_landmark=True, **kwargs):
        if self.fd_type == "s3fd":
            bbox_list = self.fd.detect_face(image)
        elif self.fd_type == "mtcnn":
            bbox_list, _ = self.fd.detect_face(image)
        if len(bbox_list) == 0:
            return [], []
        landmarks_list = []
        im_shape = image.shape
        if self.fd_type == "mtcnn":
            bbox_list = self.preprocess_mtcnn_bbox(bbox_list)
        if with_landmark:
            for bbox in bbox_list:
                pnts = self.lmd.detect_landmarks(image, bounding_box=bbox)[-1]
                landmarks_list.append(np.array(pnts))
            landmarks_list = [self.post_process_landmarks(landmarks) for landmarks in landmarks_list]
            landmarks_list = np.concatenate(landmarks_list, axis=-1)
        bbox_list = [self.post_process_bbox(bbox, im_shape) for bbox in bbox_list]
        bbox_list = [bbox for bbox in bbox_list if self.compute_face_area(bbox) >= kwargs["min_face_area"]]
        return bbox_list, landmarks_list
    
    def batch_detect_face(self, images, **kwargs):
        raise NotImplementedError
        
    @staticmethod
    def post_process_bbox(bbox, im_shape):
        # video converter expect bbox coord. in [x0, y1, x1, y0, score] ordering
        y0, x0, y1, x1, score = bbox        
        w = int(y1 - y0)
        h = int(x1 - x0)
        length = (w + h)/2
        center = (int((x1+x0)/2),int((y1+y0)/2))
        new_x0 = np.max([0, (center[0]-length//2)])#.astype(np.int32)
        new_x1 = np.min([im_shape[0], (center[0]+length//2)])#.astype(np.int32)
        new_y0 = np.max([0, (center[1]-length//2)])#.astype(np.int32)
        new_y1 = np.min([im_shape[1], (center[1]+length//2)])#.astype(np.int32)
        bbox = np.array([new_x0, new_y1, new_x1, new_y0, score])
        return bbox
    
    @staticmethod
    def preprocess_mtcnn_bbox(bbox_list):
        def bbox_coord_convert(bbox, cnovert_type="mtcnn_to_s3fd"):
            if cnovert_type == "mtcnn_to_s3fd":
                # x0y1x1y0 to y0x0y1x1
                x0, y1, x1, y0, score = bbox
                return np.array([y0, x0, y1, x1, score])
            
        for i, bbox in enumerate(bbox_list):
            if len(bbox.shape) == 2:
                bbox = bbox[0]
            bbox_list[i] = bbox_coord_convert(bbox, cnovert_type="mtcnn_to_s3fd")
        return bbox_list
        
    @staticmethod
    def post_process_landmarks(landmarks):
        # video converter expect landmarks having shape [xy, pnts]
        lms = landmarks.reshape(landmarks.shape[0] * landmarks.shape[1], 1)
        if lms.ndim == 1:
            lms = lms[:, np.newaxis]
        return lms
    
    @staticmethod
    def compute_face_area(bbox):
        x0, y1, x1, y0, _ = bbox
        area = (x1 - x0) * (y1 - y0)
        return area
        
            

    