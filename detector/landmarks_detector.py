"""
Code modified from https://github.com/1adrianb/face-alignment
"""
import cv2
import numpy as np
from keras.models import load_model

class BaseLandmarksDetector():
    def __init__(self):
        raise NotImplementedError()
        
    def detect_landmarks(self):
        raise NotImplementedError()

class FANLandmarksDetector(BaseLandmarksDetector):
    def __init__(self, path_to_weights_file="./detector/FAN/2DFAN-4_keras.h5"):
        self.net = load_model(path_to_weights_file)

    def detect_landmarks(self, image, bounding_box=None, face_detector=None):
        if (bounding_box is None) and (face_detector is None):
            raise ValueError("Neither bounding box or face detector is passed in.")
        detect = True if (bounding_box is None) else False
        prep_img, center, scale = self._preprocessing_FAN(image, detect=detect, face_detector=face_detector, bbox=bounding_box)
        pred = self.net.predict(prep_img[np.newaxis, ...])
        pnts, pnts_orig = self._get_preds_fromhm(pred[-1], center, scale)
        return pnts, pnts_orig
    
    def _preprocessing_FAN(self, img, detect=False, face_detector=None, bbox=None):
        """
        Preprocess single RGB input image to proper format as following:
            0. Detect face
            1. Resize and crop.
            2. Transform from HWC to CHW ordering.
            3. Normalize to [0,1] (devide by 255).
        """

        if img.ndim == 2:
            img = np.stack([img[..., np.newaxis], img[..., np.newaxis], img[..., np.newaxis]], axis=-1)
        elif img.ndim == 4:
            img = img[..., :3]

        # Detection
        if detect:
            try:
                assert face_detector
            except:
                AssertionError(f"face_detector has not been specified. face_detect is [{face_detector}]")
            bbox = face_detector.detect_face(img)[0]
            x0, x1, y0, y1 = int(bbox[1]), int(bbox[3]), int(bbox[0]), int(bbox[2])
        else:
            x0, x1, y0, y1 = int(bbox[1]), int(bbox[3]), int(bbox[0]), int(bbox[2])    

        # Compute center and scale
        center = np.array([(y0 + y1) / 2, (x0 + x1) / 2], np.float32)
        center[1] = center[1] - (x1 - x0) * 0.12
        # The number 195 is hard coded in 
        # https://github.com/1adrianb/face-alignment/blob/master/face_alignment/detection/sfd/sfd_detector.py
        scale = (x1 - x0 + y1 - y0) / 195

        # Resizing and cropping
        img = self._crop(img, center, scale)

        # HWC to CHW
        img = img.transpose(2,0,1)

        # Normalization
        img = img / 255   

        return img, center, scale
    
    def _crop(self, image, center, scale, resolution=256.0):
        """Center crops an image or set of heatmaps
        Arguments:
            image {numpy.array} -- an rgb image
            center {numpy.array} -- the center of the object, usually the same as of the bounding box
            scale {float} -- scale of the face
        Keyword Arguments:
            resolution {float} -- the size of the output cropped image (default: {256.0})
        Returns:
            [type] -- [description]
        """  # Crop around the center point
        """ Crops the image around the center. Input is expected to be an np.ndarray """
        ul = self._transform([1, 1], center, scale, resolution, True)
        br = self._transform([resolution, resolution], center, scale, resolution, True)
        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                               image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array(
            [np.max((1, -ul[0] + 1)), np.min((br[0], wd)) - ul[0]], dtype=np.int32)
        newY = np.array(
            [np.max((1, -ul[1] + 1)), np.min((br[1], ht)) - ul[1]], dtype=np.int32)
        oldX = np.array([np.max((1, ul[0] + 1)), np.min((br[0], wd))], dtype=np.int32)
        oldY = np.array([np.max((1, ul[1] + 1)), np.min((br[1], ht))], dtype=np.int32)
        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
               ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                            interpolation=cv2.INTER_LINEAR)
        return newImg

    def _transform(self, point, center, scale, resolution, invert=False):
        """Generate and affine transformation matrix.
        Given a set of points, a center, a scale and a targer resolution, the
        function generates and affine transformation matrix. If invert is ``True``
        it will produce the inverse transformation.
        Arguments:
            point {torch.tensor} -- the input 2D point
            center {torch.tensor or numpy.array} -- the center around which to perform the transformations
            scale {float} -- the scale of the face/object
            resolution {float} -- the output resolution
        Keyword Arguments:
            invert {bool} -- define wherever the function should produce the direct or the
            inverse transformation matrix (default: {False})
        """
        _pt = np.ones(3)
        _pt[0] = point[0]
        _pt[1] = point[1]

        h = 200.0 * scale
        t = np.eye(3)
        t[0, 0] = resolution / h
        t[1, 1] = resolution / h
        t[0, 2] = resolution * (-center[0] / h + 0.5)
        t[1, 2] = resolution * (-center[1] / h + 0.5)

        if invert:
            t = np.linalg.inv(t)

        new_point = (np.matmul(t, _pt))[0:2].astype(np.int32)

        return new_point
    
    def _get_preds_fromhm(self, hm, center=None, scale=None):
        """Obtain (x,y) coordinates given a set of N heatmaps. If the center
        and the scale is provided the function will return the points also in
        the original coordinate frame.
        Arguments:
            hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]
        Keyword Arguments:
            center {torch.tensor} -- the center of the bounding box (default: {None})
            scale {float} -- face scale (default: {None})
        """
        assert hm.ndim == 4, f"Receive hm in unexpected dimension: {hm.ndim}."
        hm_flat = hm.reshape((hm.shape[1], hm.shape[2] * hm.shape[3]))
        idx = np.argmax(hm_flat, axis=-1)[np.newaxis, ...]
        idx += 1
        preds = np.repeat(idx.reshape(idx.shape[0], idx.shape[1], 1), 2, 2).astype(np.float32)
        preds[..., 0] = (preds[..., 0] - 1) % hm.shape[3] + 1
        preds[..., 1] = np.floor((preds[..., 1] - 1) / hm.shape[2]) + 1

        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                hm_ = hm[i, j, :]
                pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
                if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                    diff = np.array(
                        [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                         hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                    preds[i, j] = preds[i, j] + (np.sign(diff) * 0.25)

        preds = preds - 0.5

        preds_orig = np.zeros_like(preds)
        if center is not None and scale is not None:
            for i in range(hm.shape[0]):
                for j in range(hm.shape[1]):
                    preds_orig[i, j] = self._transform(
                        preds[i, j], center, scale, hm.shape[2], True)

        preds_orig = [np.array(p) for p in preds_orig[0]]
        return preds, preds_orig
    
    @staticmethod
    def draw_landmarks (image, landmarks, color):        
        for i in range(len(landmarks)): 
            x, y = landmarks[i]
            image = cv2.circle(image.copy(), (int(x), int(y)), 3, color, -1)        
        return image