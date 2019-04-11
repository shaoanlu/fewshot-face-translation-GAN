import numpy as np
import scipy
from .model import s3fd_keras

class S3FD():
    def __init__(self, weights_path="./detector/s3fd/s3fd_keras_weights.h5"):
        self.net = s3fd_keras()
        self.net.load_weights(weights_path)
    
    def detect_face(self, image):
        bboxlist = self.detect(self.net, image)
        keep = self.nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]
        return bboxlist
    
    def detect(self, net, img):    
        def softmax(x, axis=-1):
            return np.exp(x - scipy.special.logsumexp(x, axis=axis, keepdims=True))
        img = img - np.array([104, 117, 123])
        if img.ndim == 3:
            img = img[np.newaxis, ...]
        elif img.ndim == 5:
            img = np.squeeze(img)

        BB, HH, WW, CC = img.shape
        olist = net.predict(img) # output a list of 12 predicitons in different resolution

        bboxlist = []
        for i in range(len(olist) // 2):
            olist[i * 2] = softmax(olist[i * 2], axis=-1)
        olist = [oelem for oelem in olist]
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            FB, FH, FW, FC = ocls.shape  # feature map size
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            anchor = stride * 4
            poss = zip(*np.where(ocls[:, :, :, 1] > 0.05))
            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0:1, hindex, windex, 1]
                loc = oreg[0:1, hindex, windex, :]
                priors = np.array([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = self.decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                bboxlist.append([x1, y1, x2, y2, score])
        bboxlist = np.array(bboxlist)
        if 0 == len(bboxlist):
            bboxlist = np.zeros((1, 5))            
        return bboxlist

    @staticmethod
    def decode(loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded bounding box predictions
        """
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @staticmethod
    def nms(dets, thresh):
        if 0 == len(dets):
            return []
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
            ovr = w * h / (areas[i] + areas[order[1:]] - w * h)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep