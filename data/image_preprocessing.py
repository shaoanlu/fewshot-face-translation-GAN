from tqdm import tqdm
import cv2
import numpy as np

from utils.utils import detect_face, get_tar_landmarks, landmarks_match, get_segm_mask, get_eyes_mask
from utils.utils import parse_face, detect_irises, draw_irises, auto_resize

def check_face_area_ratio(segm, threshold=0.3):
    """
    Count the ratio of face area w.r.t. whole image

    Images with ratio <= 0.3 are considered low quality images 
    (perhaps to be profile faces or blurred images.)
    """
    bg_mask = np.prod(segm==(0,0,0), axis=-1, keepdims=True)
    face_area_ratio = np.sum(bg_mask) / np.prod(bg_mask.shape)
    
    if face_area_ratio >= threshold:
        return True
    else:
        return False

def align_face(im, landmarks):
    lms_tar = get_tar_landmarks(im, 68)
    aligned_img, M = landmarks_match(
        im, 
        landmarks[0], np.array(lms_tar), 
        border_mode=cv2.BORDER_CONSTANT, 
        border_value=(0,0,0))
    return aligned_img, M

def process_single_image(im, fd, fp, idet):
    """
    Inputs:
        fn: A string. Path to a image file.
        fd: An instance of FaceAlignmentDetector. Face detector in face_toolbox_keras. 
        fp: An instance of FaceParer. Face parsing model in face_toolbox_keras.
    Outputs:
        aligned_face: A RGB image. Aligned face image.
        colored_parsing_map:  A RGB image of face parsing map.
        segm_mask: A RGB image of face mask generated from face landmarks.
        (x0, y0, x1, y1), A tuple of integers. Bounding box coordinates.
        landmarks: A numpy array of shape (68,2). 68-points face landmarks.
    """
    im = auto_resize(im)
    
    (x0, y0, x1, y1), landmarks = detect_face(im, fd)
    detected_face = im[x0:x1, y0:y1, :].copy()
    aligned_face, M = align_face(detected_face, [landmarks[0]-[x0,y0]])
    
    segm_mask = get_segm_mask(
        im, 
        aligned_face, 
        x0, y0, x1, y1, 
        landmarks)    
    segm_mask = cv2.warpAffine(segm_mask, M, (segm_mask.shape[1], segm_mask.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    colored_parsing_map = parse_face(aligned_face, segm_mask, fp=fp)
    return aligned_face, colored_parsing_map, segm_mask, (x0, y0, x1, y1), landmarks

    #eyes_mask = get_eyes_mask(colored_parsing_map)
    #eyes_lms = detect_irises(im, idet, landmarks)
    #eyes_lms = eyes_lms - np.array([[[x0, y0]]])
    #parsing_map_with_iris = draw_irises(colored_parsing_map, eyes_mask, eyes_lms)
    #return aligned_face, parsing_map_with_iris, segm_mask, (x0, y0, x1, y1), landmarks
