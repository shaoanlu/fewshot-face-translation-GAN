from pathlib import PurePath, Path
import random
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np

from prefetch_generator import background

random_transform_args = {
    'rotation_range': 5,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.5,
    }

# ====================
# Image dataset utilities
# ====================

def get_random_transform_matrix(image, rotation_range, zoom_range, shift_range, random_flip):
    h,w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1, 1 + zoom_range )
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D( (w//2,h//2), rotation, scale)
    mat[:,2] += (tx,ty)
    return mat

def get_random_identity(exclude_identity=None, all_identities=[]):
    rand_ids = np.random.choice(all_identities, 2, replace=False)
    if rand_ids[0] == exclude_identity:
        rand_id = rand_ids[1]
    else:
        rand_id = rand_ids[0]
    return rand_id

def get_parsing_mask(identity, raw_fn, path_trn_data, return_non_face_region=False):
    fn_segm = path_trn_data / identity / "parsing_maps" / f"{raw_fn}.png"
    fn_segm = str(fn_segm)
    segm = _load_image(fn_segm)   
    if return_non_face_region:
        hair_mask = np.prod(segm==(255,255,170), axis=-1, keepdims=True)
        hat_mask = np.prod(segm==(255,0,255), axis=-1, keepdims=True)
        earing_mask = np.prod(segm==(170,255,0), axis=-1, keepdims=True)
        background_mask = np.prod(segm==(0,0,0), axis=-1, keepdims=True)
        non_face_region_mask = np.bitwise_or(np.bitwise_or(np.bitwise_or(hair_mask, hat_mask), earing_mask), background_mask)
        non_face_region_mask = non_face_region_mask.astype(np.float32) * 255 # [0, 255]
        non_face_region_mask = np.repeat(non_face_region_mask, 3, axis=-1)
        return segm, non_face_region_mask
    else:
        return segm

def get_segmentation_mask(identity, raw_fn, path_trn_data):
    fn_segm = path_trn_data / identity / "seg_mask" / f"{raw_fn}.png"
    fn_segm = str(fn_segm)
    segm = _load_image(fn_segm)
    return segm

def get_blurred_rgb(rgb, segm, landmark_segm):
    face_mask = (landmark_segm != (-1,-1,-1)) * 255
    face_mask = np.max(face_mask, axis=-1, keepdims=True).astype(np.float32)
    kernel_size = face_mask.shape[0] // 20
    kernel_size += (kernel_size % 2) - 1
    #face_mask = cv2.erode(face_mask.astype(np.uint8), np.ones((kernel_size, kernel_size)))
    face_mask = cv2.GaussianBlur(face_mask.astype(np.uint8), (kernel_size, kernel_size), 0)
    face_mask = face_mask.astype(np.float32) / 255
    face_mask = face_mask[..., np.newaxis]
    ratio = np.random.randint(8,64) / face_mask.shape[0]
    interp1, interp2 = np.random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR], 2)
    blurred_rgb = cv2.resize(rgb.copy(), (0,0), fx=ratio, fy=ratio, interpolation=interp1)
    blurred_rgb = cv2.resize(blurred_rgb, (0,0), fx=1/ratio, fy=1/ratio, interpolation=interp2)
    
    blurred_rgb = face_mask * blurred_rgb + (1 - face_mask) * rgb    
    return blurred_rgb

def get_masked_rgb(rgb, segm, landmark_segm):
    face_mask = (landmark_segm != (-1,-1,-1)) * 255
    face_mask = np.max(face_mask, axis=-1, keepdims=True).astype(np.float32)
    kernel_size = face_mask.shape[0] // np.random.randint(13,21)
    kernel_size += (kernel_size % 2) - 1
    face_mask = cv2.erode(face_mask.astype(np.uint8), np.ones((kernel_size, kernel_size)))
    face_mask = cv2.GaussianBlur(face_mask.astype(np.uint8), (kernel_size, kernel_size), 0)
    face_mask = face_mask.astype(np.float32) / 255
    face_mask = face_mask[..., np.newaxis]
    masked_rgb = np.ones_like(rgb) * np.mean(rgb, axis=(0,1)) + np.random.normal(0, 0.1, size=rgb.shape)
    
    masked_rgb = face_mask * masked_rgb + (1 - face_mask) * rgb    
    return masked_rgb

def random_transform(im, mat, horizontal_flip=True, border_mode=cv2.BORDER_REPLICATE, border_value=(0,0,0)):
    h, w = im.shape[:2]
    im = cv2.warpAffine(im, mat, (w,h), borderMode=border_mode, borderValue=border_value)
    if horizontal_flip:
        im = im[:, ::-1, :]
    return im

def random_color_match(image, tar_img):
    r = 60
    src_img = cv2.resize(image, (256,256))
    tar_img = cv2.resize(tar_img, (256,256))
    mt = np.mean(tar_img[r:-r,r:-r,:], axis=(0,1))
    st = np.std(tar_img[r:-r,r:-r,:], axis=(0,1))
    ms = np.mean(src_img[r:-r,r:-r,:], axis=(0,1))
    ss = np.std(src_img[r:-r,r:-r,:], axis=(0,1))    
    if ss.any() <= 1e-7: return src_img    
    result = st * (src_img.astype(np.float32) - ms) / (ss+1e-7) + mt
    if result.min() < 0:
        result = result - result.min()
    if result.max() > 255:
        result = (255.0/result.max()*result).astype(np.float32)
    return result

def _load_image(fn):
    im = cv2.imread(fn)[..., ::-1]
    assert not im is None, f"Failed reading file: {fn}"
    return im

def _preprocess_image(im, image_shape):
    return cv2.resize(im, image_shape[:2]) / 255 * 2 - 1

def get_data(identity, identity2, dict_fns_img, all_identities, path_trn_data, image_shape, random_transform_args=random_transform_args):
    """
    TODO: Write random_transform_args into config files
    """
    if isinstance(dict_fns_img, dict):
        rand_fn_src = np.random.choice(dict_fns_img[identity])
        rand_fn_src2 = np.random.choice(dict_fns_img[identity])
        rand_fn_tar = np.random.choice(dict_fns_img[identity2])
    else:
        rand_fn_src = dict_fns_img.get_random_filename(identity)
        rand_fn_src2 = dict_fns_img.get_random_filename(identity)
        rand_fn_tar = dict_fns_img.get_random_filename(identity2)
    raw_fn_src = PurePath(rand_fn_src).stem
    rgb_src = _load_image(rand_fn_src)
    rgb_src2 = _load_image(rand_fn_src2)
    rgb_tar = _load_image(rand_fn_tar)
    segm_src, non_face_mask = get_parsing_mask(
        identity, 
        raw_fn_src, 
        path_trn_data, 
        return_non_face_region=True) 
    non_face_mask = _preprocess_image(non_face_mask, image_shape)
    non_face_mask = (non_face_mask + 1) / 2
    segm_src = _preprocess_image(segm_src, image_shape)
    landmark_segm = get_segmentation_mask(identity, raw_fn_src, path_trn_data)
    landmark_segm = _preprocess_image(landmark_segm, image_shape)
    
    if np.random.uniform() <= 0.25:
        rgb_src = random_color_match(rgb_src, rgb_tar) # ressize to (256,256)
    rgb_src = _preprocess_image(rgb_src, image_shape)
    rgb_src2 = _preprocess_image(rgb_src2, image_shape)
    rgb_tar = _preprocess_image(rgb_tar, image_shape)
    
    mat = get_random_transform_matrix(rgb_src, **random_transform_args)
    horizontal_flip = (np.random.uniform() <= random_transform_args['random_flip'])
    rgb_src = random_transform(rgb_src, mat, horizontal_flip)
    rgb_src2 = random_transform(rgb_src2, mat, horizontal_flip)
    rgb_tar = random_transform(rgb_tar, mat, horizontal_flip)
    segm_src = random_transform(
        segm_src, 
        mat, 
        horizontal_flip, 
        border_mode=cv2.BORDER_CONSTANT, 
        border_value=(-1,-1,-1))
    non_face_mask = random_transform(
        non_face_mask, 
        mat, 
        horizontal_flip)
    landmark_segm = random_transform(
        landmark_segm, 
        mat, 
        horizontal_flip, 
        border_mode=cv2.BORDER_CONSTANT, 
        border_value=(-1,-1,-1))
    
    # generate blurred rgb image (the generator's input image)
    if np.random.choice([True, False], p=[0.9, 0.1]):
        rgb_src_blurred = get_blurred_rgb(rgb_src, segm_src, landmark_segm)
    else:
        rgb_src_blurred = get_masked_rgb(rgb_src, segm_src, landmark_segm)
    
    return (rgb_src, segm_src, rgb_src_blurred, rgb_src2, rgb_tar, non_face_mask)

# ====================
# Data generators
# ====================

class RandomFilenameGenerator():
    def __init__(self, dict_fns):
        self.dict_fns_orig = {}
        self.dict_fns_gen = {}
        for identity, filenames in dict_fns.items():
            self.dict_fns_orig[identity] = filenames.copy()
            self.dict_fns_gen[identity] = filenames.copy()
            random.shuffle(self.dict_fns_gen[identity])
            
    def get_random_filename(self, identity):
        try:
            return self.dict_fns_gen[identity].pop()
        except:
            self.init_dict_item(identity)
            return self.dict_fns_gen[identity].pop()
        
    def init_dict_item(self, identity):
        self.dict_fns_gen[identity] = self.dict_fns_orig[identity].copy()
        random.shuffle(self.dict_fns_gen[identity])
        
    def get_length(self, identity):
        return len(self.dict_fns_orig[identity])
    
@background(12)
def minibatch(config):
    dir_trn_data = PurePath(config["dir_data"])
    batch_size = config["batch_size"]
    image_shape = (config["input_size"], config["input_size"], 3)
    all_identities = glob(str(PurePath(dir_trn_data, "*")))
    all_identities = [Path(dirs).stem for dirs in all_identities]
    dict_fns_img = {}

    print(f"Found {str(len(all_identities))} identities in folder {str(dir_trn_data)}.")
    print("Loading images...")
    for identity in tqdm(all_identities):
        dict_fns_img[identity] = glob(str(PurePath(dir_trn_data, identity, "rgb", "*.jpg")))
    print("Finished.")
        
    rand_fn_generator = RandomFilenameGenerator(dict_fns_img)
    while True:
        rgb_gt_src = np.zeros((batch_size,)+image_shape)
        rgb_rand_src = np.zeros((batch_size,)+image_shape)
        rgb_tar = np.zeros((batch_size,)+image_shape)
        segm_src = np.zeros((batch_size,)+image_shape)
        rgb_inp_src = np.zeros((batch_size,)+image_shape)
        non_face_mask_src = np.zeros((batch_size,)+image_shape)
        
        for i in range(batch_size):
            id_src = get_random_identity(all_identities=all_identities)
            id_tar = get_random_identity(all_identities=all_identities, exclude_identity=id_src)
            data = get_data(id_src, id_tar, rand_fn_generator, all_identities, dir_trn_data, image_shape)
            rgb_gt_src[i, ...] = data[0]
            segm_src[i, ...] = data[1]
            rgb_inp_src[i, ...] = data[2]
            rgb_rand_src[i, ...] = data[3]
            rgb_tar[i, ...] = data[4]
            non_face_mask_src[i, ...] = data[5]
        yield rgb_gt_src, segm_src, rgb_inp_src, rgb_rand_src, rgb_tar, non_face_mask_src
