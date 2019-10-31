from pathlib import PurePath, Path
from datetime import datetime
from glob import glob
from tqdm import tqdm
import argparse
import cv2

from data.image_preprocessing import process_single_image
from utils.config_loader import load_yaml
from face_toolbox_keras.models.parser import face_parser
from face_toolbox_keras.models.detector import face_detector
from face_toolbox_keras.models.detector.iris_detector import IrisDetector

"""
Description

The input folder expects tthe following hierarchy:

- input_dir:
    - folder_name_001:
        - image01.jpg
        - image02.jpg
        - ...
    -folder_name_002:
        - image01.jpg
        - image02.jpg
        - ...
    - ...

Each sub-folder contains face images of the same identity.
"""

def is_small_image(im, min_size=128):
    for x in im.shape[:-1]:
        if x < min_size:
            return True
    return False

if __name__ == "__main__":
    path_config = "configs/config.yaml"
    config = load_yaml(path_config)

    parser = argparse.ArgumentParser()
    parser.add_argument('dir_input', type=str, help="Path to input folder.")
    args = parser.parse_args()

    # Instantiate face detector and face parser
    fd = face_detector.FaceAlignmentDetector()
    fp = face_parser.FaceParser()    
    idet = IrisDetector()
    
    # Date time is used to create output folder path
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_output = PurePath(config["dir_preprocess"], current_datetime)
    #dir_output = str(dir_output)

    # Retrieve filenames
    filenames = glob(str(PurePath(args.dir_input, "*", "*.jpg")))
    filenames += glob(str(PurePath(args.dir_input, "*", "*.png")))
    #filenames = [fn.replace("\\", "/") for fn in filenames]

    root_path = args.dir_input
    for fn in filenames:        
        raw_fn = Path(fn).stem
        folder_id = Path(fn).parts[-2] 
        Path(PurePath(dir_output, folder_id, "rgb")).mkdir(parents=True, exist_ok=True)
        Path(PurePath(dir_output, folder_id, "seg_mask")).mkdir(parents=True, exist_ok=True)
        Path(PurePath(dir_output, folder_id, "parsing_maps")).mkdir(parents=True, exist_ok=True)

        # Read data
        im = cv2.imread(fn)[..., ::-1]
        if im is None:
            print(f"Faild reading {fn}. Skip this file.")
            continue
        if is_small_image(im): 
            continue

        # Process data
        face, parsing_map, segm_mask, _, _ = process_single_image(im, fd, fp, idet)

        # Save processed data
        fname_face = PurePath(dir_output, folder_id, "rgb", f"{raw_fn}.jpg")
        fname_seg_mask = PurePath(dir_output, folder_id, "seg_mask", f"{raw_fn}.png")
        fname_parsing_map = PurePath(dir_output, folder_id, "parsing_maps", f"{raw_fn}.png")
        cv2.imwrite(str(fname_face), face[..., ::-1])
        cv2.imwrite(str(fname_seg_mask), seg_mask[..., ::-1])
        cv2.imwrite(str(fname_parsing_map), parsing_map[..., ::-1])

