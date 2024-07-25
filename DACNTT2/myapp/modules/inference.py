import os
import sys
import cv2
import numpy as np

dir = os.path.dirname(__file__)
sys.path.append(dir)
from tf2net_openvino import Object_Segmentation_TF2


class FaceMainipulation:

    def __init__(self) -> None:
        self.model_path = os.path.join(dir, 'model/saved_model.xml')
        self.labelmap_path = os.path.join(dir, 'model/labelmap.pbtxt')         
        self.debug_dir = os.path.join(dir, 'debug')
        os.makedirs(self.debug_dir, exist_ok=True)

    def load_model(self):
        self.segmentor = Object_Segmentation_TF2(self.model_path, self.labelmap_path, 0.65)
        self.segmentor.load_model()
    
    def free_model(self):
        del self.segmentor

    def segment_modified(self, image_path):
        try:
            image = cv2.imread(image_path)
            bboxes, segment_masks = self.segmentor(image)
            if len(segment_masks) > 0:
                merge_mask = segment_masks[0]
                for i in range(len(segment_masks)):
                    merge_mask = cv2.bitwise_or(merge_mask, segment_masks[i])
                merge_mask = np.transpose(np.array([merge_mask, merge_mask, merge_mask]), (1,2,0))
                image = cv2.addWeighted(image, 1, merge_mask, 0.2, 0)
            
            vis_image, _ = self.segmentor.visualize(image, bboxes)
            # heat map, vis image
            return vis_image, vis_image
        except:
            return None
