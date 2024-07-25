import os
import cv2
import numpy as np
from openvino.runtime import Core
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Object_Segmentation_TF2:
    
    def __init__(self, model_path, labelmap_path, threshold) -> None:
        self.model_path = model_path
        self.labelmap_path = labelmap_path
        self.threshold = threshold
        self.max_dim = 1280
        self.input_name = 'input_tensor'
        self.output_names = ['detection_boxes', 'detection_masks', 'num_detections', 'detection_classes', 'detection_scores']
    
    def load_model(self):
        try:
            self.ie = Core()
            self.compiled_openvino_model = self.ie.compile_model(model=self.ie.read_model(model=self.model_path), device_name='CPU')
            self.labelmap = self.load_labelmap(self.labelmap_path)
        except Exception as e: print(e)

    def load_labelmap(self, labelmap_path):
        labels = {}
        with open(labelmap_path, "r") as file:
            for line in file:
                if 'id:' in line:
                    id = int(line.strip().replace('id:', '').strip())
                elif 'name:' in line:
                    name = line.strip().replace('name:', '').strip().strip("'")
                    labels[id] = name
        return labels

    def extract_bboxes(self, bboxes, masks, bclasses, bscores, im_width, im_height):
        bboxes_ = []
        masks_ = []
        for idx in range(len(bboxes)):
            if bscores[idx] >= self.threshold:
                y_min = int(bboxes[idx][0] * im_height)
                x_min = int(bboxes[idx][1] * im_width)
                y_max = int(bboxes[idx][2] * im_height)
                x_max = int(bboxes[idx][3] * im_width)
                class_name = self.labelmap[int(bclasses[idx])]
                bboxes_.append([x_min, y_min, x_max, y_max, class_name, float(bscores[idx])])     
                masks_.append(masks[idx])
        return bboxes_, masks_

    def visualize(self, image, bboxes):
        mask = np.zeros(image.shape, dtype=np.uint8)
        visualized_image = image.copy()
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, cls_name, score = bbox
            cv2.rectangle(visualized_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (255, 225, 225), -1)
            cv2.putText(visualized_image, str(cls_name), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(visualized_image, '{score}%'.format(score=int(score*100)), (xmax, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 255, 100), 1, cv2.LINE_AA)
        return visualized_image, mask[:, :, 0]

    def calculate_iou(self, mask_1, mask_2):
        intersection = mask_1 & mask_2
        area_intersection = np.sum(intersection)
        area_mask1 = np.sum(mask_1)
        area_mask2 = np.sum(mask_2)
        iou = area_intersection / (area_mask1 + area_mask2 - area_intersection)
        return iou

    def nms_masks(self, bboxes, masks, iou_threshold=0.3):
        selected_boxes, selected_masks = [], []
        bboxes.sort(key=lambda x: x[-1], reverse=True)
        while len(masks) > 0:
            best_mask = masks[0]
            selected_masks.append(best_mask)
            selected_boxes.append(bboxes[0])
            bboxes = bboxes[1:]
            masks = masks[1:]
            masks = [mask for mask in masks if self.calculate_iou(best_mask, mask) < iou_threshold]
        return selected_boxes, selected_masks

    def resize_max_dim(self, image):
        H, W = image.shape[:2]
        if W > H and W > self.max_dim: 
            image = cv2.resize(image.copy(), (self.max_dim, int(H/W*self.max_dim)))
        elif H > W and H > self.max_dim: 
            image = cv2.resize(image.copy(), (int(W/H*self.max_dim), self.max_dim))
        return image

    def __call__(self, image):
        image = image.copy()
        H, W = image.shape[:2]
        image = self.resize_max_dim(image)
        image = np.expand_dims(image, axis=0).astype(np.uint8)
        detections = self.compiled_openvino_model(image)
        detections = {name: detections[name] for name in self.output_names}
        bbox = detections['detection_boxes'][0]
        bclass = detections['detection_classes'][0].astype(np.int32)
        bscore = detections['detection_scores'][0]
        masks = detections['detection_masks'][0]
        bboxes, masks = self.extract_bboxes(bbox, masks, bclass, bscore, W, H)
        
        segment_masks = []
        for i, mask_ in enumerate(masks):
            segment_mask = np.zeros((H, W), dtype=np.uint8)
            xmin, ymin, xmax, ymax, cls_name, score = bboxes[i]
            w, h = xmax - xmin, ymax - ymin
            mask_ = cv2.resize(mask_, (w, h))
            mask_[mask_>=0.5] = 255
            mask_ = mask_.astype(np.uint8)
            mask_ = cv2.erode(mask_, (3,3), 1)
            segment_mask[ymin:ymax, xmin:xmax] = mask_
            segment_mask = cv2.threshold(segment_mask, 30, 255, cv2.THRESH_BINARY)[1]
            segment_masks.append(segment_mask)
            
        bboxes, segment_masks = self.nms_masks(bboxes.copy(), segment_masks.copy())
        return bboxes, segment_masks
