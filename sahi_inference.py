import cv2
import numpy as np
from ultralytics import YOLO


class SAHIDetector:
    def __init__(self, model, slice_w=640, slice_h=360,
                 overlap=0.2, conf=0.35, iou=0.45):
        """
        SAHI - Sliced Inference per objekte te vogla
        slice_w, slice_h : dimensionet e cdo patch
        overlap          : perputhja mes patches (0.2 = 20%)
        """
        self.model   = model
        self.slice_w = slice_w
        self.slice_h = slice_h
        self.overlap = overlap
        self.conf    = conf
        self.iou     = iou

    def _get_slices(self, img_w, img_h):
        """Gjeneron koordinatat e te gjitha patches me overlap"""
        slices = []
        step_x = int(self.slice_w * (1 - self.overlap))
        step_y = int(self.slice_h * (1 - self.overlap))

        y = 0
        while y < img_h:
            x = 0
            while x < img_w:
                x2 = min(x + self.slice_w, img_w)
                y2 = min(y + self.slice_h, img_h)
                x1 = max(0, x2 - self.slice_w)
                y1 = max(0, y2 - self.slice_h)
                slices.append((x1, y1, x2, y2))
                if x2 == img_w:
                    break
                x += step_x
            if y2 == img_h:
                break
            y += step_y

        return slices

    def _nms(self, boxes, scores, iou_threshold=0.45):
        """Non-Maximum Suppression per te mbivendosura"""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep  = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w   = np.maximum(0, xx2 - xx1)
            h   = np.maximum(0, yy2 - yy1)
            inter     = w * h
            iou_vals  = inter / (areas[i] + areas[order[1:]] - inter)
            inds      = np.where(iou_vals <= iou_threshold)[0]
            order     = order[inds + 1]

        return keep

    def detect(self, frame, classes=None):
        """
        Run SAHI inference ne frame te plote
        Returns: list e [{box, conf, cls}]
        """
        img_h, img_w = frame.shape[:2]
        slices       = self._get_slices(img_w, img_h)

        all_boxes  = []
        all_scores = []
        all_cls    = []

        for (x1, y1, x2, y2) in slices:
            patch   = frame[y1:y2, x1:x2]
            results = self.model(
                patch,
                conf=self.conf,
                iou=self.iou,
                classes=classes,
                device=0,
                verbose=False
            )

            for box, conf, cls in zip(
                results[0].boxes.xyxy,
                results[0].boxes.conf,
                results[0].boxes.cls
            ):
                # konverto koordinatat e patch ne koordinata te frame-it
                bx1 = float(box[0]) + x1
                by1 = float(box[1]) + y1
                bx2 = float(box[2]) + x1
                by2 = float(box[3]) + y1
                all_boxes.append([bx1, by1, bx2, by2])
                all_scores.append(float(conf))
                all_cls.append(int(cls))

        if not all_boxes:
            return []

        boxes_arr  = np.array(all_boxes)
        scores_arr = np.array(all_scores)
        cls_arr    = np.array(all_cls)

        # NMS per te hequr detektimet e dyfishta
        keep = self._nms(boxes_arr, scores_arr, self.iou)

        detections = []
        for k in keep:
            detections.append({
                "box":  boxes_arr[k],
                "conf": scores_arr[k],
                "cls":  cls_arr[k]
            })

        return detections

    def get_slice_count(self, img_w=1280, img_h=720):
        return len(self._get_slices(img_w, img_h))
