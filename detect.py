import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.image import non_max_suppression
from utils import config


if __name__=='__main__':
    # GET PATH TO IMAGE AS ARGUMENT
    parser = argparse.ArgumentParser(description='Detect car licence plate on image')
    parser.add_argument(
        '-i', '--image',
        required=True,
        help='Path to image'
    )

    args = parser.parse_args()

    # LOAD IMAGE AND TRAINED MODEL TO RAM
    image_name = args.image.split('.')[0]
    image = cv2.imread(args.image)
    image_copy = image.copy()
    model = load_model(config.MODEL_PATH)

    # SELECTIVE SEARCH
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    boxes = list()
    rois = list()

    # ITERATE THROUGH PROPOSES
    for x, y, width, height in rects[:config.MAX_PROPOSALS]:
        roi = image[y:y + height, x:x + width]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, config.INPUT_SHAPE, interpolation=cv2.INTER_CUBIC)
        rois.append(roi)

        boxes.append((x, y, x+width, y+height))

    # PREDICT PROBABILITIES
    rois = np.asarray(rois)
    boxes = np.asarray(boxes)
    preds = model.predict(rois)

    # SELECT ONLY HIGH ENOUGH PROBS
    idx = (preds > config.MIN_PROBABILITY).reshape(-1)
    boxes = boxes[idx, :]
    preds = preds[idx].reshape(-1)

    for box, prob in zip(boxes, preds):
        (x_min, y_min, x_max, y_max) = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
        y = y_min - 10 if y_min - 10 > 10 else y_min + 10
        text= f'Car plate: {prob*100:.1f}'
        cv2.putText(image, text, (x_min, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(f'{image_name}_no_nms.png', image)

    selected_idx = non_max_suppression(boxes, preds, 50, config.MIN_IOU_FOR_NMS).numpy()
    selected_boxes = boxes[selected_idx]
    selected_probs = preds[selected_idx]

    for box, prob in zip(selected_boxes, selected_probs):
        (x_min, y_min, x_max, y_max) = box
        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
        y = y_min - 10 if y_min - 10 > 10 else y_min + 10
        text= f'Car plate: {prob*100:.1f}'
        cv2.putText(image_copy, text, (x_min, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(f'{image_name}_with_nms.png', image_copy)
