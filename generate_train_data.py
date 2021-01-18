import os
import bs4 as bs
import cv2
from tqdm import tqdm
from utils import config, intersection_over_union


# CREATE TRAIN DIRS IF THEY DOESN'T EXIST
for dir_path in (config.POSITIVE_TRAIN_PATH, config.NEGATIVE_TRAIN_PATH):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# GET ALL IMAGE PATHS
image_paths = os.listdir(config.IMAGES_PATH)

total_positives = 0
total_negatives = 0

for image_path in tqdm(image_paths):
    # OPEN ANNOTATION FILE AND READ DATA
    annot_filename = image_path.split('.')[0] + '.xml'
    with open(os.path.join(config.ANNOTATIONS_PATH, annot_filename)) as annotation:
        content = annotation.read()
        soup = bs.BeautifulSoup(content, 'lxml')

    image_w = int(soup.find('width').text)
    image_h = int(soup.find('height').text)

    gt_boxes = list()
    for obj in soup.find_all('object'):
        x_min = int(obj.find('xmin').string)
        y_min = int(obj.find('ymin').string)
        x_max = int(obj.find('xmax').string)
        y_max = int(obj.find('ymax').string)

        # SANITY CHECK
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_w, x_max)
        y_max = min(image_h, y_max)

        gt_boxes.append((x_min, y_min, x_max, y_max))

    # LOAD IMAGE
    image = cv2.imread(os.path.join(config.IMAGES_PATH, image_path))

    # RUN SELECTIVE SEARCH
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    proposes = list()
    for x, y, width, height in rects[:config.MAX_PROPOSALS_TRAIN]:
        proposes.append((x, y, x+width, y+height))

    positive_ROIs = 0
    negative_ROIs = 0

    # ITERATE THROUGH REGION PROPOSALS:
    for proposed_box in proposes:
        prop_x_min, prop_y_min, prop_x_max, prop_y_max = proposed_box

        for gt_box in gt_boxes:
            # CHECK INTERSECTION OVER UNION
            iou = intersection_over_union(gt_box, proposed_box)
            gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_box

            # INITIALIZE ROI VARIABLE
            roi = None
            roi_path = None

            # CHECK IF PROPOSED BOX IS IN GROUND TRUTH BOX
            overlap = prop_x_min >= gt_x_min
            overlap = overlap and prop_y_min >= gt_y_min
            overlap = overlap and prop_x_max <= gt_x_max
            overlap = overlap and prop_y_max <= gt_y_max

            if iou > 0.7 and positive_ROIs < config.MAX_POSITIVE:
                roi = image[prop_y_min:prop_y_max, prop_x_min:prop_x_max]
                roi_filename = f'{total_positives}.png'
                roi_path = os.path.join(config.POSITIVE_TRAIN_PATH, roi_filename)

                positive_ROIs += 1
                total_positives += 1
            elif not overlap and iou < 0.05 and negative_ROIs < config.MAX_NEGATIVE:
                roi = image[prop_y_min:prop_y_max, prop_x_min:prop_x_max]
                roi_filename = f'{total_negatives}.png'
                roi_path = os.path.join(config.NEGATIVE_TRAIN_PATH, roi_filename)

                negative_ROIs += 1
                total_negatives += 1

            if roi is not None:
                roi = cv2.resize(roi, config.INPUT_SHAPE, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(roi_path, roi)

        if positive_ROIs >= config.MAX_POSITIVE and negative_ROIs >= config.MAX_NEGATIVE:
            break
