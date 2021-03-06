import os

# PATH TO ORIGINAL DATASET
DATA_BASE_PATH = 'plates'
ANNOTATIONS_PATH = os.path.join(DATA_BASE_PATH, 'annotations')
IMAGES_PATH = os.path.join(DATA_BASE_PATH, 'images')

# PATH TO TRAINING DATA FOR CNN GENERATED BY SCRIPT
TRAIN_BASE_PATH = 'train'
NEGATIVE_TRAIN_PATH = os.path.join(TRAIN_BASE_PATH, 'negative')
POSITIVE_TRAIN_PATH = os.path.join(TRAIN_BASE_PATH, 'positive')

# MAX PROPOSALS FOR SELECTIVE SEARCH DURING GENERATING TRAINING DATA AND PERFORMING INFERENCE
MAX_PROPOSALS_TRAIN = 5000
MAX_PROPOSALS = 200

# MAX SAMPLES TO SAVE PER EACH IMAGE
MAX_POSITIVE = 40
MAX_NEGATIVE = 10

INPUT_SHAPE = (224, 224)
MODEL_PATH = 'model.h5'

MIN_PROBABILITY = 0.9
MIN_IOU_FOR_NMS = 0.25