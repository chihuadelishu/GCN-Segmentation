import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from datasets import data as dataset
from models.nn import GCN as ConvNet
from learning.utils import draw_pixel

# Set image size and number of class
IM_SIZE = (512, 512)
NUM_CLASSES = 3

""" 2. Set test hyperparameters """
hp_d = dict()

# FIXME: Test hyperparameters
hp_d['batch_size'] = 1

""" 3. Build graph, load weights, initialize a session """
# Initialize
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([IM_SIZE[0], IM_SIZE[1], 3], NUM_CLASSES, **hp_d)
saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, './model.ckpt')    # restore learned weights

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)


while True:
    ret, frame = capture.read()
    resize = cv2.resize(frame, dsize=(512, 512), interpolation=cv2.INTER_AREA)

    test_y_pred = model.predict_video(sess, [resize, ], **hp_d)
    mask = draw_pixel(test_y_pred)
    result = mask.reshape(512, 512, 3)

    cv2.imshow("origin", resize)
    cv2.imshow("mask", result)
    if cv2.waitKey(1) > 0: break

capture.release()
cv2.destroyAllWindows()