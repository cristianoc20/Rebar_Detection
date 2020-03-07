#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate.py
#   Author      : YunYang1994
#   Created date: 2019-02-21 15:30:26
#   Description :
#
#================================================================

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOV3
import glob as gb

class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.TEST.ANNOT_PATH
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.write_image      = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label       = cfg.TEST.SHOW_LABEL

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')

        model = YOLOV3(self.input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox

        self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape[:3]

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            }
        )
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        pred_bbox = utils.nms(pred_bbox, 0.45, method='soft-nms')
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)#
        #bboxes = utils.nms(bboxes, 0.5, method='nms')
        return bboxes

    def evaluate(self):
        num =0
        predicted_dir_path = './mAP/predicted'
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        os.mkdir(predicted_dir_path)
        os.mkdir(self.write_image_path)
        img_path = gb.glob("./data/test_dataset/*.jpg")
        list1 = []
        predict_result_path = os.path.join(predicted_dir_path, 'submmit.txt')
        for path in img_path:
            image = cv2.imread(path)
            (filepath, tempfilename) = os.path.split(path)
            print('=> predict result of %s:' % tempfilename)
            bboxes_pr = self.predict(image)

            if self.write_image:
                image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                cv2.imwrite(self.write_image_path+ tempfilename , image)
            #predict_result_path = os.path.join(predicted_dir_path, tempfilename + '.txt')
            tempfilename = tempfilename + ","
            with open(predict_result_path, 'w') as f:
                for bbox in bboxes_pr:
                    num += 1
                    coor = np.array(bbox[:4], dtype=np.int32)
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = tempfilename + xmin
                    bbox_mess = ' '.join([bbox_mess, ymin, xmax, ymax]) + '\n'
                    # f.write(bbox_mess)
                    # txt_path = open(predict_result_path, 'r')
                    # for line in txt_path.readlines():
                    #     ss = line.strip()
                    list1.append(bbox_mess)
                    # print('\t' + str(bbox_mess).strip())
                    #txt_path.close()
        print(len(list1))
        txt_path = open(predict_result_path, 'w')
        for line in list1:
            txt_path.write(line)
        txt_path.close()
        print(num)

if __name__ == '__main__':YoloTest().evaluate()



