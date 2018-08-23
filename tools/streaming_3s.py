import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import sys
 
from threading import Thread, Lock
 
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
 
class WebcamVideoStream :
    def __init__(self, src = 0, width = 1024, height = 768) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()
 
    def start(self) :
        if self.started :
            print "already started!!"
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self
 
    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()
 
    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame
 
    def stop(self) :
        self.started = False
 
    def stop(self) :
        self.started = False
        self.thread.join()
 
    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()
 
 
if __name__ == "__main__" :
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
 
    prototxt = os.path.join(cfg.MODELS_DIR, "ZF", "faster_rcnn_end2end", "test.prototxt")
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', "ZF_faster_rcnn_final.caffemodel")
 
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
            'fetch_faster_rcnn_models.sh?').format(caffemodel))
 
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
 
    print "starting capture"
    vs = WebcamVideoStream().start()
    while True :
        frame = vs.read()
 
        # do detection and classification
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, frame)
        timer.toc()
 
        # print stats
        print ('Detection took {:.3f}s for '
                        '{:d} object proposals').format(timer.total_time, boxes.shape[0])
 
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
 
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) == 0:
                continue
 
            print CLASSES[cls_ind], "detected"
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]), (0, 255, 0), 3)
 
        # show image
        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) == 27 :
            break
 
    vs.stop()
    cv2.destroyAllWindows()



