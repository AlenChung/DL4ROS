import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, cv2
import argparse
import dlib

from threading import Thread, Lock

CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane','bus','train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter','bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack','umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl','banana', 'apple', 'sandwich', 'orange', 'broccoli','carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table','toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven','toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier','toothbrush')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}
## no effect
def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
## no effect
def demo(net, im, scale_factor, classes):
    """Detect object classes in an image using pre-computed object proposals."""

   #im2 = cv2.resize(im, (0,0), fx=1.0/scale_factor, fy=1.0/scale_factor)

    obj_proposals_in = []
    dlib.find_candidate_object_locations(im, obj_proposals_in, min_size=70)

    obj_proposals = np.empty((len(obj_proposals_in),4))
    for idx in range(len(obj_proposals_in)):
        obj_proposals[idx] = [obj_proposals_in[idx].left(), obj_proposals_in[idx].top(), obj_proposals_in[idx].right(), obj_proposals_in[idx].bottom()]

    # Detect all object classes and regress object bounds
    

	timer = Timer()
	timer.tic()
	scores, boxes = im_detect(net, im, obj_proposals)
	#scores, boxes = im_detect(net, im)
	timer.toc()
	print (('Detection took {:.3f}s for '
		'{:d} object proposals').format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
	CONF_THRESH = 0.8
	NMS_THRESH = 0.3
	for cls_ind, cls in enumerate(CLASSES[1:]):
	    cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            vis_detections(im, cls, dets, thresh=CONF_THRESH)
	return[im, cls, dets,CONF_THRESH]





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
   



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'coco_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                               'coco_vgg16_faster_rcnn_final.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))


    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    caffe.set_device(args.gpu_id)
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

    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
       _, _= im_detect(net, im)

    vs.stop()
    cv2.destroyAllWindows()
    
