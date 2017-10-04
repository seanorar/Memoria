import cv2
import numpy as np
import xml.etree.ElementTree as ET
from code_m import get_img_bbox, get_dataset_bbox, get_img_bbox2, init_net
from fast_rcnn.nms_wrapper import nms

def apply_nms(bbox_list, nms_thresh):
    fake_scores = []
    for i in range(0,len(bbox_list)):
        fake_scores.append(1.0 - (0.01*i))

    keep = nms(np.hstack((bbox_list, fake_scores)), nms_thresh)
    result_bbox = bbox_list[keep, :]
    return result_bbox

def bb_intersection_over_union(bbox1, bbox2):
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    inter = (xB - xA + 1) * (yB - yA + 1)
    if ( ((xB - xA + 1) < 0) or ((yB - yA + 1) < 0) ):
        return 0
    else:

        area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

        iou = inter / float(area1 + area2 - inter)
        if (iou > 1):
            print "error" 
        return iou

def get_bbox_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_bbox = []
    for obj in root.iter('object'):
        for bbox in obj.iter("bndbox"):
            xmin = int(bbox.find('xmin').text)
            xmax = int(bbox.find('xmax').text)
            ymin = int(bbox.find('ymin').text)
            ymax = int(bbox.find('ymax').text)
            gt_bbox.append([xmin,ymin, xmax, ymax])
    return gt_bbox


def evaluate_iou(bboxs_gt, bboxs_predicted):
    avg_iou = 0
    for bbox_gt in bboxs_gt:
        max_iou = 0
        for bbox_predicted in  bboxs_predicted:
            iou = bb_intersection_over_union(bbox_gt, bbox_predicted)
            if (max_iou < iou):
                max_iou = iou
        avg_iou = avg_iou + max_iou
    return float(avg_iou / len(bboxs_gt))


def get_dataset_iou(txt_data, path_imgs, path_xmls, prototxt, caffemodel, mode = "cpu"):
    with open(txt_data) as f:
        lines = f.readlines()
        avg_iou = 0
        num_lines = 0
        net = init_net(prototxt, caffemodel, mode)
        for line in lines:
            path_img = path_imgs + line.split(" ")[0] + ".JPEG"
            path_xml = path_xmls + line.split(" ")[0] + ".xml"
            bboxs_gt = get_bbox_from_xml(path_xml)
            if (len(bboxs_gt) > 0):
                bboxs_predicted = get_img_bbox2(path_img, net)
                iou = evaluate_iou(bboxs_gt, bboxs_predicted)
                avg_iou += iou
                print iou
                num_lines += 1
        print "dataset iou = " + str(float(avg_iou/num_lines))


txt_data = "/home/sormeno/Datasets/Imagenet/ILSVRC13/data/det_lists/val.txt"
path_imgs = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_val/"
path_xmls = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_bbox_val/"
prototxt =  "/home/sormeno/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt"
caffemodel = "/home/sormeno/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel" 
get_dataset_iou(txt_data, path_imgs, path_xmls, prototxt, caffemodel, "gpu")

