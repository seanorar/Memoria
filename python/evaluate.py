import cv2
import numpy
import xml.etree.ElementTree as ET
from code_m import get_img_bbox

def bb_intersection_over_union(bbox1, bbox2):
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    inter = (xB - xA + 1) * (yB - yA + 1)

    area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    iou = inter / float(area1 + area2 - inter)

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
        for line in lines:
            num_lines += 1
            path_img = path_imgs + line.split(" ")[0] + ".JPEG"
            path_xml = path_xmls + line.split(" ")[0] + ".xml"
            bboxs_gt = get_bbox_from_xml(path_xml)
            bboxs_predicted = get_img_bbox(path_img, prototxt, caffemodel, mode)
            iou = evaluate_iou(bboxs_gt, bboxs_predicted)
            avg_iou += iou
        print "dataset iou = " + str(float(avg_iou/num_lines))

#txt_data = "/home/sormeno/Datasets/Imagenet/ILSVRC2014_devkit/data/det_lists/val.txt"
#path_imgs = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_val/"
#path_xmls = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_bbox_val/"
#prototxt =  "/home/sormeno/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt"
#caffemodel = "/home/sormeno/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel" 
#get_dataset_iou(txt_data, path_imgs, path_xmls, prototxt, caffemodel, "gpu")

