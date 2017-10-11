import cv2
import numpy as np
import xml.etree.ElementTree as ET
from code_m import get_img_bbox, get_dataset_bbox, get_img_bbox2, init_net
from fast_rcnn.nms_wrapper import nms
import matplotlib.pyplot as plt

def apply_nms(bbox_list, nms_thresh):
    fake_scores = []
    for i in range(0,len(bbox_list)):
        fake_scores.append(np.array([1.0 - (0.001*i)], dtype='f'))

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
            xmin = int(bbox.find('xmin').text.split(".")[0])
            xmax = int(bbox.find('xmax').text.split(".")[0])
            ymin = int(bbox.find('ymin').text.split(".")[0])
            ymax = int(bbox.find('ymax').text.split(".")[0])
            gt_bbox.append([xmin,ymin, xmax, ymax])
    return gt_bbox


def evaluate_iou(bboxs_gt, bboxs_predicted):
    avg_iou = 0.0
    n_relevants = 0.0
    gt_finded = 0.0
    for bbox_gt in bboxs_gt:
        max_iou = 0
        for bbox_predicted in  bboxs_predicted:
            iou = bb_intersection_over_union(bbox_gt, bbox_predicted)
            if (iou >= 0.5):
                n_relevants += 1.0
            if (max_iou < iou):
                max_iou = iou
        if (n_relevants > 0):
            gt_finded += 1
        avg_iou = avg_iou + max_iou
    return (float(avg_iou / len(bboxs_gt)), n_relevants, gt_finded)


def get_dataset_iou(txt_data, path_imgs, path_xmls, prototxt, caffemodel, nms_iou=0.5, mode="cpu"):
    with open(txt_data) as f:
        lines = f.readlines()
        avg_iou = 0.0
        avg_precision = 0.0
        avg_recall = 0.0
        num_lines = 0
        net = init_net(prototxt, caffemodel, mode)
        for line in lines:
            line = line.rstrip()
            path_img = path_imgs + line.split(" ")[0] + ".jpg"
            path_xml = path_xmls + line.split(" ")[0] + ".xml"
            bboxs_gt = get_bbox_from_xml(path_xml)
            if (len(bboxs_gt) > 0):
                bboxs_predicted = get_img_bbox2(path_img, net)
                filtered_bboxs = apply_nms(bboxs_predicted, nms_iou)
                iou, n_relevant, gt_finded = evaluate_iou(bboxs_gt, filtered_bboxs)
                precision = float(n_relevant / len(filtered_bboxs))
                recall = float(gt_finded / len(bboxs_gt))
                #print "max iou = " + str(iou)
                #print "precision = " + str(precision)
                print str(gt_finded)+ " / " + str(len(bboxs_gt))
                print "recall = " + str (recall)
                num_lines += 1
                avg_iou += iou
                avg_precision += precision
                avg_recall += recall 
        print "dataset iou = " + str(float(avg_iou / num_lines))
        print "dataset map = " + str(float(avg_precision / num_lines))
        print "dataset recall = " + str(float(avg_recall / num_lines))
        return (float(avg_iou / num_lines),float(avg_precision / num_lines),float(avg_recall / num_lines))


def data_to_graphs(txt_data, path_imgs, path_xmls, prototxt, caffemodel, mode="cpu", output="out"):
    r_iou = []
    r_presicion = []
    r_recall = []
    for i in range(5, 100 , 5):
        iou = float(i/100.0)
        r = get_dataset_iou(txt_data, path_imgs, path_xmls, prototxt, caffemodel, iou, mode)
        r_iou.append(r[0])
        r_presicion.append(r[1])
        r_recall.append(r[2])
    np.savez(output + "_iou", r_iou)
    np.savez(output + "_presicion", r_presicion)
    np.savez(output + "_recall", r_recall)

def to_plot():
    r_iou = [0.44584649527506504, 0.5062535687118969, 0.5342011713109763, 0.5532779513454696, 0.5690704146371486, 0.5838816962721233, 0.598112450293735, 0.6119567583581103, 0.6259686894492862, 0.6397117606770278, 0.6559784835854291, 0.6708644089387157, 0.6884734388748447, 0.7091674284905564, 0.749944744786584, 0.7509030749337933, 0.7509030749337933, 0.7509030749337933, 0.7509030749337933, 0.7509030749337933, 0.7509030749337933]
    r_presicion = [0.27912128143837894, 0.1576551129848432, 0.13440007231696488, 0.11668162937366347, 0.10163280880613884, 0.0889425862375645, 0.07977745336089577, 0.07446203093336243, 0.0720217676800067, 0.07375549934508299, 0.07758539567458296, 0.0830987734610237, 0.08955421848038438, 0.10010830165600668, 0.12440397421323002, 0.12492992890257856, 0.12492992890257856, 0.12492992890257856, 0.12492992890257856, 0.12492992890257856, 0.12492992890257856]
    r_recall = [0.6675047257722235, 0.7304993383984629, 0.7654896926297471, 0.7965042258040684, 0.8318227465598921, 0.8747422384753438, 0.9162677577373299, 0.9466677801628421, 0.9700637983480157, 0.9840750082912066, 0.9906483323571336, 0.9925456660733788, 0.99410473916508, 0.9948819852676105, 0.9956407834770172, 0.9956906737184861, 0.9956906737184861, 0.9956906737184861, 0.9956906737184861, 0.9956906737184861, 0.9956906737184861]


    list_touple=[]
    for i in range(0, len(r_presicion)):
        list_touple.append((r_presicion[i], r_recall[i]))

    sorted_by_second = sorted(list_touple, key=lambda tup: tup[1])
    n_p = []
    n_r = []
    for i in range(0, len(r_presicion)):
        n_p.append(sorted_by_second[i][0])
        n_r.append(sorted_by_second[i][1])

    plt.plot(n_r, n_p)
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.show()


#imagenet
#-------------------------------------------------------------------------

#txt_data = "/home/sormeno/Datasets/Imagenet/ILSVRC13/data/det_lists/val.txt"
#path_imgs = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_val/"
#path_xmls = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_bbox_val/"

#prototxt =  "/home/sormeno/py-faster-rcnn/models/imagenet/VGG16/faster_rcnn_end2end/test.prototxt"
#caffemodel = "/home/sormeno/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_imagenet.caffemodel"

#data_to_graphs(txt_data, path_imgs, path_xmls, prototxt, caffemodel, "gpu", "/home/sormeno/pascal_1") 


#pascal
#---------------------------------------------------------------------------
txt_data = "/home/sormeno/Datasets/Pascal/val.txt"
path_imgs = "/home/sormeno/Datasets/Pascal/Images/"
path_xmls = "/home/sormeno/Datasets/Pascal/xmls/"

prototxt =  "/home/sormeno/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt"
caffemodel = "/home/sormeno/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel"
#----------------------------------------------------------------------------
data_to_graphs(txt_data, path_imgs, path_xmls, prototxt, caffemodel, "gpu", "/home/sormeno/pascal_1")

prototxt =  "/home/sormeno/py-faster-rcnn/models/imagenet/VGG16/faster_rcnn_end2end/test.prototxt"
caffemodel = "/home/sormeno/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_imagenet.caffemodel"

data_to_graphs(txt_data, path_imgs, path_xmls, prototxt, caffemodel, "gpu", "/home/sormeno/imagenet_1")

