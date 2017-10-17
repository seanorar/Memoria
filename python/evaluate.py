import cv2
import numpy as np
import xml.etree.ElementTree as ET
from code_m import get_img_bbox, get_dataset_bbox, get_img_bbox2, init_net
from fast_rcnn.nms_wrapper import nms
import matplotlib.pyplot as plt
from random import shuffle

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

def get_bbox_from_txt(txt_path, img_id):
        with open(txt_path) as f:
            lines = f.readlines()
            result = []
            for line in lines:
                str_data = line.strip()
                split_data =str_data.split(" ")
                if (split_data[0] == img_id):
                    aux = [int(i) for i in split_data]
                    result.append([aux[3],aux[4],aux[1] + aux[3], aux[2] + aux[4]])
            return result

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

def evaluate_iou(bboxs_gt, bboxs_predicted, iou_relevant=0.5):
    avg_iou = 0.0
    check_relevant = np.zeros(len(bboxs_predicted))
    gt_finded_total = 0.0
    for bbox_gt in bboxs_gt:
        max_iou = 0
        gt_finded = 0
        for i in range(0, len(bboxs_predicted)):
            bbox_predicted = bboxs_predicted[i]
            iou = bb_intersection_over_union(bbox_gt, bbox_predicted)
            if (iou >= iou_relevant):
                check_relevant[i] = 1.0
                gt_finded = 1
            if (max_iou < iou):
                max_iou = iou
        if (gt_finded > 0):
            gt_finded_total += 1
        avg_iou = avg_iou + max_iou
    n_relevants = sum(check_relevant)
    return (float(avg_iou / len(bboxs_gt)), n_relevants, gt_finded_total)

def get_dataset_iou(txt_data, path_imgs, path_bbox_data, prototxt, caffemodel, nms_iou=0.5, iou_relevant=0.5, mode="cpu"):
    with open(txt_data) as f:
        lines = f.readlines()
        avg_iou = 0.0
        avg_precision = 0.0
        avg_recall = 0.0
        num_lines = 0
        net = init_net(prototxt, caffemodel, mode)
        for line in lines:
            line = line.rstrip()
            img_id = line.split(" ")[0]
            path_img = path_imgs + img_id + ".jpg"
            if ".txt" in path_bbox_data:
                bboxs_gt = get_bbox_from_txt(path_bbox_data, img_id)
            else:
                path_xml = path_bbox_data + img_id + ".xml"
                bboxs_gt = get_bbox_from_xml(path_xml)
            if (len(bboxs_gt) > 0):
                bboxs_predicted = get_img_bbox2(path_img, net)
                print "rois detectados: " + str(len(bboxs_predicted))
                filtered_bboxs = apply_nms(bboxs_predicted, nms_iou)
                print "rois despues del filtro: " + str(len(filtered_bboxs))
                iou, n_relevant, gt_finded = evaluate_iou(bboxs_gt, filtered_bboxs, iou_relevant)
                precision = float(n_relevant / len(filtered_bboxs))
                recall = float(gt_finded / len(bboxs_gt))
                #print "max iou = " + str(iou)
                print "precision = " + str(precision)
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

def show_best_roi(img, gt, predicted):
    best_rois = []
    iou_scores = []
    for bb_gt in gt:
        max_iou = -1
        best_roi = []
        for bb_predicted in predicted:
            iou = bb_intersection_over_union(bb_gt, bb_predicted)
            if (iou > max_iou):
                best_roi = bb_predicted
                max_iou = iou
        best_rois.append(best_roi)
        iou_scores.append(max_iou)
    for i in range(0, len(gt)):
        print iou_scores[i]
        aux_img = img.copy()
        cv2.rectangle(aux_img, (gt[i][0], gt[i][1]), (gt[i][2], gt[i][3]), (255, 255, 255), 3)
        cv2.rectangle(aux_img, (best_rois[i][0], best_rois[i][1]), (best_rois[i][2], best_rois[i][3]), (0, 255, 0), 3)
        cv2.imshow('image', aux_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def data_to_graphs(txt_data, path_imgs, path_xmls, prototxt, caffemodel, mode="cpu", output="out"):
    r_iou = []
    r_presicion = []
    r_recall = []
    for i in range(5, 100 , 5):
        iou = float(i/100.0)
        r = get_dataset_iou(txt_data, path_imgs, path_xmls, prototxt, caffemodel, iou, 0.5, mode)
        r_iou.append(r[0])
        r_presicion.append(r[1])
        r_recall.append(r[2])
    np.savez(output + "_iou", r_iou)
    np.savez(output + "_presicion", r_presicion)
    np.savez(output + "_recall", r_recall)

def plot_presicion_vs_recall(dataset, id):
    path = "/home/sebastian/Escritorio/universidad/memoria/py-faster-rcnn/tools/Memoria/calculados"
    trained = "imagenet"
    i_iou, i_presicion, i_recall = load_data(path, dataset, trained, id)
    trained = "pascal"
    p_iou, p_presicion, p_recall = load_data(path, dataset, trained, id)
    to_plot([i_recall, p_recall],[i_presicion, p_presicion],['imagenet', 'pascal'],'recall', 'presicion')

def plot_data_vs_trsh(dataset, id):
    path = "/home/sebastian/Escritorio/universidad/memoria/py-faster-rcnn/tools/Memoria/calculados"
    trained = "imagenet"
    i_iou, i_presicion, i_recall = load_data(path, dataset, trained, id)
    trained = "pascal"
    p_iou, p_presicion, p_recall = load_data(path, dataset, trained, id)
    x_val = [i / 100.0 for i in range(5,100,5)]

    to_plot([x_val, x_val], [p_recall, i_recall], ['pascal', 'imagenet'], 'thr_iou', 'recall')
    to_plot([x_val, x_val], [p_presicion, i_presicion], ['pascal', 'imagenet'], 'thr_iou', 'presicion')

def load_data(path, dataset, trained, id):
    iou = np.load(path + "/" + dataset + "_" + trained +"_"+str(id)+"_iou.npz")['arr_0']
    presicion = np.load(path + "/" + dataset + "_" + trained + "_"+str(id)+"_presicion.npz")['arr_0']
    recall = np.load(path + "/" + dataset + "_" + trained + "_" + str(id)+"_recall.npz")['arr_0']
    return (iou, presicion, recall)

def to_plot(data_x,data_y, labels, axis_x, axis_y):
    for i in range(len(data_x)):
        plt.plot(data_x[i], data_y[i], label=labels[i])
    plt.ylabel(axis_y)
    plt.xlabel(axis_x)
    plt.legend()
    plt.show()

def create_mini_imagenet(path_val_imagenet):
    with open(path_val_imagenet+"val.txt") as f:
        lines = f.readlines()
        shuffle(lines)
        selected = lines[:5000]
        f = open(path_val_imagenet + 'min_val.txt', 'w')
        for line in selected:
            f.write(line)
        f.close()

#imagenet
#-------------------------------------------------------------------------
"""
txt_data = "/home/sormeno/Datasets/Imagenet/ILSVRC13/data/det_lists/val.txt"
path_imgs = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_val/"
path_xmls = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_bbox_val/"

prototxt =  "/home/sormeno/py-faster-rcnn/models/imagenet/VGG16/faster_rcnn_end2end/test.prototxt"
caffemodel = "/home/sormeno/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_imagenet.caffemodel"

data_to_graphs(txt_data, path_imgs, path_xmls, prototxt, caffemodel, "gpu", "/home/sormeno/pascal_1") 
"""

#pascal
#---------------------------------------------------------------------------
"""
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
"""
#dataset
#---------------------------------------------------------------------------
"""
txt_data = "/home/sormeno/Desktop/videos/1/val.txt"
path_imgs = "/home/sormeno/Desktop/videos/1/shots/"
path_xmls = "/home/sormeno/Desktop/videos/1/bbox_data.txt"

prototxt = "/home/sormeno/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt"
caffemodel = "/home/sormeno/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel"

#----------------------------------------------------------------------------
data_to_graphs(txt_data, path_imgs, path_xmls, prototxt, caffemodel, "gpu", "/home/sormeno/mdata_pascal_1")

prototxt =  "/home/sormeno/py-faster-rcnn/models/imagenet/VGG16/faster_rcnn_end2end/test.prototxt"
caffemodel = "/home/sormeno/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_imagenet.caffemodel"

data_to_graphs(txt_data, path_imgs, path_xmls, prototxt, caffemodel, "gpu", "/home/sormeno/mdata_imagenet_1")
"""
#plot_presicion_vs_recall("mdata", 2)
#plot_data_vs_trsh("mdata", 2)

#print get_bbox_from_txt("/home/sebastian/Escritorio/universidad/memoria/py-faster-rcnn/tools/videos/1/bbox_detected.txt", 122)

"""
prototxt = "/home/sebastian/Escritorio/data_app/test_pascal.prototxt"
caffemodel = "/home/sebastian/Escritorio/data_app/VGG16_faster_rcnn_final.caffemodel"
path_img = "/home/sebastian/Escritorio/universidad/memoria/py-faster-rcnn/tools/videos/1/to_proces/12.jpg"
path_xml = "/home/sebastian/Escritorio/ILSVRC2012_val_00000001.xml"
path_txt = "/home/sebastian/Escritorio/universidad/memoria/py-faster-rcnn/tools/videos/1/bbox_detected.txt"
gt = get_bbox_from_txt(path_txt, "12")
net = init_net(prototxt, caffemodel, "cpu")
predicted = get_img_bbox2(path_img, net)
img = cv2.imread(path_img)
show_best_roi(img,gt, predicted)
"""
create_mini_imagenet("/home/sormeno/Datasets/Imagenet/ILSVRC13/data/det_lists/")
