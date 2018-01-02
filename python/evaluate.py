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


def get_bbox_from_xml(xml_path, get_classes=False):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_bbox = []
    gt_classes = []
    for obj in root.iter('object'):
        name = obj.find('name').text
        for bbox in obj.findall("bndbox"):
            xmin = int(bbox.find('xmin').text.split(".")[0])
            xmax = int(bbox.find('xmax').text.split(".")[0])
            ymin = int(bbox.find('ymin').text.split(".")[0])
            ymax = int(bbox.find('ymax').text.split(".")[0])
            gt_bbox.append([xmin,ymin, xmax, ymax])
            gt_classes.append(name)
    if (get_classes):
        return (gt_bbox, gt_classes)
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


def get_dataset_iou(txt_data, path_imgs, path_bbox_data, prototxt, caffemodel, img_ext , nms_iou=0.5, iou_relevant=0.5, mode="cpu"):
    with open(txt_data) as f:
        lines = f.readlines()
        avg_iou = 0.0
        avg_precision = 0.0
        avg_recall = 0.0
        num_lines = 0
        net = init_net(prototxt, caffemodel, mode)
        for id in range(0,min(600, len(lines))):
            line = lines[id]
            line = line.rstrip()
            img_id = line.split(" ")[0]
            path_img = path_imgs + img_id + img_ext
            if ".txt" in path_bbox_data:
                bboxs_gt = get_bbox_from_txt(path_bbox_data, img_id)
            else:
                path_xml = path_bbox_data + img_id + ".xml"
                bboxs_gt = get_bbox_from_xml(path_xml)
            if (len(bboxs_gt) > 0):
                bboxs_predicted = get_img_bbox2(path_img, net)
		print "iou relevante = " + str(iou_relevant)
		print "nms iou = " + str(nms_iou)
                print "rois detectados: " + str(len(bboxs_predicted))
                filtered_bboxs = apply_nms(bboxs_predicted, nms_iou)
                print "rois despues del filtro: " + str(len(filtered_bboxs))
                iou, n_relevant, gt_finded = evaluate_iou(bboxs_gt, filtered_bboxs, iou_relevant)
		precision = float(gt_finded/ (len(filtered_bboxs) - n_relevant + gt_finded))
                print gt_finded
                print (len(filtered_bboxs) - n_relevant)
                recall = float(gt_finded / len(bboxs_gt))
                #print "max iou = " + str(iou)
                print "precision = " + str(precision)
                #print str(gt_finded)+ " / " + str(len(bboxs_gt))
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


def sort_data_to_plot(x_data, y_data):
    list_tuples = []
    for i in range(0, len(x_data)):
        list_tuples.append([x_data[i], y_data[i]])
    list_tuples = sorted(list_tuples, key=lambda x: x[0])
    new_data_x, new_data_y = map(list, zip(*list_tuples))
    return (new_data_x, new_data_y)


def data_to_graphs(txt_data, path_imgs, path_xmls, prototxt, caffemodel, img_ext , mode="cpu", output="out"):
    r_iou = []
    r_presicion = []
    r_recall = []
    for i in range(5, 100 , 5):
        iou = float(i/100.0)
        r = get_dataset_iou(txt_data, path_imgs, path_xmls, prototxt, caffemodel, img_ext, 0.5, iou, mode) #nms_iou, iou_relevant
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
    to_plot([i_recall, p_recall],[i_presicion, p_presicion],['imagenet', 'pascal'],'recall', 'presicion', "presicion vs recall en " +dataset)


def plot_data_vs_trsh(dataset, id):
    path = "/home/sebastian/Escritorio/universidad/memoria/py-faster-rcnn/tools/Memoria/calculados"
    trained = "imagenet"
    i_iou, i_presicion, i_recall = load_data(path, dataset, trained, id)
    trained = "pascal"
    p_iou, p_presicion, p_recall = load_data(path, dataset, trained, id)
    x_val = [i / 100.0 for i in range(5,100,5)]

    to_plot([x_val, x_val], [p_recall, i_recall], ['pascal', 'imagenet'], 'thr_iou', 'recall', "Variacion recall en " + dataset)
    to_plot([x_val, x_val], [p_presicion, i_presicion], ['pascal', 'imagenet'], 'thr_iou', 'presicion', "Variacion presicion en " + dataset)


def load_data(path, dataset, trained, id):
    iou = np.load(path + "/" + dataset + "_" + trained +"_"+str(id)+"_iou.npz")['arr_0']
    presicion = np.load(path + "/" + dataset + "_" + trained + "_"+str(id)+"_presicion.npz")['arr_0']
    recall = np.load(path + "/" + dataset + "_" + trained + "_" + str(id)+"_recall.npz")['arr_0']
    return (iou, presicion, recall)


def plot_pca_evolution():
    x = [1,2,3,4,5,6,7,8,9,10,11,12]
    obj4 = [0.00883692, 0.00651609, 0.0172129, 0.0229793, 0.11, 0.18197, 0.200522, 0.171355, 0.206803, 0.175619, 0.197121, 0.196426]
    obj1 = [0.018042, 0.0130235, 0.0265138, 0.0972047, 0.231526, 0.299594, 0.287851, 0.286413, 0.277041, 0.279485, 0.283419, 0.275661]
    obj5 = [0.0110082, 0.00670436, 0.0139472, 0.308673, 0.45524, 0.759859, 0.806162, 0.808848, 0.79134, 0.790885, 0.751334, 0.734012]
    obj2 = [0.00886471, 0.0136486, 0.0863969, 0.213382, 0.472477, 0.519718, 0.553845, 0.549711, 0.553203, 0.572478, 0.588356, 0.593926]
    obj3 = [0.0113546, 0.0083661, 0.0139098, 0.0285699, 0.0411208, 0.052026, 0.0709322, 0.0722278, 0.0748704, 0.108918, 0.110862, 0.113706]
    obj6 = [0.00838178, 0.0678976, 0.522621, 0.925769, 0.942025, 0.943789, 0.943952, 0.945374, 0.947401, 0.948489, 0.950158, 0.954167]
    all_data_y = [obj1, obj2, obj3, obj4, obj5, obj6]
    all_data_x = [x,x,x,x,x,x,x]
    labels = ["obj1","obj2","obj3","obj4","obj5","obj6", "MAP promedio"]
    data_prom = []
    for i in range(0, 12):
        val = 0
        for j in range(0, 6):
            val += all_data_y[j][i]
        val = val/6
        data_prom.append(val)
    all_data_y.append(data_prom)
    for element in all_data_y:
        print element[8]
    to_plot(all_data_x, all_data_y, labels, "largo de descriptor", "MAP", "Evolucion PCA")

def to_plot(data_x,data_y, labels, axis_x, axis_y, title):
    for i in range(len(data_x)):
        x, y = sort_data_to_plot(data_x[i], data_y[i])
        plt.plot(x, y, label=labels[i])
    plt.ylabel(axis_y)
    plt.xlabel(axis_x)
    #plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], [2,4,8,16,32,64,128,256,512,1024,2048,4096,1])
    plt.title(title)
    plt.grid()
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


def bbox_val_imagenet(path_val_imagenet, path_xmls, path_imgs):
    with open(path_val_imagenet + "val.txt") as f:
        lines = f.readlines()
        f = open(path_val_imagenet + 'bboxs2.txt', 'w')
        for line in lines:
            name = line.strip().split(" ")[0]
            bboxs = get_bbox_from_xml(path_xmls + name + ".xml")
            img = cv2.imread(path_imgs + name + ".JPEG")
            h,w,c = img.shape
            #print bboxs
            for bbox in bboxs:
                if (bbox[0] > bbox[2] or bbox[1] > bbox[3] or w < bbox[2] or h < bbox[3] ):
                    print "error"
                else:
                    newline = name + " " + str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + "\n"
                    f.write(newline)
        f.close()

def evolucion_map():
    data = [0.24, 0.29, 0.39,0.42, 0.41, 0.37,0.38,0.35,0.43,0.44,0.47]
    data_y = [data]
    data_x = [range(1,12)]
    label=["MAP promedio"]
    to_plot(data_x, data_y, label, "ID prueba", "MAP", "Evolucion MAP en pruebas realizadas")

def barra():
    x = np.arange(30)
    money = [0.0802002, 0.337806, 0.0102041, 0.253691, 0.0872828, 0, 0.00131579, 0.22747, 0, 0.394273, 0.299594, 0.223268, 0.215824, 0, 0.162621, 0.107558, 0, 0.451023, 0, 0.0221769, 0.0199107, 0.10084, 0.261753, 0.0666667, 0.0158555, 0.243871, 0.0827959, 0, 0.267557, 0.00555556]
    #def millions(x, pos):
    #    'The two args are the value and tick position'
    #    return '$%1.1fM' % (x * 1e-6)

    #formatter = FuncFormatter(millions)

    fig, ax = plt.subplots()
    plt.grid()
    #ax.yaxis.set_major_formatter(formatter)
    plt.bar(x, money)
    plt.title("Map Consultas")
    plt.ylabel("MAP")
    plt.xlabel("ID consultas")
    #plt.xticks(x, range(9069,9098))
    plt.show()
