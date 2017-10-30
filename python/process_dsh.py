import array
import os
from scipy.spatial.distance import hamming
import numpy as np

def load_bin_feature(txt_data, path_to_bin, len_feature):
    with open(txt_data) as f:
        lines = f.readlines()
        list_classes = []
        for line in lines:
            name, class_id = line.rstrip().split(" ")
            list_classes.append(class_id)
        bf = open(path_to_bin, 'rb')
        seek_counter = 0
        map = 0.0
        while (True):
            try:
                data = array.array('f')
                data.fromfile(bf, len_feature)
                result = compare_features(data, list_classes[seek_counter], list_classes, path_to_bin, hamming)
                list_id = get_id_list(result, list_classes)
                ap = get_ap(list_classes[seek_counter], list_id)
                map = map + ap
                print seek_counter
                print ap
                seek_counter += 1
                bf.seek((seek_counter * len_feature) * 4, os.SEEK_SET)
            except:
                break
        print (map / seek_counter)

def compare_tupples(a, b , class_r, list_classes):
    if (a[1] == b[1]):
        if list_classes[a[0]] == class_r:
            return  -1
        else:
            return 1
    else:
        if a[1] < b[1]:
            return  -1
        else:
            return 1

def compare_features(feature, feature_id, list_classes, bin_data, dist):
    compare_result = []
    bf = open(bin_data, 'rb')
    id_element = 0
    len_feature = len(feature)
    while True:
        try:
            data = array.array('f')
            data.fromfile(bf, len_feature)
            compare_result.append((id_element, dist(np.sign(feature), np.sign(data))))
            id_element += 1
            bf.seek((id_element * len_feature) * 4, os.SEEK_SET)
        except:
            break
    result = sorted(compare_result, cmp= lambda x,y: compare_tupples(x,y,feature_id, list_classes))
    return result


def get_id_list(pos_list, classes_list):
    result_list = []
    for element in pos_list:
        result_list.append(classes_list[element[0]])
    return result_list


def get_ap(relevant_id, list_id):
    ap = 0.0
    n_relevants = 0.0
    for i in range(0, len(list_id)):
        if (relevant_id == list_id[i]):
            n_relevants += 1.0
            ap = ap + (n_relevants / (i + 1.0))
    return (ap / n_relevants)

txt_data = "/home/sebastian/Escritorio/val_1.txt"
path_to_bin = "/home/sebastian/Escritorio/dsh_result.bin"
load_bin_feature(txt_data, path_to_bin, 12)