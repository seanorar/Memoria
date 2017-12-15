import array
import os
from scipy.spatial.distance import hamming, euclidean
import numpy as np


def evaluate_map(txt_data, path_to_bin, len_feature, comp_func):
    with open(txt_data) as f:
        lines = f.readlines()
        list_classes = []
        for line in lines:
            name, class_id = line.rstrip().split(" ")
            list_classes.append(class_id)
        bf = open(path_to_bin, 'rb')
        seek_counter = 0
        map = 0.0
        #all_features = load_all_features(path_to_bin, len_feature)
        while (True):
            try:
                data = array.array('f')
                data.fromfile(bf, len_feature)
                result = compare_features(data, list_classes[seek_counter], list_classes, path_to_bin, comp_func)
                #result = compare_features2(data, list_classes[seek_counter], list_classes, all_features, comp_func)
                list_id = get_id_list(result, list_classes)
                ap = get_ap(list_classes[seek_counter], list_id)
                map = map + ap
                print seek_counter
                #print result
                #print list_id
                seek_counter += 1
                print map / seek_counter
                bf.seek((seek_counter * len_feature) * 4, os.SEEK_SET)
            except:
                break
        print "------------------------"
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


def load_all_features(bin_data, len_feature):
    result = []
    bf = open(bin_data, 'rb')
    id_element = 0
    while True:
        try:
            data = array.array('f')
            data.fromfile(bf, len_feature)
            result.append(data)
            id_element += 1
            bf.seek((id_element * len_feature) * 4, os.SEEK_SET)
        except:
            break
    return result

def load_n_features(bin_data, len_feature, n):
    result = []
    bf = open(bin_data, 'rb')
    id_element = 0
    while True:
        for i in range(0,n):
            try:
                data = array.array('f')
                data.fromfile(bf, len_feature)
                result.append(data)
                id_element += 1
                bf.seek((id_element * len_feature) * 4, os.SEEK_SET)
            except:
                break
    return result

def compare_features(feature, feature_id, list_classes, bin_data, dist):
    compare_result = []
    bf = open(bin_data, 'rb')
    id_element = 0
    len_feature = len(feature)
    while True:
        try:
            data = array.array('f')
            data.fromfile(bf, len_feature)
            compare_result.append((id_element, dist(feature, data)))
            id_element += 1
            bf.seek((id_element * len_feature) * 4, os.SEEK_SET)
        except:
            break
    result = sorted(compare_result, cmp= lambda x,y: compare_tupples(x,y,feature_id, list_classes))
    return result


def compare_features_without_classes(feature, bin_data, dist):
    compare_result = []
    bf = open(bin_data, 'rb')
    id_element = 0
    len_feature = len(feature)
    while True:
        try:
            data = array.array('f')
            data.fromfile(bf, len_feature)
            print feature
            print data
            print "-------------------------"
            compare_result.append(dist(feature, data))
            id_element += 1
            bf.seek((id_element * len_feature) * 4, os.SEEK_SET)
        except:
            break
    result = sorted(compare_result)
    return result

def compare_features2(feature, feature_id, list_classes, all_data, dist):
    compare_result = []
    for id_element in range(0, len(all_data)):
        data = all_data[id_element]
        compare_result.append((id_element, dist(feature, data)))
    result = sorted(compare_result, cmp= lambda x,y: compare_tupples(x,y,feature_id, list_classes))
    return result


def get_id_list(pos_list, classes_list):
    result_list = []
    for element in pos_list:
        result_list.append(classes_list[element[0]])
    return result_list


def hamming_dist(feature1, feature2):
    return hamming(np.sign(feature1), np.sign(feature2))


def get_ap(relevant_id, list_id):
    ap = 0.0
    n_relevants = 0.0
    for i in range(0, len(list_id)):
        if (relevant_id == list_id[i]):
            n_relevants += 1.0
            ap = ap + (n_relevants / (i + 1.0))
    return (ap / n_relevants)

#txt_data = "/home/sebastian/Escritorio/datos_recibidos/val_pascal.txt"
#path_to_bin = "/home/sebastian/Escritorio/datos_recibidos/pascal_4096_result.bin"

#txt_data = "/home/sebastian/Escritorio/datos_recibidos/val_imnet.txt"
#path_to_bin = "/home/sebastian/Escritorio/datos_recibidos/imnet_4096_result.bin"

#evaluate_map(txt_data, path_to_bin, 4096, euclidean)
#print 1