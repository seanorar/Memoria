from skimage import feature, io
import scipy
from scipy.spatial import distance
from new_demo import get_all_bbox, get_all_features, get_features_from_list
import cv2
import os
import numpy as np
import math
from PIL import Image
import caffe

#obtiene todos los frames de un video
def list_frames(video_name, fps, folder, min_limit = -1):
    cap = cv2.VideoCapture(video_name)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if (fps > fps_video):
        fps = fps_video
    contador = int(fps_video / fps)
    frame_contador = 0
    ret = True
    limit = min_limit * fps_video * 60
    while(ret):
        ret, frame = cap.read()
        if (contador == int(fps_video / fps)):
            contador = 0
            filename = "img_out_" + str(frame_contador) + ".jpg"
            frame_contador += 1
            cv2.imwrite(folder + filename, frame)
        contador += 1
        limit-=1
        if (limit==0):
            cap.release()
            return
    cap.release()

def lbp_uniform(image):
    desc = feature.local_binary_pattern(image, 4, 10, method = 'uniform')
    histogram = scipy.stats.itemfreq(desc)
    return histogram[:,1]

def multizone_lbp(image):
    h = len(image)
    w = len(image[0])
    descriptor = []
    descriptor.append(lbp_uniform(image[0:h / 2, 0:w / 2]))
    descriptor.append(lbp_uniform(image[h / 2:h, 0:w / 2]))
    descriptor.append(lbp_uniform(image[0:h / 2, w / 2:w]))
    descriptor.append(lbp_uniform(image[h / 2:h, w / 2:w]))
    return descriptor

def complete_lbp(image):
    desc = feature.local_binary_pattern(image,24, 3,  method = "uniform")
    histogram = scipy.stats.itemfreq(desc)
    return [histogram[:,1]]

def operate_dist_list1(list):
    total = sum(list)/len(list)
    return total

def get_lbp_distance(lbp1, lbp2):
    result = []
    for i in range (0, len(lbp1)):
        result.append(distance.euclidean(lbp1[i],lbp2[i]))
    return result

def shot_detection(path_images, n_images, out_shots):
    lbpi = complete_lbp(cv2.imread(path_images + str(0) + ".jpg", 0))
    umbral = 12000
    shot_contador = 0
    os.makedirs(out_shots + "shots_" + str(shot_contador))
    filename = "shots_" + str(shot_contador) + "/img_out_shot" + str(shot_contador) + str("_") + str(0) + ".jpg"
    r_shot = "img_out_shot" + str(shot_contador) + str("_") + str(0) + ".jpg"
    cv2.imwrite(out_shots + r_shot, cv2.imread(path_images + str(0) + ".jpg"))
    cv2.imwrite(out_shots + filename, cv2.imread(path_images + str(0) + ".jpg"))
    for i in range(1, n_images):
        lbpf = complete_lbp(cv2.imread(path_images + str(i) + ".jpg", 0))
        distancia = abs(operate_dist_list1(get_lbp_distance(lbpi, lbpf)))
        if (umbral < distancia):
            print i
            shot_contador += 1
            os.makedirs(out_shots + "shots_" + str(shot_contador))
            r_shot = "img_out_shot" + str(shot_contador) + str("_") + str(0) + ".jpg"
            cv2.imwrite(out_shots + r_shot, cv2.imread(path_images + str(i) + ".jpg"))
        filename = "shots_"+str(shot_contador)+"/img_out_shot" + str(shot_contador)+ str("_")+str(i) + ".jpg"
        cv2.imwrite(out_shots + filename, cv2.imread(path_images + str(i) + ".jpg"))
            #shot_element_count = 0
        #else:
            #cv2.imwrite(out_shots+"shots_" + str(shot_contador)+"/img_"+str(shot_element_count)+".jpg", imgs[i])
            #shot_element_count += 1
        lbpi = lbpf
    return shot_contador

def init_net(prototxt, caffemodel, mode="cpu"):
    if (mode=="gpu"):
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    return net


#calcula los bbox
def save_bbox(path,inicio,fin, path_prototxt, path_caffemodel, out, mode="cpu"):
    images = []
    for i in range(inicio, fin):
        im_name = path + str(i) + ".jpg"
        images.append(im_name)
    net = init_net(path_prototxt, path_caffemodel, mode)
    result = get_all_bbox(images,net)
    np.save(out + "data_names", images)
    np.save(out + "data_bbox", result)

def get_dataset_bbox(img_path_list, prototxt, caffemodel, mode = "cpu"):
    net = init_net(prototxt, caffemodel, mode)
    bboxs = get_all_bbox(img_path_list, net)
    return bboxs

def get_img_bbox(img_path, prototxt, caffemodel,mode ="cpu"):
    return get_dataset_bbox([img_path], prototxt, caffemodel, mode)[0]

def get_img_bbox2(img_path, net):
    bboxs = get_all_bbox([img_path], net)[0]
    return bboxs

def vis_img_bbox(img_path, prototxt, caffemodel,mode ="cpu"):
    img = cv2.imread(img_path)
    bboxs = get_img_bbox(img_path, prototxt, caffemodel,mode)
    rois = []
    for bbox in bboxs:
        roi = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        rois.append(roi)
    num_rois = len(rois)
    num_rows = int(math.ceil(num_rois/4))
    make_grid(rois, 5, num_rows)


def test_f():
    names = np.load("data_names.npy")
    bbox  = np.load("data_bbox.npy")
    result = get_all_features(names,bbox)
    #np.save("data_features", result)

#guarda las imagenes de los bbox
def save_bbox_images(data_names, data_bbox, out_path,  modo = 1):
    names = np.load(data_names)
    bbox  = np.load(data_bbox)
    if (modo ==1 ): # imagenes separadas en carpetas por shots
        out_path = out_path + "separated/"
        for i in range(0, len(names)):
            name_folder = out_path + str(i)
            im = cv2.imread(names[i])
            #cv2.imwrite(name_folder + ".jpg", im)
            os.makedirs(name_folder)
            for j in range(0, len(bbox[i])):
                roi = im[int(bbox[i][j][1]):int(bbox[i][j][3]), int(bbox[i][j][0]):int(bbox[i][j][2])]
                cv2.imwrite(name_folder + "/bbox_" + str(j) + ".jpg", roi)
    else:
        counter=0
        out_path = out_path + "all/"
        os.makedirs(out_path)
        for i in range(0, len(names)):
            name_folder = out_path
            im = cv2.imread(names[i])
            for j in range(0, len(bbox[i])):
                roi = im[int(bbox[i][j][1]):int(bbox[i][j][3]), int(bbox[i][j][0]):int(bbox[i][j][2])]
                cv2.imwrite(name_folder + "/bbox_" + str(counter) + ".jpg", roi)
                counter +=1

def test_feauture():
    list_images = []
    for i in range(0, 201):
        list_images.append("test/bbox_"+str(i)+".jpg")
    features, names = get_features_from_list(list_images)
    np.save("test/data_features", features)
    np.save("test/data_names", names)

def evaluate_distance(img_test, data):
    if (data==0):
        names = np.load("data_names1.npy")
        features = np.load("data_features1.npy")
        bbox = np.load("data_bbox1.npy")
    else:
        names = np.load("data_names_caffenet_1.npy")
        features = np.load("data_features_caffenet_1.npy")
        bbox = np.load("data_bbox_caffenet_1.npy")
    img_test = normalize_feature(img_test)
    list_result=[]
    for i in range(0, len(names)):
        print names[i]
        min_dist=[]
        for j in range(0, len(features[i])):
            comp = features[i][j]
            if (not math.isnan(comp[0])):
                comp = normalize_feature(features[i][j])
                dist = distance.euclidean(img_test, comp)
                min_dist.append((dist,bbox[i][j]))
            else:
                print "error"
        min_dist = sorted(min_dist, key=lambda x: x[0])
        min_bbox = []
        for k in range(0,2):
            min_bbox.append(min_dist[k])
        list_result.append((min_bbox[0][0], names[i], min_bbox))
            #list_result.append((dist, names[i]))

    result = sorted(list_result, key=lambda x: x[0])
    for i in range(0,10):
        aux_bbox = result[i][2]
        aux_dist = result[i][0]
        aux_name = result[i][1]
        im_aux = cv2.imread(aux_name)
        for s_bboxin in aux_bbox:
            cv2.rectangle(im_aux,(s_bboxin[1][0], s_bboxin[1][1]),(s_bboxin[1][2], s_bboxin[1][3]), (255, 0, 0), 4)
        cv2.imshow('dist = '+ str(aux_dist) ,im_aux )
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def normalize_feature(feature):
    total=0
    for i in range(0, len(feature)):
        sign = 0
        val = feature[i]
        if val >= 0:
            sign = 1
        else:
            sign = -1
        val = np.sqrt(abs(val))*sign
        feature[i]=val
    for i in range(0, len(feature)):
        total += feature[i] * feature[i]
    total = np.sqrt(total)
    return feature/total


def compare_roi(img_test, data):
    if (data==0):
        names = np.load("data_names_with_mean.npy")
        features = np.load("data_features_with_mean.npy")
        bbox = np.load("data_bbox_with_mean.npy")
    elif(data==1):
        names = np.load("data_names_caffenet_1.npy")
        features = np.load("data_features_caffenet_1.npy")
        bbox = np.load("data_bbox_caffenet_1.npy")
    else:
        names = np.load("data_namesc6_2.npy")
        features = np.load("data_featuresc6_2.npy")
        bbox = np.load("data_bboxc6_2.npy")
    img_test = normalize_feature(img_test)
    list_result = []

    for i in range(0,len(names)):
        print names[i]
        for j in range(0, len(features[i])):
            comp = features[i][j]
            if (not math.isnan(comp[0])):
                comp = normalize_feature(features[i][j])
                dist = distance.euclidean(img_test, comp)
                list_result.append((dist, names[i], bbox[i][j]))

    result = sorted(list_result, key = lambda x: x[0])
    print "se compararon "+str(len(result))+" rois"
    rois=[]
    n_resultados = len(result)
    for k in range(0, n_resultados):
        aux_bbox = result[k][2]
        aux_dist = result[k][0]
        aux_name = result[k][1]
        #print str(k)+ " -> " + str(aux_dist)
        im_aux = cv2.imread(aux_name)
        bbox_roi = im_aux[aux_bbox[0]:aux_bbox[2],aux_bbox[1]:aux_bbox[3]]
        #rois.append(bbox_roi)
        cv2.imwrite("all_rois_j/r_"+str(k)+".jpg", bbox_roi)
    #make_grid(rois,n_resultados, n_resultados)
        #im_aux2 = cv2.imread(aux_name)
        #cv2.rectangle(im_aux, (aux_bbox[0],aux_bbox[1]), (aux_bbox[2], aux_bbox[3]), (255, 0, 0), 4)
        #cv2.imwrite("prueba_rois/r_"+str(k)+".jpg", im_aux2[aux_bbox[0]:aux_bbox[2],aux_bbox[1]:aux_bbox[3]])

#5, mucho
def make_grid(img_list,size_x, size_y):
    new_im = Image.new('RGB', (200 * size_x, 200 * size_y))
    index = 0
    for i in xrange(0, 200 * size_y, 200):
        for j in xrange(0, 200 * size_x, 200):
            if(index < len(img_list)):
                cv2.imwrite("aux.jpg",img_list[index])
                im = Image.open("aux.jpg")
                im.thumbnail((200, 200))
                new_im.paste(im, (j, i))
                index += 1
    new_im.save("resultado.jpg")


def normalize_des():
    for i in range(0, 17000):
        print i
        im_name = "all_rois_j/r_" + str(i)
        comp = np.load(im_name + ".npy")
        comp = normalize_feature(comp)
        np.save("all_rois_j/r_n_" + str(i), comp)

def test_dist(feature):
    result_list = []
    feature = normalize_feature(feature)
    for i in range(0, 17000):
        im_name = "all_rois_j/r_"+str(i)
        im_name2 = "all_rois_j/r_n_" + str(i)
        comp = np.load(im_name2+".npy")
        dist = distance.euclidean(feature, comp)
        #print str(i) + " -> " + str(dist)
        result_list.append((dist, im_name + ".jpg"))
    result = sorted(result_list, key=lambda x: x[0])
    size = 10
    to_show = []
    for i in range(0, size * size):
        if result[i][0] > result[i + 1][0]:
            print "error"
            print i
        print result[i][1]
        to_show.append(cv2.imread(result[i][1]))
    make_grid(to_show, size, size)
