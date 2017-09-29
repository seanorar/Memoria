from code_m import *
#video_name = "/home/sebastian/py-faster-rcnn/tools/videos/1/5339374493641274357.mp4"
#list_frames(video_name, 2, "frames2/")
#path = "/home/sebastian/Escritorio/universidad/memoria/codigo/python/frames2/img_out_"
#n_shots = shot_detection(path, 14920, "shots/")

path_prototxt = "/home/sebastian/Escritorio/universidad/memoria/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt"
path_caffemodel = "/home/sebastian/Escritorio/universidad/memoria/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel"

save_bbox("/home/sebastian/Escritorio/universidad/memoria/py-faster-rcnn/tools/videos/1/shots/", 1850, 1851, path_prototxt, path_caffemodel,"/home/sebastian/Escritorio/")
#save_bbox_images("prueba_rois/data_names.npy", "prueba_rois/data_bbox.npy", "prueba_rois/", 0)

#save_bbox("/home/sormeno/Desktop/videos/1/shots/", 0, 5924, path_prototxt, path_caffemodel, "/home/sormeno/Desktop/videos/1/", mode)
#save_bbox("/home/sormeno/Desktop/videos/2/shots/", 0, 6111, "/home/sormeno/Desktop/videos/2/")
#save_bbox("/home/sormeno/Desktop/videos/3/shots/", 0, 6333, "/home/sormeno/Desktop/videos/3/")
#save_bbox("/home/sormeno/Desktop/videos/4/shots/", 0, 9057, "/home/sormeno/Desktop/videos/4/")

#lista = [0, 13296, 1941, 338, 4747, 13484, 586, 11711, 18310, 18695, 9329, 13583, 18241, 1 5548, 517, 18599, 18200, 38, 16883, 191, 14803, 8115, 1967, 18750, 17173]
#lista_f = []
#for element in lista:
#    a = cv2.imread("all_rois_im/r_"+str(element)+".jpg")
#    lista_f.append(a)
#make_grid(lista_f,4,4)

#normalize_des()
#feature = np.load("consultas/c6.npy")
#evaluate_distance(feature, 0)
#feature = np.load("consultas/c11_caffe.npy")
#evaluate_distance(feature, 1)
#compare_roi(feature, 0)

#test_dist(feature)