from evaluate import *

#prototxt_i = "/home/sormeno/py-faster-rcnn/models/imagenet/VGG16/faster_rcnn_end2end/test.prototxt"
#caffemodel_i = "/home/sormeno/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_imagenet.caffemodel"
prototxt_i = "/home/sormeno/Desktop/ZF_ILSVRC.prototxt"
caffemodel_i = "/home/sormeno/Desktop/ZF_ILSVRC_170W_600_31_0.v2.caffemodel"

prototxt_p =  "/home/sormeno/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt"
caffemodel_p = "/home/sormeno/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel"

#imagenet
#-------------------------------------------------------------------------
"""
txt_data = "/home/sormeno/Datasets/Imagenet/ILSVRC13/data/det_lists/val.txt"
path_imgs = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_val/"
path_xmls = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_bbox_val/"

data_to_graphs(txt_data, path_imgs, path_xmls, prototxt_p, caffemodel_p, "gpu", "/home/sormeno/imagenet_pascal")
data_to_graphs(txt_data, path_imgs, path_xmls, prototxt_i, caffemodel_i, "gpu", "/home/sormeno/imagenet_imagenet")
"""

#pascal
#---------------------------------------------------------------------------
"""
txt_data = "/home/sormeno/Datasets/Pascal/val.txt"
path_imgs = "/home/sormeno/Datasets/Pascal/Images/"
path_xmls = "/home/sormeno/Datasets/Pascal/xmls/"

data_to_graphs(txt_data, path_imgs, path_xmls, prototxt_p, caffemodel_p, "gpu", "/home/sormeno/pascal_pascal")
data_to_graphs(txt_data, path_imgs, path_xmls, prototxt_i, caffemodel_i, "gpu", "/home/sormeno/pascal_imagenet")
"""

#dataset
#---------------------------------------------------------------------------
"""
txt_data = "/home/sormeno/Desktop/videos/1/val.txt"
path_imgs = "/home/sormeno/Desktop/videos/1/shots/"
path_xmls = "/home/sormeno/Desktop/videos/1/bbox_data.txt"

data_to_graphs(txt_data, path_imgs, path_xmls, prototxt_p, caffemodel_p, "gpu", "/home/sormeno/mdata_pascal_1")
data_to_graphs(txt_data, path_imgs, path_xmls, prototxt_i, caffemodel_i, "gpu", "/home/sormeno/mdata_imagenet_1")
"""
#plot_data_vs_trsh("mdata",2)

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

#bbox_val_imagenet("/home/sormeno/Datasets/Imagenet/ILSVRC2014_devkit/data/det_lists/", "/home/sormeno/Datasets/Imagenet/ILSVRC2013_DET_bbox_val/", "/home/sormeno/Datasets/Imagenet/ILSVRC2013_DET_val/")