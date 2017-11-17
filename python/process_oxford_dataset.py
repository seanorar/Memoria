from process_retrieval_dataset import *

list_names = ["all_souls", "ashmolean", "balliol", "bodleian", "christ_church", "cornmarket", "hertford", "keble", "magdalen", "pitt_rivers", "radcliffe_camera"]
list_calification = ["good", "ok"]

txt_all_imgs = "/home/sormeno/Datasets/Oxford/oxford.txt"
path_gt = "/home/sormeno/Datasets/Oxford/gt/"
output_dir = "/home/sormeno/Datasets/Oxford/"
path_imgs= "/home/sormeno//Datasets/Oxford/imgs/"

create_gt_evaluation(list_names, list_calification, txt_all_imgs, path_gt, output_dir)

prototxt = "/home/sormeno/py-faster-rcnn/models/imagenet/ZF/ZF_ILSVRC.prototxt"
caffemodel = "/home/sormeno/py-faster-rcnn/models/imagenet/ZF/ZF_ILSVRC_170W_600_31_0.v2.caffemodel"
path_imgs = "/home/sormeno/Datasets/Oxford/imgs/"
txt = "/home/sormeno/Datasets/Oxford/oxford.txt"

#extract_rois(path_imgs, txt, prototxt, caffemodel, "gpu", "/home/sormeno/Datasets/Oxford/")