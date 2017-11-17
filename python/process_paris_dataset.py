from process_retrieval_dataset import *

list_names = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame", "pantheon", "pompidou", "sacrecoeur", "triomphe"]
list_calification = ["good", "ok"]

txt_all_imgs = "/home/sormeno/Datasets/Paris/paris.txt"
path_gt = "/home/sormeno/Datasets/Paris/gt/"
output_dir = "/home/sormeno/Datasets/Paris/"
path_imgs= "/home/sormeno//Datasets/Paris/imgs/"

create_gt_evaluation(list_names, list_calification, txt_all_imgs, path_gt, output_dir)
#extract_queries(list_names,path_imgs, path_gt,output_dir)

#prototxt = "/home/sormeno/py-faster-rcnn/models/imagenet/ZF/ZF_ILSVRC.prototxt"
#caffemodel = "/home/sormeno/py-faster-rcnn/models/imagenet/ZF/ZF_ILSVRC_170W_600_31_0.v2.caffemodel"
#path_imgs = "/home/sormeno/Datasets/Paris/imgs/"
#txt = "/home/sormeno/Datasets/Paris/paris.txt"

#extract_rois(path_imgs, txt, prototxt, caffemodel, "gpu", "/home/sormeno/Datasets/Paris/")

