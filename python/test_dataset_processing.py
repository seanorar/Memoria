from dataset_processing import *

"""
txt_path = "/home/sormeno/Datasets/Imagenet/ILSVRC13/data/det_lists/val.txt"
imgs_dir_path = "/home/sormeno/Datasets/Imagenet/ILSVRC2013_DET_val/"
xml_dir_path = "/home/sormeno/Datasets/Imagenet/ILSVRC2013_DET_bbox_val/"
output1  = "/home/sormeno/Datasets/Imagenet/val_f/"
output2  = "/home/sormeno/Datasets/Imagenet/val_xml_f/"
padding_to_dataset(imgs_dir_path, txt_path, output1)
padding_xml_to_dataset(xml_dir_path, txt_path, output2)
"""

"""
path_imgs = "/home/sormeno/Datasets/VOCdevkit/VOC2007/JPEGImages/"
txt_train = "/home/sormeno/Datasets/VOCdevkit/VOC2007/ImageSets/Layout/train.txt"
txt_val = "/home/sormeno/Datasets/VOCdevkit/VOC2007/ImageSets/Layout/val.txt"
extension = "jpg"
output_dir = "/home/sormeno/Datasets/VOCdevkit/VOC2007/"
separete_dataset(path_imgs, txt_train, txt_val, extension, output_dir)
"""

"""
#Pascal
path_imgs = "/home/sormeno/Datasets/VOCdevkit/VOC2007/JPEGImages/"
path_xmls = "/home/sormeno/Datasets/VOCdevkit/VOC2007/Annotations/"
txt_data =  "/home/sormeno/Datasets/VOCdevkit/VOC2007/ImageSets/Layout/val.txt"
extension = ".jpg"
output_dir ="/home/sormeno/Datasets/VOCdevkit/"
txt_classes ="/home/sormeno/Datasets/VOCdevkit/VOC2007/classes.txt"
extract_objects_from_dataset(path_imgs, path_xmls, txt_data, extension, output_dir, txt_classes)
"""

#"""
#Imagenet
extension = ".JPEG"
txt_classes ="/home/sormeno/Datasets/Imagenet/ILSVRC13/data/classes.txt"

#train
path_imgs = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_train/"
path_xmls = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_bbox_train/"
txt_data =  "/home/sormeno/Datasets/Imagenet/ILSVRC13/data/det_lists/train.txt"
output_dir ="/home/sormeno/Datasets/Imagenet/train/"

multiclass_est(txt_data, path_xmls)

#extract_objects_from_dataset(path_imgs, path_xmls, txt_data, extension, output_dir, txt_classes)

#val
path_imgs = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_val/"
path_xmls = "/home/sormeno/Datasets/Imagenet/ILSVRC13/ILSVRC2013_DET_bbox_val/"
txt_data =  "/home/sormeno/Datasets/Imagenet/ILSVRC13/data/det_lists/val.txt"
output_dir ="/home/sormeno/Datasets/Imagenet/val/"

#extract_objects_from_dataset(path_imgs, path_xmls, txt_data, extension, output_dir, txt_classes)

#"""

#check_files("/home/sormeno/Datasets/Imagenet2014/ILSVRC13/data/det_lists/train.txt", "/home/sormeno/Datasets/Imagenet2014/ILSVRC13/")
#get_files_to_txt("/home/sormeno/Datasets/Imagenet/join/", "/home/sormeno/")
#generate_train_set("/home/sebastian/Escritorio/output.txt", "/home/sebastian/Escritorio/output2.txt")
