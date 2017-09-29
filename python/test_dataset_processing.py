from dataset_processing import *

txt_path = "/home/sormeno/Datasets/Imagenet/ILSVRC13/data/det_lists/val.txt"
imgs_dir_path = "/home/sormeno/Datasets/Imagenet/ILSVRC2013_DET_val/"
xml_dir_path = "/home/sormeno/Datasets/Imagenet/ILSVRC2013_DET_bbox_val/"
output1  = "/home/sormeno/Datasets/Imagenet/val_f/"
output2  = "/home/sormeno/Datasets/Imagenet/val_xml_f/"
#padding_to_dataset(imgs_dir_path, txt_path, output1)
padding_xml_to_dataset(xml_dir_path, txt_path, output2)

#check_files("/home/sormeno/Datasets/Imagenet2014/ILSVRC13/data/det_lists/train.txt", "/home/sormeno/Datasets/Imagenet2014/ILSVRC13/")
#get_files_to_txt("/home/sormeno/Datasets/Imagenet/join/", "/home/sormeno/")
#generate_train_set("/home/sebastian/Escritorio/output.txt", "/home/sebastian/Escritorio/output2.txt")
