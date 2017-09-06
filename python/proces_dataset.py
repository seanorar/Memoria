import glob
#import requests
from pathlib import Path
from img_processing import zero_padding
import cv2

def get_files_to_txt(path, path_out):
    file_names = glob.glob(path + "*")
    for i in xrange(len(file_names)):
        sep = file_names[i].split("/")
        temp_name = sep[len(sep)-1]
        sep = temp_name.split(".")
        file_names[i] = sep[0]
    file_names.sort()
    with open(path_out + 'output.txt', 'w') as file:
	index = 1
        for element in file_names:
            file.write(element +" "+ str(index) + "\n")
            index+=1


def load_200(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        clases = []
        for i in xrange(len(lines)):
            lines[i] = lines[i].rstrip()
            sep = lines[i].split(" ")
            clases.append(sep[0])
        return clases


def generate_train_set(txt_file, path_out):
    with open(txt_file) as f:
        lines = f.readlines()
        dict_clases = {}
        det_clases = load_200("/home/sebastian/Escritorio/imagenet_200.txt")
        for i in xrange(len(lines)):
            lines[i] = lines[i].rstrip()
            sep = lines[i].split("_")
            img_class = sep[0]
            if (img_class in det_clases):
                if (img_class in dict_clases):
                    list_r = dict_clases[img_class]
                    list_r.append(lines[i])
                    dict_clases[img_class] = list_r
                else:
                    dict_clases[img_class] = [lines[i]]

    keys = dict_clases.keys()
    with open(path_out, 'w') as file:
        for key in keys:
            for element in dict_clases[key]:
                file.write(element + "\n")

def get_img_from_url(image_url):
    img_data = requests.get(image_url).content
    name_split = image_url.split("/")
    img_name = name_split[len(name_split)-1]
    with open('/home/sebastian/Escritorio/img_instant_search/' + img_name, 'wb') as handler:
        handler.write(img_data)

def padding_to_dataset(imgs_dir_path, txt_path, output):
    with open(txt_path) as f:
        lines = f.readlines()
        num_files = len(lines)
        counter = 0.0
        for line in lines:
            counter +=1
            img_name = line.split(" ")[0]
            img = cv2.imread(imgs_dir_path+img_name+".JPEG")
            new_img = zero_padding(img, 25)
            cv2.imwrite(output+img_name+".JPEG", new_img)
            if (counter%100==0):
                print str(int((counter/num_files)*100)) + "%"

def check_files(path_txt, path_folder):
	with open(path_txt) as f:
        	lines = f.readlines()
		for line in lines:
			file_name = line.split(" ")[0]
			#print path_folder +"ILSVRC2013_DET_bbox_trai/" + file_name + ".xml"
			file_xml = Path(path_folder +"ILSVRC2013_DET_bbox_trai/" + file_name + ".xml")
			file_jpg = Path(path_folder +"ILSVRC2013_DET_trai/" + file_name + ".JPEG")
			if not (file_xml.is_file() and file_jpg.is_file()):
				print "error in " + file_name

def get_urls_imgs():
    with open("/home/sebastian/Escritorio/img_instant_search/imgs.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            url_img = "http://www-nlpir.nist.gov/projects/tv2013/pastdata/instance.search.topics/tv13.example.images/" + line
            print(url_img)
            get_img_from_url(url_img)


txt_path = "/home/sormeno/Datasets/Imagenet2014/ILSVRC13/data/det_lists/train.txt"
imgs_dir_path = "/home/sormeno/Datasets/Imagenet2014/ILSVRC13/ILSVRC2013_DET_trai/"
output  = "/home/sormeno/Datasets/Imagenet2014/ILSVRC13/ILSVRC2013_DET_trai_2/"
padding_to_dataset(imgs_dir_path, txt_path, output)
#check_files("/home/sormeno/Datasets/Imagenet2014/ILSVRC13/data/det_lists/train.txt", "/home/sormeno/Datasets/Imagenet2014/ILSVRC13/")
#get_files_to_txt("/home/sormeno/Datasets/Imagenet2014/ILSVRC13/train_xml/", "/home/sormeno/1_")
#generate_train_set("/home/sebastian/Escritorio/output.txt", "/home/sebastian/Escritorio/output2.txt")
