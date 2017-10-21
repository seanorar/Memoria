import glob
#import requests
#from pathlib import Path
from img_processing import zero_padding
from evaluate import *
import cv2
import os, shutil

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

def mod_xml(xml_path, size, outh_xml):
    new_xml = []
    with open(xml_path) as f:
        lines = f.readlines()
        for line in lines:
            if "<width>" in line:
                val = line.split(">")[1].split("<")[0]
                new_val = int(val) + size*2
                line_split = line.split(val)
                new_line = line_split[0] + str(new_val) + line_split[1]
            elif "<height>" in line:
                val = line.split(">")[1].split("<")[0]
                new_val = int(val) + size * 2
                line_split = line.split(val)
                new_line = line_split[0] + str(new_val) + line_split[1]
            elif "<xmin>" in line:
                val = line.split(">")[1].split("<")[0]
                new_val = int(val) + size
                line_split = line.split(val)
                new_line = line_split[0] + str(new_val) + line_split[1]
            elif "<xmax>" in line:
                val = line.split(">")[1].split("<")[0]
                new_val = int(val) + size
                line_split = line.split(val)
                new_line = line_split[0] + str(new_val) + line_split[1]
            elif "<ymin>" in line:
                val = line.split(">")[1].split("<")[0]
                new_val = int(val) + size
                line_split = line.split(val)
                new_line = line_split[0] + str(new_val) + line_split[1]
            elif "<ymax>" in line:
                val = line.split(">")[1].split("<")[0]
                new_val = int(val) + size
                line_split = line.split(val)
                new_line = line_split[0] + str(new_val) + line_split[1]
            else:
                new_line = line
            new_xml.append(new_line)
            with open(outh_xml, 'w') as file:
                for line in new_xml:
                    file.write(line)

def padding_xml_to_dataset(xml_dir_path, txt_path, output):
    with open(txt_path) as f:
        lines = f.readlines()
        num_files = len(lines)
        counter = 0.0
        for line in lines:
            counter += 1
            xml_name = line.split(" ")[0]
            mod_xml(xml_dir_path + xml_name + ".xml", 25, output + xml_name + ".xml")
            if (counter%1000==0):
                print str((counter/num_files)*100) + " %"
                

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

def separete_dataset(path_imgs, txt_train, txt_val, extension, output_dir):
    mode = [(txt_train, "train"),(txt_val, "val")]
    for element in mode:
        with open(element[0]) as f:
            directory = output_dir + element[1]
            if not os.path.exists(directory):
                os.makedirs(directory)
            lines = f.readlines()
            for line in lines:
                name = line.rstrip()
                old_img = path_imgs + name + "." + extension
                new_img = directory
                shutil.copy2(old_img, new_img)

def generate_clases_dict(txt_classes):
    output_dict = {}
    with open(txt_classes) as f:
        id = 0
        lines = f.readlines()
        for line in lines:
            class_name = line.rstrip()
            output_dict[class_name] = str(id)
            id += 1
        return output_dict

def extract_objects_from_dataset(path_imgs, path_xmls, txt_data, extension, output_dir, txt_classes):
    with open(txt_data) as f:
        id_image = 0
        classes_dict = generate_clases_dict(txt_classes)
        directory = output_dir + "objects"
        if not os.path.exists(directory):
            os.makedirs(directory)
        lines = f.readlines()
        txt_out1 = open(output_dir + 'output_1.txt', 'w')
        txt_out2 = open(output_dir + 'output_2.txt', 'w')
        for line in lines:
            name = line.rstrip()
            img = cv2.imread(path_imgs + name + extension)
            bboxs, clases = get_bbox_from_xml(path_xmls + name + ".xml", True)
            for i in range(0,len(bboxs)):
                bbox = bboxs[i]
                txt_out1.write(str(id_image) + " " + classes_dict[clases[i]] + "\n")
                txt_out2.write(str(id_image) + extension + "\n")
                roi = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                cv2.imwrite(directory + "/" + str(id_image) + extension, roi)
                id_image += 1
        txt_out1.close()
        txt_out2.close()