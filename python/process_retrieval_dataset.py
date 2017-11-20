import cv2
from code_m import get_dataset_bbox


def check_images(txt, path_images):
    with open(txt) as f:
        lines = f.readlines()
        for line in lines:
            img = cv2.imread(path_images + line.rstrip())
            if img is None:
                print "error con " + path_images + line.rstrip()

def generate_gt_relevant(list_calification, path, name, id):
    list_relevant = []
    for calification in list_calification:
        txt_data = path + name + "_" + str(id) + "_" + calification + ".txt"
        with open(txt_data) as f:
            lines = f.readlines()
            for line in lines:
                list_relevant.append(line.rstrip() + ".jpg")
    return list_relevant


def create_gt_evaluation(list_names, list_calification, txt_all_imgs, path_gt_files, output_dir):
    for name in list_names:
        for id in range(1,6):
            list_relevant = generate_gt_relevant(list_calification, path_gt_files, name, id)
            output_path  = output_dir + "gt_" + name + "_" + str(id) + ".txt"
            write_file = open(output_path, "w")
            with open(txt_all_imgs) as f:
                lines = f.readlines()
                for line in lines:
                    file_name = line.rstrip()
                    if list_relevant.__contains__(file_name):
                        write_file.write(file_name + " " + "1\n")
                    else:
                        write_file.write(file_name + " " + "0\n")
            write_file.close()


def extract_queries(list_names, path_all_img, path_gt_files, output_dir ):
    for name in list_names:
        for id in range(1, 6):
            query_txt = path_gt_files + name + "_" + str(id) + "_query.txt"
            with open(query_txt) as f:
                lines = f.readlines()
                for line in lines:
                    img_data = line.rstrip()
                    sep_data = img_data.split(" ")
                    img_name = sep_data[0]
                    img_bbox = [int(sep_data[1].split(".")[0]), int(sep_data[2].split(".")[0]),
                                int(sep_data[3].split(".")[0]), int(sep_data[4].split(".")[0])]
                    im = cv2.imread(path_all_img + img_name + ".jpg")
                    roi = im[img_bbox[1]:img_bbox[3], img_bbox[0]:img_bbox[2]]
                    cv2.imwrite(output_dir + name + "_" + str(id) + "_query.jpg", roi)


def extract_rois(path_imgs, txt_all_imgs, prototxt, caffemodel, mode, output_dir):
    list_images_full_path = []
    list_image_names = []
    with open(txt_all_imgs) as f:
        lines = f.readlines()
        for line in lines:
            list_images_full_path.append(path_imgs + line.rstrip())
            list_image_names.append(line.rstrip())
    result = get_dataset_bbox(list_images_full_path, prototxt, caffemodel, mode)
    output_path = output_dir + "detected_rois.txt"
    write_file = open(output_path, "w")
    for i in range (0, len(list_image_names)):
        for bbox in result[i]:
            to_write = str(int(bbox[0])) + " " + str(int(bbox[1])) + " " + str(int(bbox[2])) \
                       + " " + str(int(bbox[3])) + " " + list_image_names[i] + "\n"
            write_file.write(to_write)
    write_file.close()

