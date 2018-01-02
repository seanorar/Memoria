import numpy as np
import cv2

def npy_to_txt(name_file_npy, name_file_txt):
    npy_data = np.load(name_file_npy)
    np.savetxt(name_file_txt, npy_data)

def generate_txt_bbox(bbox_file_name, result_name, im_per_shot = 1):
    bbox_data = np.load(bbox_file_name)
    result=[]
    for i in range(0, len(bbox_data)):
        for j in range(0, len(bbox_data[i])):
            bbox = bbox_data[i][j]
            bbox_f = [bbox[0], bbox[1], bbox[2], bbox[3], int(i/im_per_shot), i]
            result.append(bbox_f)
    np.savetxt(result_name, result) 

