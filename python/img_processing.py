import cv2
import numpy as np

def change_background(img_name, img_out):
    image = cv2.imread(img_name)
    image[np.where((image == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    cv2.imwrite(img_out, image)


def save_bbox_from_frame(id_frame, bbox_data, path_imgs, path_out):
    img = cv2.imread(path_imgs + str(id_frame) + ".jpg")
    with open(bbox_data, "r") as myfile:
        id_contador = 0
        for line in myfile:
            sep = line.split(" ")
            if (id_frame == float(sep[5])):
                bbox_data = [float(sep[0]), float(sep[1]), float(sep[2]), float(sep[3])]
                new_img = img[int(bbox_data[1]):int(bbox_data[3]), int(bbox_data[0]):int(bbox_data[2])]
                cv2.imwrite(path_out + str(id_contador) + ".jpg", new_img)
                id_contador += 1


def get_bbox_from_masc(img_masc):
    xmax = ymax = 0
    xmin = ymin = 10000
    w,h = img_masc.shape
    for i in range(0,w):
        for j in range(0, h):
            val = img_masc[i][j]
            if (val == 255):
                if (xmax < j):
                    xmax = j
                if (ymax < i):
                    ymax = i
                if (xmin > j):
                    xmin = j
                if (ymin > i):
                    ymin = i
    return [xmin, ymin, xmax, ymax]

def get_roi_from_masc(img_masc, img_orig):
    bbox = get_bbox_from_masc(img_masc)
    new_img = img_orig[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return new_img

def apply_mask(img, mask):
    img2gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    img2_fg = cv2.bitwise_and(img, img, mask=mask)
    tmp = cv2.cvtColor(img2_fg, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(img2_fg)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    cv2.imwrite("test.png", dst)

def zero_padding(img, sp):
    result = cv2.copyMakeBorder(img, sp,sp,sp,sp , cv2.BORDER_CONSTANT)
    return  result

def process_consultas():
    id_list = [9070, 9076, 9086, 9101, 9103, 9112]
    for id in id_list:
        for i in range(1,5):
            im1 = cv2.imread("videos/objetos/c/" + str(id) + "." + str(i) + ".mask.png", 0)
            im2 = cv2.imread("videos/objetos/c/" + str(id) + "." + str(i) + ".src.png", 1)
            im = get_roi_from_masc(im1, im2)
            cv2.imwrite("videos/objetos/r/" + str(id) + "." + str(i) + ".src.png", im)


#process_consultas()
#save_bbox_from_frame(5507,
#                     "test_vgg_2.txt",
#                     "/home/sebastian/Escritorio/universidad/memoria/py-faster-rcnn/tools/videos/2/shots/",
#                     "/home/sebastian/Escritorio/vis/")
#img = cv2.imread("/home/sebastian/Escritorio/img_instant_search/9069.4.src.bmp")
#mask = cv2.imread("/home/sebastian/Escritorio/img_instant_search/9069.4.mask.bmp")
#apply_mask(img, mask)
