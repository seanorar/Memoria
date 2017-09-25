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

def test_nearest():
    with open('/home/sebastian/Escritorio/universidad/memoria/py-faster-rcnn/tools/test1.txt') as fp:
        list_result = []
        for line in fp:
            sep = line.split(" ")
            list_result.append([float(sep[0]),float(sep[1]),float(sep[2]),float(sep[3]),float(sep[4])])
        #a = [193155, 76381, 114099, 407478, 54976, 408466, 316518, 141563, 307375, 338004, 230746, 55435, 250558, 234960, 249911, 308003, 274674, 94335, 235960, 230301, 420313, 316841, 240690, 122814, 10633]
        #final_list = []
        final_list=[839, 323, 492, 1828, 228, 1833, 1404, 606, 1358, 1507, 1013, 230, 1100, 1031, 1098, 1360, 1216, 408, 1035, 1011, 1885, 1405, 1054, 528, 45]
        #for element in a:
        #    print list_result[element][4]
        #    final_list.append(list_result[element])
        for element in final_list:
            im  = cv2.imread("/home/sebastian/Escritorio/universidad/memoria/py-faster-rcnn/tools/videos/1/shots/"+ str(int(element-1)) + ".jpg")
            #cv2.rectangle(im, (int(element[0]),int(element[1])), (int(element[2]),int(element[3])),( 0, 255, 255 ))
            cv2.imshow('image', im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

generate_txt_bbox("/home/sormeno/Desktop/videos/1/data_bbox.npy", "/home/sormeno/Desktop/videos/1/test_s6_1.txt",3)
#generate_txt_bbox("/home/sormeno/Desktop/videos/2/data_bbox.npy", "/home/sormeno/Desktop/videos/2/bbox_2.txt",3)
#generate_txt_bbox("/home/sormeno/Desktop/videos/3/data_bbox.npy", "/home/sormeno/Desktop/videos/3/bbox_3.txt",3)
#generate_txt_bbox("/home/sormeno/Desktop/videos/4/data_bbox.npy", "/home/sormeno/Desktop/videos/4/bbox_4.txt",3)
#test_nearest()
