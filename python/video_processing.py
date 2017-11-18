import cv2
from skimage import feature
import scipy
from scipy.spatial import distance
import xml.etree.ElementTree as ET


def operate_dist_list1(list):
    total = sum(list)/len(list)
    return total


def get_lbp_distance(lbp1, lbp2):
    result = []
    for i in range (0, len(lbp1)):
        result.append(distance.euclidean(lbp1[i],lbp2[i]))
    return result


def complete_lbp(image):
    desc = feature.local_binary_pattern(image,24, 3,  method = "uniform")
    histogram = scipy.stats.itemfreq(desc)
    return [histogram[:,1]]


def shot_detection(path_video, video_name, id_video, path_output):
    with open(path_output + 'output.shots', 'w') as file:
        cap = cv2.VideoCapture(path_video + video_name)
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        time_slap = 1.0 / fps_video
        ret , frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lbpi = complete_lbp(frame)
        umbral = 13000
        shot_contador = 0
        i = 1
        end = -1 
        while (ret):
            if (i == i):
                ret , frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lbpf = complete_lbp(frame)
                distancia = abs(operate_dist_list1(get_lbp_distance(lbpi, lbpf)))
                if (umbral < distancia):
                    print i
                    shot_contador += 1
                    #cv2.imwrite("/home/sormeno/Desktop/test/" + str(i) + ".jpg", frame)
		    video_id = id_video + "_" + str(shot_contador)
                    start = end + 1
                    end = i
                    time_start = start * time_slap
                    time_end = end * time_slap
                    file.write(video_id + "\t"+str(start) + "\t" + str(end) + "\t" + "vlc " + video_name + " --start-time=" + str(time_start) + " --stop-time=" + str(time_end)+"\n")
                lbpi = lbpf
                i += 1


def get_shots_id(shots_file):
    id_shots=[]
    fo = open(shots_file, "rw+")
    current_id = 0
    for line in fo.readlines():
        line_split = line.split("\t")
        shot_ini = int(line_split[1])
        shot_fin = int(line_split[2])
        id_shots.append((shot_ini, current_id))
        id_shots.append((int((shot_fin + shot_ini) / 2), current_id))
        id_shots.append((shot_fin, current_id))
        current_id += 1
    fo.close()
    return id_shot


def save_frames(video_path, list_id ,path_out):
    cap = cv2.VideoCapture(video_path)
    ret = True
    actual_index_shot = 0
    actual_index = 0
    n_shots = len(list_id)
    while(ret):
        ret, frame = cap.read()
        if (actual_index_shot < n_shots):
            if (list_id[actual_index_shot][0] == actual_index):
                filename = str(actual_index_shot) + ".jpg"
                cv2.imwrite(path_out + filename, frame)
                actual_index_shot += 1
            actual_index += 1
    cap.release()


def get_video_id_relevant_gt(gt_file, path_output):
    #obtiene los id y cantidad de shots relevantes que aparecen en el gt de trecvid
    ids_videos = [0] * 300
    with open(gt_file) as f:
        lines = f.readlines()
        for line in lines:
            query_id, video_info = line.rstrip().split("\t")
            video_name, shot_info = video_info.split("_")
            video_id = video_name.split("shot")[1]
            ids_videos[int(video_id)] += 1
    final_result = []
    for i in range (0, 250):
            final_result.append((i, ids_videos[i]))
    final_result.sort(key=lambda x:-x[1])
    with open(path_output, 'w') as file:
        for element in final_result:
            if element[1]!=0:
                file.write(str(element[0]) + " " + str(element[1]) + "\n")


def get_shot_time(shot_number, shot_info_file):
    with open(shot_info_file) as f:
        lines = f.readlines()
        info_shot = lines[shot_number-1]
        split_str = info_shot.split(" ")
        start_time = float(split_str[2].split("--start-time=")[1])
        end_time = float(split_str[3].split("--stop-time=")[1])
        return (start_time, end_time)


def obtain_videoID_videName(xml_path):
    dict_data={}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.iter('VideoFile'):
        video_id  = obj.find('id').text
        video_name = obj.find('filename').text
        dict_data[video_id] = video_name
    return dict_data


def get_shots_from_videos(txt_gt_videos, xml_path, video_path):
    with open(txt_gt_videos) as f:
        lines = f.readlines()
        dict_data = obtain_videoID_videName(xml_path)
        for line in lines:
            video_id = line.rstrip()
            print dict_data[video_id]
