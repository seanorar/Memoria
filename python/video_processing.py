import cv2
import numpy as np

def get_shots_id(shots_file):
    id_shots=[]
    fo = open(shots_file, "rw+")
    current_id = 0
    for line in fo.readlines():
        line_split = line.split("\t")
        shot_ini = int(line_split[1])
        shot_fin = int(line_split[2])
        id_shots.append((shot_ini, current_id))
        id_shots.append((int((shot_fin+shot_ini)/2), current_id))
        id_shots.append((shot_fin, current_id))
        current_id+=1
    fo.close()
    return id_shots

def list_frames(video_path, list_id ,path_out):
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


#r = get_shots_id("videos/1/5339374493641274357.shots")
#list_frames("videos/1/5339374493641274357.mp4", r, "videos/1/shots/")

#r = get_shots_id("videos/2/5092591256270557686.shots")
#list_frames("videos/2/5092591256270557686.mp4", r, "videos/2/shots/")

#r = get_shots_id("videos/3/5154964489329981991.shots")
#list_frames("videos/3/5154964489329981991.mp4", r, "videos/3/shots/")

#r = get_shots_id("videos/4/5419885803088443114.shots")
#list_frames("videos/4/5419885803088443114.mp4", r, "videos/4/shots/")
