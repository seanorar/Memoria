from video_processing import *

#print get_shot_time(10, "/home/sormeno/Desktop/videos/1/5339374493641274357.shots")
# shot_detection("/home/sormeno/Desktop/videos/1/","5339374493641274357.mp4", "video1", "/home/sormeno/Desktop/")
# r = get_shots_id("/home/sormeno/Desktop/videos/1/5339374493641274357.shots")
# save_frames("videos/1/5339374493641274357.mp4", r, "videos/1/shots/")

# r = get_shots_id("videos/2/5092591256270557686.shots")
# save_frames("videos/2/5092591256270557686.mp4", r, "videos/2/shots/")

# r = get_shots_id("videos/3/5154964489329981991.shots")
# save_frames("videos/3/5154964489329981991.mp4", r, "videos/3/shots/")

# r = get_shots_id("videos/4/5419885803088443114.shots")
# save_frames("videos/4/5419885803088443114.mp4", r, "videos/4/shots/")

prototxt = "/home/sormeno/py-faster-rcnn/models/imagenet/ZF/ZF_imnet.prototxt"
caffemodel = "/home/sormeno/py-faster-rcnn/models/imagenet/ZF/ZF_imnet.caffemodel"

get_bbox_from_videos("/home/sormeno/Datasets/Trecvid/orden_videos.txt", "/home/sormeno/test/", prototxt, caffemodel)
