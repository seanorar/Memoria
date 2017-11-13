from process_retrieval_dataset import *

list_names = ["defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay", "notredame", "pantheon", "pompidou", "sacrecoeur", "triomphe"]
list_calification = ["good", "junk", "ok"]

txt_all_imgs = "/home/sebastian/Escritorio/Datasets/Paris/paris.txt"
path_gt = "/home/sebastian/Escritorio/Datasets/Paris/gt/"
output_dir = "/home/sebastian/Escritorio/Datasets/Paris/"
path_imgs= "/home/sebastian/Escritorio/Datasets/Paris/imgs/"

create_gt_evaluation(list_names, list_calification, txt_all_imgs, path_gt, output_dir)
extract_queries(list_names,path_imgs, path_gt,output_dir)

