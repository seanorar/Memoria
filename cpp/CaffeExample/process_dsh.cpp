#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cvision/caffe_predictor.h"
#include "jmsr/JUtil.h"
#include  <iostream>
#include <bits/stdc++.h>

using namespace std;

void get_features(string imgs_path_folder, string imgs_txt, string data_out){
        //extraccion de features a partir de los bbox
        string str_pt("/home/sormeno/Models/DSH/dsh_ft.prototxt");
        string str_caffemodel("/home/sormeno/Models/DSH/dsh_ft.caffemodel");
        CaffePredictor caffe_predictor(str_pt, str_caffemodel, 256, 256, CAFFE_GPU_MODE);

	ofstream writeFile (data_out, ios::out | ios::binary);
	ifstream file(imgs_txt);
	string str;
	while (getline(file, str)){
		cout << str << endl;
        	int des_size = 0;
        	cv::Mat img = cv ::imread(imgs_path_folder + str);
        	float * des_im = caffe_predictor.getCaffeDescriptor(img, &des_size, "ip1_f");
		writeFile.write((char*) des_im, sizeof(float) * des_size);
	}
	writeFile.close();
}

int main(int argc, char* argv[]){
	string imgs_path_folder = "/home/sormeno/Datasets/VOC/val/val/";
	string imgs_txt = "/home/sormeno/Datasets/VOC/val/val.txt";
	string data_out = "/home/sormeno/dsh_ft_result.bin";
	get_features(imgs_path_folder, imgs_txt, data_out);
}
