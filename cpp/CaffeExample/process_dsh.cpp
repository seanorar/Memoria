#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cvision/caffe_predictor.h"
#include "jmsr/JUtil.h"
#include  <iostream>
#include <bits/stdc++.h>

using namespace std;

void get_features(string imgs_path_folder, string imgs_txt, string data_out){
        //extraccion de features a partir de los bbox	
	string str_pt("/home/sormeno/Models/Alexnet-DSH/alex_hfc8_12.prototxt");
        string str_caffemodel("/home/sormeno/Models/Alexnet-DSH/alex_hfc8_12.caffemodel");

        CaffePredictor caffe_predictor(str_pt, str_caffemodel, 256, 256, CAFFE_GPU_MODE);

	ofstream writeFile (data_out, ios::out | ios::binary);
	ifstream file(imgs_txt);
	string str;
	while (getline(file, str)){
		cout << str << endl;
        	int des_size = 0;
        	cv::Mat img = cv ::imread(imgs_path_folder + str);
        	float * des_im = caffe_predictor.getCaffeDescriptor(img, &des_size, "hfc8");
		writeFile.write((char*) des_im, sizeof(float) * des_size);
	}
	writeFile.close();
}


cv::Mat get_feature(string name_query, int mode){
        //extracción de característica de una imagen
        string str_pt, str_caffemodel, layer_name;
	if (mode == 0){
       		str_pt = "/home/sormeno/Models/Alexnet-DSH/alex_hfc8_12.prototxt";
        	str_caffemodel = "/home/sormeno/Models/Alexnet-DSH/alex_hfc8_12.caffemodel";
        	layer_name = "hfc8";
	}
	else{

                str_pt = "/home/sormeno/Models/DSH/dsh.prototxt";
                str_caffemodel = "/home/sormeno/Models/DSH/dsh.caffemodel";
                layer_name = "ip1";
        }

        CaffePredictor caffe_predictor(str_pt, str_caffemodel, 256, 256, CAFFE_GPU_MODE);

        cv::Mat mat_image = cv::imread(name_query);
        JUtil::jmsr_assert(!mat_image.empty(), " image failed");
        int des_size = 0;
        float *des_consulta = caffe_predictor.getCaffeDescriptor(mat_image, &des_size, layer_name);
        cv::Mat mat_consulta = cv::Mat(1, des_size, CV_32F, des_consulta);
	for (int p = 0; p < des_size; p += 1){
		cout << mat_consulta.at<float>(0, p)<<"  ";
	}
	cout <<endl;
        return mat_consulta;
}


int main(int argc, char* argv[]){
	//string imgs_path_folder = "/home/sormeno/Datasets/VOC/val/val/";
	//string imgs_txt = "/home/sormeno/Datasets/VOC/val/val.txt";
	
	//string imgs_path_folder = "/home/sormeno/Datasets/imagenet/val/objects/";
        //string imgs_txt = "/home/sormeno/Datasets/imagenet/val/output_2.txt";

	//string data_out = "/home/sormeno/imnet_48_result.bin";
	//get_features(imgs_path_folder, imgs_txt, data_out);

	get_feature("/home/sormeno/Datasets/Pascal/Images/005349.jpg", 1);
	get_feature("/home/sormeno/Datasets/Pascal/Images/003008.jpg",1);
}


