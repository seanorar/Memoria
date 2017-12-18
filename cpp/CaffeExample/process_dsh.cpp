#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cvision/caffe_predictor.h"
#include "jmsr/JUtil.h"
#include  <iostream>
#include <bits/stdc++.h>

using namespace std;


void normalize(float lista[], int len){

        float total = 0;
        for (int i = 0; i < len; i += 1){
                float sign = 0;
                float val = lista[i];
                if (val > 0){
                        sign = 1;
                }
                else{
                        sign = -1;
                }
                val = sqrt(abs(val)) * sign;
                lista[i] = val;
        }
        for (int i =0; i< len; i+=1){
                total += (lista[i] * lista[i]);
        }

        total = sqrt(total);
        for (int i = 0; i < len; i += 1){
                lista[i] = lista[i]/total;
        }
}


void get_features(string imgs_path_folder, string imgs_txt, string data_out, int mode){
        //extraccion de features a partir de los bbox	
	string str_pt, str_caffemodel, layer_name;
	int size_mean;
	if (mode == 0){
                str_pt = "/home/sormeno/Models/Alexnet-DSH/alex_hfc8_256_2.prototxt";
                str_caffemodel = "/home/sormeno/Models/Alexnet-DSH/alex_hfc8_256_2.caffemodel";
                layer_name = "hfc8";
		size_mean = 224;
        }
        else{
                str_pt = "/home/sormeno/Models/Alexnet/bvlc_alexnet_memory.prototxt";
                str_caffemodel = "/home/sormeno/Models/Alexnet/bvlc_alexnet.caffemodel";
                layer_name = "fc7";
		size_mean = 256;
        }

        CaffePredictor caffe_predictor(str_pt, str_caffemodel, size_mean, size_mean, CAFFE_GPU_MODE);

	ofstream writeFile (data_out, ios::out | ios::binary);
	ifstream file(imgs_txt);
	string str;
	while (getline(file, str)){
		cout << str << endl;
        	int des_size = 0;
        	cv::Mat img = cv ::imread(imgs_path_folder + str);
        	float * des_im = caffe_predictor.getCaffeDescriptor(img, &des_size, layer_name);
		if (mode ==1){
			normalize(des_im, des_size);
		}
		writeFile.write((char*) des_im, sizeof(float) * des_size);
	}
	writeFile.close();
}


cv::Mat get_feature(string name_query, int mode){
        //extracción de característica de una imagen
        string str_pt, str_caffemodel, layer_name;
	int size_mean;
	if (mode == 0){
       		str_pt = "/home/sormeno/Models/Alexnet-DSH/alex_hfc8_256_2.prototxt";
        	str_caffemodel = "/home/sormeno/Models/Alexnet-DSH/alex_hfc8_256_2.caffemodel";
        	layer_name = "hfc8";
		size_mean = 224;
	}
	else{
		str_pt = "/home/sormeno/Models/Alexnet/bvlc_alexnet_memory.prototxt";
                str_caffemodel = "/home/sormeno/Models/Alexnet/bvlc_alexnet.caffemodel";
                layer_name = "fc7";
		size_mean = 256;
	}
        CaffePredictor caffe_predictor(str_pt, str_caffemodel, size_mean, size_mean, CAFFE_GPU_MODE);

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
	
	string imgs_path_folder = "/home/sormeno/Datasets/imagenet/train/objects/";
        string imgs_txt = "/home/sormeno/Datasets/imagenet/train/output_2.txt";

	string data_out = "/home/sormeno/data/ndata/imnet_train_4096.bin";
	get_features(imgs_path_folder, imgs_txt, data_out, 1);

	//get_feature("/home/sormeno/Datasets/Pascal/Images/005349.jpg", 0);
	//get_feature("/home/sormeno/Datasets/Pascal/Images/003008.jpg", 0);
}


