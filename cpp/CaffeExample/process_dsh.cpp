#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cvision/caffe_predictor.h"
#include "jmsr/JUtil.h"
#include  <iostream>
#include <bits/stdc++.h>

using namespace std;

void get_features(string dataset_shots, string dataset_bbox, string data_out){
        //extraccion de features a partir de los bbox

        string str_pt("/home/sormeno/AlexNet/bvlc_alexnet_memory.prototxt");
        string str_caffemodel("/home/sormeno/AlexNet/bvlc_alexnet.caffemodel");
        CaffePredictor caffe_predictor(str_pt, str_caffemodel, 256, 256, CAFFE_GPU_MODE);
        string im_folder = dataset_shots;
        int des_size = 0;
        string im_name = ".jpg";
        cv::Mat img = cv ::imread(im_name);
        float * des_im = caffe_predictor.getCaffeDescriptor(img, &des_size, "fc7");
	for(int i = 0; i < des_size; i += 1){
		cout << des_im[i] << " ";
	}
	cout << endl;
}
