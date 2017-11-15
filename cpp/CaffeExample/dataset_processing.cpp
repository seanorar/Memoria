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

void proces_line_bbox_txt(string line, string buffer[]){

        istringstream iss(line);
        int index = 0;
        for(string s; iss >> s; ){
                buffer[index] = s;
                index += 1;
        }
}

long num_lines(string file_name){

        string line;
        ifstream f(file_name);
        long i;
        for (i = 0; getline(f, line); ++i)
        ;

        return i;
}

vector <string> split(string str, char delimiter) {

        vector<string> internal;
        stringstream ss(str);
        string tok;
        while(getline(ss, tok, delimiter)) {
                internal.push_back(tok);
        }

        return internal;
}


double hammming_distance(cv::Mat f1, cv::Mat f2){
        double len = f1.cols;
        double total = 0.0;
        for (int i = 0; i < len; i+=1){
                if (f1.at<float>(0,i) != f2.at<float>(0, i)){
                        total += 1.0;
                }
        }
        return (total / len);
}

cv::Mat feature_sign(cv::Mat feature){
        int len = feature.cols;
        cv::Mat result = cv::Mat::zeros(1,len,CV_32F);
        for (int i =0; i <len ; i += 1){
                if (feature.at<float>(0,i) > 0 ){
                        result.at<float>(0,i) = 1;
                }
        }
        return result;
}

void get_features_dataset(string prototxt, string caffemodel, int img_size, string layer_name, string imgs_folder, string bbox_txt, string data_out, bool normed = false){
        //extraccion de features a partir de los bbox

        CaffePredictor caffe_predictor(prototxt, caffemodel, img_size, img_size, CAFFE_GPU_MODE);
        ofstream writeFile (data_out, ios::out | ios::binary);
        ifstream file(bbox_txt);
        string str;
        int des_size = 0;
        string actual = "";
        long n_total = 0;
        while (getline(file, str)){
                string buffer[5];
                proces_line_bbox_txt(str, buffer);
                if (actual != buffer[4]){
                        cout << buffer[4] << endl;
                        actual = buffer[4];
                }
                string im_name = imgs_folder  + buffer[4];
                cv::Mat aux_image = cv ::imread(im_name);
                cv::Rect r(stoi(buffer[0]),stoi(buffer[1]), (int)(stoi(buffer[2])-stoi(buffer[0])), (int) (stoi(buffer[3])-stoi(buffer[1])));
                cv::Mat im_bbox(aux_image, r);
                float * des_im = caffe_predictor.getCaffeDescriptor(im_bbox, &des_size, layer_name);
                if (normed){
                        normalize(des_im, des_size);
                }
                writeFile.write((char*) des_im, sizeof(float) * des_size);
                n_total += 1;
        }
        writeFile.close();
        cout<< "n rois =  "<< n_total << endl;
}

vector<vector<string>> compare_datasat_features(vector <cv::Mat> mat_consultas, string feature_data, string bbox_data, bool is_binary, bool all_elements = true){
        //función que a partir de un descriptor, entrega una lista de los elementos semejantes ordenados desde el más cercano
        int feature_len = mat_consultas.at(0).cols;
        vector <pair <double, vector <string>>> dist_list;
        float result [feature_len];
        ifstream readFile (feature_data, ios::in | ios::binary);
        ifstream file(bbox_data);
        string str;
        long limit = num_lines(bbox_data);
        for(int i = 0; i < limit; i += 1){
                getline(file, str);
                string buffer[5];
                proces_line_bbox_txt(str, buffer);
                string im_name = buffer[4];
                readFile.read ((char*)result, sizeof(float) * feature_len);
                cv::Mat mat_im = cv::Mat(1, feature_len, CV_32F, result);
                int n_vectores = mat_consultas.size();
                for(int k = 0; k < n_vectores; k += 1){
                        cv::Mat mat_consulta = mat_consultas.at(k);
                        double dist = 0;
                        if (is_binary){
                                dist = hammming_distance(feature_sign(mat_consulta), feature_sign(mat_im));
                        }
                        else{
                                dist = cv::norm(mat_consulta, mat_im);
                        }
			vector <string> v (buffer, buffer + sizeof buffer / sizeof buffer[0]);
                        dist_list.push_back(make_pair(dist , v ));
                }
        }
        file.close();
        readFile.close();
        sort(dist_list.begin(),dist_list.end());
        vector <vector<string>> final_result;
	vector <string> check_names;
        int len_dist_list = dist_list.size();
        if (all_elements){
                for(int i = 0; i < len_dist_list; i += 1){
                        final_result.push_back(dist_list.at(i).second);
                }
        }
        else{
		for(int i = 0; i < len_dist_list; i += 1){
                	if (!(find(check_names.begin(), check_names.end(), dist_list.at(i).second.at(4)) != check_names.end())){
                        	final_result.push_back(dist_list.at(i).second);
				check_names.push_back( dist_list.at(i).second.at(4));
                	}
		}
        }
        return final_result;
}

cv::Mat get_feature_dataset(string prototxt, string caffemodel, int img_size, string layer_name, string img_name,  bool normed = false){
        //extraccion de features a partir de los bbox

        CaffePredictor caffe_predictor(prototxt, caffemodel, img_size, img_size, CAFFE_GPU_MODE);
        int des_size = 0;
        cv::Mat img = cv ::imread(img_name);
        float * des_im = caffe_predictor.getCaffeDescriptor(img, &des_size, layer_name);
        if (normed){
        	normalize(des_im, des_size);
	}
	cv::Mat mat_consulta = cv::Mat(1, des_size, CV_32F, des_im);
        return mat_consulta;
}


void save_imgs(vector <vector<string>> list_images, int n, string imgs_folder, string path_out, bool extract_roi = true){
	for (int i = 0; i < n ; i += 1){
		if (extract_roi){
			vector <string> buffer = list_images.at(i);
			cv::Mat aux_image = cv :: imread(imgs_folder + buffer.at(4));
                	cv::Rect r(stoi(buffer.at(0)), stoi(buffer.at(1)), stoi(buffer.at(2)) - stoi(buffer.at(0)), stoi(buffer.at(3)) - stoi(buffer.at(1)));
                	cv::Mat img(aux_image, r);
			cv::imwrite(path_out + to_string(i) + ".jpg", img);
		}
		else{
			cv::Mat img = cv ::imread(imgs_folder + list_images.at(i).at(4));
			cv::imwrite(path_out + to_string(i) + ".jpg", img);
		}
	}
}

vector <string> get_relevant(string txt_gt){
	vector <string> result;
	ifstream file(txt_gt);
	string str;
	while (getline(file, str)){
		vector <string> sep = split(str, ' ');
		if (sep.at(1)=="1"){
			result.push_back(sep.at(0));
		}
	}
	return result;
}

bool is_relevant(string name, vector <string> relevant_list){
	return find(relevant_list.begin(), relevant_list.end(), name) != relevant_list.end();
}

float eval_ap(vector <vector<string>> result, string gt_file){
	float ap = 0.0;
	float num_relevant = 0.0;
	vector<string> relevant_list = get_relevant(gt_file);
	for (int i = 0; i < result.size() ; i += 1){
		if (is_relevant(result.at(i).at(4), relevant_list)){
			num_relevant += 1;
			ap += (num_relevant/(i+1));
			cout << "aparece en " << i+1 << endl;
		}
	
	}
	cout << "ap = "<< ap/num_relevant  << endl;
	return ap/num_relevant;
}

float eval_dataset_map(vector <string> names, string path_queries, string path_gt, string prototxt, string caffemodel, int img_size, string layer_name,
		      string imgs_folder, string bbox_txt, string data_bin, bool normed){
	float map = 0;
	float num_elem = 0;
	for (int i = 0; i < names.size(); i += 1){
		for (int j = 1; j < 3; j += 1){
			string query = path_queries + names.at(i) + "_" + to_string(j) + "_query.jpg";
			string gt = path_gt + "gt_" + names.at(i) + "_"  +to_string(j) + ".txt";
			
			cv::Mat img_feature = get_feature_dataset(prototxt, caffemodel, img_size, layer_name, query, normed);
			vector <cv::Mat> mat_consultas;
        		mat_consultas.push_back(img_feature);
			
			vector <vector<string>> result = compare_datasat_features(mat_consultas, data_bin, bbox_txt, !normed, false);
			float ap = eval_ap(result, gt);
			map += ap;
			num_elem += 1;
		}
	}
	cout << "map  = " << (map / num_elem) << endl; 
	return (map / num_elem);
}

void extract_features_Oxford(string prototxt, string caffemodel, int img_size, string layer_name, string data_out, bool normed){
	string imgs_folder = "/home/sormeno/Datasets/Oxford/imgs/";
        string bbox_txt = "/home/sormeno/Datasets/Oxford/detected_rois.txt";
	get_features_dataset(prototxt, caffemodel, img_size, layer_name, imgs_folder, bbox_txt, data_out, normed);
}

void extract_features_Paris(string prototxt, string caffemodel, int img_size, string layer_name, string data_out, bool normed){
        string imgs_folder = "/home/sormeno/Datasets/Paris/imgs/";
        string bbox_txt = "/home/sormeno/Datasets/Paris/detected_rois.txt";
        get_features_dataset(prototxt, caffemodel, img_size, layer_name, imgs_folder, bbox_txt, data_out, normed);
}


float eval_Oxford(string prototxt, string caffemodel, int img_size, string layer_name, string data_bin, bool normed){
        string list_names [] = {"all_souls"};
//, "ashmolean", "balliol", "bodleian", "christ_church", "cornmarket",
//                                "hertford", "keble", "magdalen", "pitt_rivers", "radcliffe_camera"};
        
	vector <string> v (list_names, list_names + sizeof list_names / sizeof list_names[0]);
        string imgs_folder = "/home/sormeno/Datasets/Oxford/imgs/";
        string bbox_txt = "/home/sormeno/Datasets/Oxford/detected_rois.txt";
        string path_queries = "/home/sormeno/Datasets/Oxford/queries/";
        string path_gt = "/home/sormeno/Datasets/Oxford/gt_data/";
        float r = eval_dataset_map(v, path_queries, path_gt, prototxt, caffemodel, img_size, layer_name, imgs_folder, bbox_txt, data_bin, normed);
	return r;
}


float eval_Paris(string prototxt, string caffemodel, int img_size, string layer_name, string data_bin, bool normed){
        string list_names [] = {"defense", "eiffel", "invalides", "louvre", "moulinrouge", "museedorsay",
                                "notredame", "pantheon", "pompidou", "sacrecoeur", "triomphe"};

        vector <string> v (list_names, list_names + sizeof list_names / sizeof list_names[0]);
        string imgs_folder = "/home/sormeno/Datasets/Paris/imgs/";
        string bbox_txt = "/home/sormeno/Datasets/Paris/detected_rois.txt";
        string path_queries = "/home/sormeno/Datasets/Paris/queries/";
        string path_gt = "/home/sormeno/Datasets/Paris/gt_data/";
        float r = eval_dataset_map(v, path_queries, path_gt, prototxt, caffemodel, img_size, layer_name, imgs_folder, bbox_txt, data_bin, normed);
	return r;
}

int main(int argc, char* argv[]){
        //*
        string prototxt = "/home/sormeno/Models/Alexnet/bvlc_alexnet_memory.prototxt";
        string caffemodel = "/home/sormeno/Models/Alexnet/bvlc_alexnet.caffemodel";
        int img_size = 256;
        string layer_name = "fc7";
        string data_bin1 = "/home/sormeno/data/ndata/paris_4096.bin";
	string data_bin2 = "/home/sormeno/data/ndata/oxford_4096.bin";
        bool normed = true;
	
	//extract_features_Paris(prototxt, caffemodel, img_size, layer_name, data_bin1, normed);
	//extract_features_Oxford(prototxt, caffemodel, img_size, layer_name, data_bin2, normed);

	//float r1 = eval_Paris(prototxt, caffemodel, img_size,layer_name, data_bin1, normed);	

	float r2 = eval_Oxford(prototxt, caffemodel, img_size,layer_name, data_bin2, normed);
	
	//cout << "paris " << r1 << endl;
	cout << "oxford " << r2 << endl;
        //get_features_dataset(prototxt, caffemodel, img_size, layer_name, imgs_folder, bbox_txt, data_out, normed);
        //*/
        return 0;
}

