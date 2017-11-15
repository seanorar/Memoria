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
 
void proces_line(string line, double buffer[]){
	
	istringstream iss(line);
	int index = 0;
	for(string s; iss >> s; ){
		buffer[index] = atof(s.c_str());
		index += 1;
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

void get_features(string dataset_shots, string dataset_bbox, string data_out, bool is_binary){
	//extraccion de features a partir de los bbox
	string str_pt, str_caffemodel, layer_name;
        
	if (is_binary){
		str_pt = "/home/sormeno/Models/Alexnet-DSH/alex_hfc8_1024.prototxt";
        	str_caffemodel = "/home/sormeno/Models/Alexnet-DSH/alex_hfc8_1024.caffemodel";
		layer_name = "hfc8_f";
	}
	else {
		str_pt = "/home/sormeno/Models/Alexnet/bvlc_alexnet_memory.prototxt";
        	str_caffemodel = "/home/sormeno/Models/Alexnet/bvlc_alexnet.caffemodel";
        	layer_name = "fc7";
	}

        CaffePredictor caffe_predictor(str_pt, str_caffemodel, 224, 224, CAFFE_GPU_MODE);
	string im_folder = dataset_shots;
        ofstream writeFile (data_out, ios::out | ios::binary);
	ifstream file(dataset_bbox);
	string str;
	int des_size = 0;
	int actual = 0;
	long n_total = 0;
	while (getline(file, str)){
		double buffer[6];
		proces_line(str, buffer);
		if (actual != buffer[4]){
			cout << buffer[4] << endl;
			actual = buffer[4];
		}
                string im_name = dataset_shots  + to_string((int)buffer[5]) + ".jpg";
                cv::Mat aux_image = cv ::imread(im_name);
		cv::Rect r((int)buffer[0],(int)buffer[1], (int)( buffer[2]-buffer[0]), (int) (buffer[3]-buffer[1]));		
        	cv::Mat im_bbox(aux_image, r);
                float * des_im = caffe_predictor.getCaffeDescriptor(im_bbox, &des_size, layer_name);
		cout << des_size << endl;
		if (!is_binary){
			normalize(des_im, des_size);
		}
                writeFile.write((char*) des_im, sizeof(float) * des_size);
		n_total += 1;
        }
        writeFile.close();
	cout<< "n rois =  "<< n_total << endl;
}

cv::Mat get_feature(string name_query, bool is_binary){
        //extracción de característica de una imagen
        string str_pt, str_caffemodel, layer_name;
        int mean_size;
        if (is_binary){
                str_pt = "/home/sormeno/Models/Alexnet-DSH/alex_hfc8_1024.prototxt";
                str_caffemodel = "/home/sormeno/Models/Alexnet-DSH/alex_hfc8_1024.caffemodel";
                layer_name = "hfc8_f";
                mean_size = 224;
        }
        else{
                str_pt = "/home/sormeno/Models/Alexnet/bvlc_alexnet_memory.prototxt";
                str_caffemodel = "/home/sormeno/Models/Alexnet/bvlc_alexnet.caffemodel";
                layer_name = "fc7";
                mean_size = 256;
        }

        CaffePredictor caffe_predictor(str_pt, str_caffemodel, mean_size, mean_size, CAFFE_GPU_MODE);

        cv::Mat mat_image = cv::imread(name_query);
        JUtil::jmsr_assert(!mat_image.empty(), " image failed");
        int des_size = 0;
        float *des_consulta = caffe_predictor.getCaffeDescriptor(mat_image, &des_size, layer_name);
        cout << des_size <<endl;
        if (!is_binary){
                normalize(des_consulta, des_size);
        }
        cv::Mat mat_consulta = cv::Mat(1, des_size, CV_32F, des_consulta);
        return mat_consulta;
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

vector <int> get_shot_id(string file_name){
	//obtiene el id del shot asociado al respectivo roi	
	ifstream file(file_name.c_str());
        string line;
	vector <int> result;
	while (getline(file, line)){
		double buffer[6]; //cambiar 5 por 6 
                proces_line(line, buffer);
		result.push_back(buffer[4]);
	}
	return result;
}

vector <vector<double>> get_shot_info(string file_name){
        //obtiene los datos del txt de los shots     
        ifstream file(file_name.c_str());
        string line;
        vector <vector<double>> result;
        while (getline(file, line)){
                double buffer[6];  //cambiar 5 por 6
                proces_line(line, buffer);
                vector<double> aux{begin(buffer), end(buffer)}; 
                result.push_back(aux);
        }
        return result;
}


vector <int> gt(string file_name, int video_id, int img_id){
	//obtiene los shots pertenecientes al gt especificando el id del video y de la imagen de consulta.
	
	string videos_id[5] = {"0","shot96", "shot8", "shot11", "shot130"};
	ifstream file(file_name.c_str());
	string line;
	vector<int> gt;
	while (getline(file, line)){
		istringstream iss(line);
		string sep[2];
		int index = 0;
		for(string s; iss >> s;){
			sep[index] = s;
			index += 1;
		}
		vector <string> v = split(sep[1], '_');
		if (img_id == stoi(sep[0])  && videos_id[video_id] == v.at(0)){
			gt.push_back(stoi(v.at(1)));
		}
	}

	return gt;
}

bool is_relevant(int im_id, vector<int> gt_list){
	//determina si un elemento es relevante dentro de la consulta
	
	if(find(gt_list.begin(), gt_list.end(), im_id) != gt_list.end()) {
		return true;
	}

	return false;
}


tuple <float, vector<int> > get_ap(vector <int> similar_list, vector <int> gt_list){
	//evaluación del average_precision 
	
	float num_relevant = 0;
	float average = 0;
	int n = similar_list.size();
	cout<< "Posiciones : ";	
	vector <int> r_pos;
	for(int i = 0; i < n; i += 1){
		if (is_relevant(similar_list.at(i), gt_list)){
			num_relevant += 1;
			average += (num_relevant / (i + 1));
			cout  << i <<  " - ";
			r_pos.push_back(i);
		}
	}
	cout << endl;
	return make_tuple(average / num_relevant, r_pos);	
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


vector <int> get_similar(vector <cv::Mat> mat_consultas, string feature_data, string bbox_data, bool is_binary){
        //función que a partir de un descriptor, entrega una lista de los elementos semejantes ordenados desde el más cercano
        int feature_len = mat_consultas.at(0).cols;
	vector <pair<double, string>> dist_list;
        float result [feature_len];
        ifstream readFile (feature_data, ios::in | ios::binary);
        ifstream file(bbox_data);
        string str;
	long limit = num_lines(bbox_data);
        for(int i = 0; i < limit; i += 1){
		getline(file, str);
                double buffer[6];
                proces_line(str, buffer);
                string im_name = to_string(i);
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
			dist_list.push_back(make_pair(dist ,im_name));
		}
        }
        file.close();
        readFile.close();
        sort(dist_list.begin(),dist_list.end());
	vector<int> final_result;
	int n_dist = dist_list.size();
        for(int i = 0; i < n_dist; i += 1){
                final_result.push_back(stoi(dist_list.at(i).second));
	} 
	return final_result;
}
void test_pca(){
        float result [4096];
	string feature_data = "/home/sormeno/data/ndata/features_vgg_1.bin";
	string bbox_data = "/home/sormeno/data/ndata/bbox_vgg_1.txt";
        ifstream readFile (feature_data, ios::in | ios::binary);
        long limit = num_lines(bbox_data);
	cv::Mat all_features;
        for(int i = 0; i < 100; i += 1){
                readFile.read ((char*)result, sizeof(float)*4096);
                cv::Mat mat_im = cv::Mat(1, 4096, CV_32F, result);
		all_features.push_back(mat_im);
        }
        readFile.close();
	cv::PCA pca(all_features, cv::Mat(), cv::PCA::DATA_AS_ROW, 100);
	cv::FileStorage fs("/home/sormeno/pcadata",cv::FileStorage::WRITE);
	pca.write(fs);
	fs.release();
}


void save_result_im(vector <int> f_roi_id, vector <vector< double >> shots_info, string path_img,  string path_out, vector <int> pos = vector<int>()){
	int len_pos = pos.size();
	if ( len_pos > 0){
		for (int k = 0; k < len_pos; k+=1){
			int i = pos.at(k);
			int x1 = shots_info.at(f_roi_id.at(i)).at(0);
                        int x2 = shots_info.at(f_roi_id.at(i)).at(2);
                        int y1 = shots_info.at(f_roi_id.at(i)).at(1);
                        int y2 = shots_info.at(f_roi_id.at(i)).at(3);
                        int img_id = shots_info.at(f_roi_id.at(i)).at(5);
                        cv::Mat image = cv ::imread(path_img + to_string(img_id)+".jpg");
                        cv::Point pt1(x1, y1);
                        cv::Point pt2(x2, y2);
                        cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0));
                        cv::imwrite(path_out+to_string(img_id)+".jpg", image);
		}
		
	}
	else{
		for(int i = 0; i < 25; i += 1){
			int x1 = shots_info.at(f_roi_id.at(i)).at(0);
			int x2 = shots_info.at(f_roi_id.at(i)).at(2);		
			int y1 = shots_info.at(f_roi_id.at(i)).at(1);
			int y2 = shots_info.at(f_roi_id.at(i)).at(3);
			int img_id = shots_info.at(f_roi_id.at(i)).at(5);
			cv::Mat image = cv ::imread(path_img + to_string(img_id)+".jpg");
			cv::Point pt1(x1, y1);
			cv::Point pt2(x2, y2);
			cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0));
			cv::imwrite(path_out+to_string(i)+".jpg", image);
		}
	}
}


void get_top_100_roi(string img_path, bool is_boolean, string f_data, string bbox_data, string imgs_path, string output_path){
	vector <cv::Mat> mat_consulta;
	mat_consulta.push_back(get_feature(img_path, is_boolean));
	vector <int> r = get_similar(mat_consulta, f_data, bbox_data, is_boolean);
	vector <vector<double>> shots_data = get_shot_info(bbox_data);
	for (int i = 0; i < 100; i+=1){
		//cout << r.at(i) << endl;
		vector <double> buffer = shots_data.at(r.at(i));
		string im_name = imgs_path  + to_string((int)buffer[5]) + ".jpg";
                cv::Mat aux_image = cv ::imread(im_name);
                cv::Rect r((int)buffer[0],(int)buffer[1], (int)( buffer[2]-buffer[0]), (int) (buffer[3]-buffer[1]));
                cv::Mat im_bbox(aux_image, r);
		cv::imwrite(output_path + to_string(i) + ".jpg", im_bbox);
	}
}


double eval_map(string img_folder,string im_name, int num_images, int c_id, string work_path, int mode, bool is_binary){
	vector <cv::Mat> mat_consultas;
	for(int i = 1; i <= num_images; i += 1){
		string str_image;
		if (mode==1){
			str_image = img_folder + im_name +"." +to_string(i) + ".src.png";
		}
		else{
			str_image = img_folder + im_name +"." +to_string(i) + ".src2.png";
		}
		cout << str_image << endl;
        	cv::Mat mat_consulta = get_feature(str_image, is_binary);
		mat_consultas.push_back(mat_consulta);
	}
	string gt_filename = "/home/sormeno/data/gt2.txt";
        float final_map = 0;
	cout << "---------------- " << endl;
	int denominador = 0;

        for (int id = 1; id < 2; id += 1){

                cout << "Trabajando en video " << id << endl;
                string video_id = to_string(id);
		string shots_img = work_path + "shots" + video_id + "/";
                string bbox_data = work_path + "bbox_t11_" + video_id + ".txt";
		string f_data;
		if (is_binary){
			f_data = work_path + "bfeatures_t11_256_" + video_id  + ".bin";
		}
		else{
			f_data = work_path + "features_t11_" + video_id + ".bin";
		}
                vector <int> r = get_similar(mat_consultas, f_data, bbox_data, is_binary);
                vector <int> shots_id = get_shot_id(bbox_data);
		int len_shots_id = shots_id.size();
                int n_shots = shots_id.at(len_shots_id - 1);
                vector <int> in_list(n_shots + 1, 0);
		vector <int> f_roi_id;
                vector <int> f_shot_id;
                for (int i = 0; i < len_shots_id; i += 1){
                        int id_element = shots_id.at(r.at(i));
                        if (in_list[id_element] == 0){
                                f_shot_id.push_back(id_element + 1);
				f_roi_id.push_back(r.at(i));
                                in_list.at(id_element) = 1;
			}
                }
		cout  << f_shot_id.size() << " ++++++++++++++++++++++++++++++++++++++ "<< endl;
		vector <int> gt_list = gt(gt_filename, stoi(video_id), c_id);
		vector <vector<double>> shots_info = get_shot_info(bbox_data);
		if (gt_list.size() == 0){
			string path_out = "/home/sormeno/data/result/" + to_string(mode) + "/" + im_name  + "/m2/" + video_id + "/";
			cout << endl <<"Shots detectados" << endl;
			cout << "[";
			for(int k = 0;k < 25; k += 1){
				cout << f_shot_id.at(k) << ", ";
			}
			cout << "]" << endl;
			save_result_im(f_roi_id, shots_info, shots_img, path_out);
		}
		else{
			denominador += 1;
			string path_out = "/home/sormeno/data/result/" + to_string(mode) + "/" + im_name  + "/m1/" + video_id + "/";
                	tuple <float, vector <int>> ap = get_ap(f_shot_id, gt_list);
                	cout  << "AP = " << get<0>(ap) << endl;
                	final_map += get<0>(ap);
			save_result_im(f_roi_id, shots_info, shots_img, path_out, get<1>(ap));
		}
		cout << "---------------- " << endl;
        }
	if (denominador ==0){
		cout << "MAP = " << (0) << endl;
        	return 0;
	}
        cout << "MAP = " << (final_map / denominador) << endl;
	return (final_map / denominador);
}


void evaluacion(int mode, bool is_binary){
	vector<int> consultas;
	vector<int> ids = { 9070, 9076, 9086, 9101, 9103, 9112};
	for(int i = 0; i < ids.size(); i += 1){
		consultas.push_back(ids.at(i));
	}
	float f_result = 0;
	float den = 0;
	vector <float> parcial_result;
	for (int i = 0; i < consultas.size(); i += 1){
		string path = "/home/sormeno/data/queries/";
		string im_name = to_string(consultas.at(i));
		int num_img = 4;
		int c_id = consultas.at(i);
		string path_data = "/home/sormeno/data/ndata/";
		float result_eval = eval_map(path, im_name ,num_img, c_id, path_data, mode, is_binary);
		parcial_result.push_back(result_eval);
		if (result_eval != 0){
			f_result += result_eval;
			den += 1;
		}
	}
	cout << "Resultados: " << endl;
	for(int i = 0; i < consultas.size(); i += 1){
		cout << "Imagen  " << consultas.at(i) << " = " << parcial_result.at(i)<<endl;
	}
	cout << "------------------------------------"<<endl;
	cout <<"MAP final = "<< (f_result/den) << endl<<endl;
	cout << "------------------------------------"<<endl;
}


void save_videos_roi(string path_shots,string  bbox_file, string path_out){
	vector <vector<double>> shots_info = get_shot_info(bbox_file);
	int len_shots_info = shots_info.size();
	for (int i = 0; i < len_shots_info; i += 1){
		vector<double> info = shots_info.at(i);
		int x1 = info.at(0);
                int x2 = info.at(2);
                int y1 = info.at(1);
                int y2 = info.at(3);
                int img_id = info.at(5);
		cv::Mat aux_image = cv ::imread(path_shots + to_string(img_id)+ ".jpg");
                cv::Rect r((int)x1,(int)y1, (int)(x2 - x1), (int) (y2 - y1));
                cv::Mat im_bbox(aux_image, r);
		cv::imwrite(path_out + to_string(i)+".jpg", im_bbox);
	}
}


int main(int argc, char* argv[]){
	/*
	for(int i = 11; i < 12; i += 1){
		string shots_data = "/home/sormeno/data/ndata/shots1/";
		string bbox_data = "/home/sormeno/data/ndata/bbox_t" + to_string(i) + "_1.txt";
		string out_p = "/home/sormeno/data/ndata/bfeatures_t" + to_string(i) + "_1024_1.bin";
		get_features(shots_data, bbox_data, out_p, true);
	}
	*/
	/*
        string dataset_shots = "/home/sormeno/Datasets/imagenet/ILSVRC2013_DET_val/";
        string dataset_bbox = "/home/sormeno/bboxs.txt";
        string data_out = "/home/sormeno/data/ndata/imagenet_features.bin";
        get_features_imagenet(dataset_shots, dataset_bbox, data_out);
	*/
        //evaluacion(1, true);
	//*
	//string f_data = "/home/sormeno/data/ndata/features_t11_1.bin";
	string f_data = "/home/sormeno/data/ndata/bfeatures_t11_1024_1.bin";
        string bbox_data = "/home/sormeno/data/ndata/bbox_t11_1.txt";
        string imgs_path = "/home/sormeno/data/ndata/shots1/";
	string output_path = "/home/sormeno/data/ndata/r/";
        get_top_100_roi("/home/sormeno/data/queries/9112.1.src.png", true, f_data, bbox_data, imgs_path, output_path);
	//*/
	//test_pca();
	///*
	return 0;
}
