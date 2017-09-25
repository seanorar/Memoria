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

void get_features(string dataset_shots, string dataset_bbox, string data_out){
	//extraccion de features a partir de los bbox
        
	string str_pt("/home/sormeno/AlexNet/bvlc_alexnet_memory.prototxt");
        string str_caffemodel("/home/sormeno/AlexNet/bvlc_alexnet.caffemodel");
        CaffePredictor caffe_predictor(str_pt, str_caffemodel, 256, 256, CAFFE_GPU_MODE);

	string im_folder = dataset_shots;
        ofstream writeFile (data_out, ios::out | ios::binary);
	ifstream file(dataset_bbox);
	string str;
	int des_size = 0;
	int actual = 0;
	long n_total = 0;
	while (getline(file, str)){
		double buffer[5];
		proces_line(str, buffer);
		if (actual != buffer[4]){
			cout << buffer[4]<<endl;
			actual = buffer[4];
		}
                string im_name = dataset_shots  + to_string((int)buffer[4]) + ".jpg";
                cv::Mat aux_image = cv ::imread(im_name);
		cv::Rect r((int)buffer[0],(int)buffer[1], (int)( buffer[2]-buffer[0]), (int) (buffer[3]-buffer[1]));		
        	cv::Mat im_bbox(aux_image, r);
                float * des_im = caffe_predictor.getCaffeDescriptor(im_bbox, &des_size, "fc7");
		normalize(des_im, 4096);
                writeFile.write((char*) des_im, sizeof(float) *des_size);
		n_total += 1;
        }
        writeFile.close();
	cout<< "n rois =  "<< n_total << endl;
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
        //obtiene el id del shot asociado al respectivo roi     
        ifstream file(file_name.c_str());
        string line;
        vector <vector<double>> result;
        while (getline(file, line)){
                double buffer[6];  //cambiar 5 por 6
                proces_line(line, buffer);
		vector<double> aux{begin(buffer), end(buffer)};
		//aux.at(4) = (int)(aux.at(4)/3); 
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

cv::Mat get_feature(string name_query){
	//extracción de característica de una imagen.
	string str_pt("/home/sormeno/AlexNet/bvlc_alexnet_memory.prototxt");
        string str_caffemodel("/home/sormeno/AlexNet/bvlc_alexnet.caffemodel");
        CaffePredictor caffe_predictor(str_pt, str_caffemodel, 256, 256, CAFFE_GPU_MODE);

        cv::Mat mat_image = cv::imread(name_query);
        JUtil::jmsr_assert(!mat_image.empty(), " image failed");
        int des_size = 0;
        float *des_consulta = caffe_predictor.getCaffeDescriptor(mat_image, &des_size, "fc7");
        normalize(des_consulta, 4096);
        cv::Mat mat_consulta = cv::Mat(1, des_size, CV_32F, des_consulta);
	
	return mat_consulta;
}


vector <int> get_similar(vector <cv::Mat> mat_consultas, string feature_data, string bbox_data){
        //función que a partir de un descriptor, entrega una lista de los elementos semejantes ordenados desde el más cercano
        
	vector <pair<double, string>> dist_list;
        float result [4096];
        ifstream readFile (feature_data, ios::in | ios::binary);
	long limit = num_lines(bbox_data);
        for(int i = 0; i < limit; i += 1){
                string im_name = to_string(i);
                readFile.read ((char*)result, sizeof(float)*4096);
                cv::Mat mat_im = cv::Mat(1, 4096, CV_32F, result);
		int n_vectores = mat_consultas.size();
		for(int k = 0; k < n_vectores; k += 1){
			cv::Mat mat_consulta = mat_consultas.at(k);
                	double dist = cv::norm(mat_consulta, mat_im);
                	dist_list.push_back(make_pair(dist ,im_name));
		}
        }
        readFile.close();
        sort(dist_list.begin(),dist_list.end());
	vector<int> final_result;
	int n_dist = dist_list.size();
        for(int i = 0; i < n_dist; i += 1){
                final_result.push_back(stoi(dist_list.at(i).second));
	} 
	return final_result;
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

void get_top_100_roi(string img_path){
	vector <cv::Mat> mat_consulta;
	mat_consulta.push_back(get_feature(img_path));
	string f_data = "/home/sormeno/data/ndata/features1_vgg.bin";
	string bbox_data = "/home/sormeno/data/ndata/test_vgg_1.txt";
	vector <int> r = get_similar(mat_consulta, f_data, bbox_data);
	for (int i = 0; i < 100; i+=1){
		cout << r.at(i) << endl;
		cv::Mat image = cv ::imread("/home/sormeno/data/ndata/s/" + to_string(r.at(i)) + ".jpg");
		cv::imwrite("/home/sormeno/data/ndata/r/" + to_string(i) + ".jpg",image);
	}
}

double eval_map(string img_folder,string im_name, int num_images, int c_id, string work_path, int mode){
	vector <cv::Mat> mat_consultas;
	for(int i = 1; i <= num_images; i += 1){
		string str_image;
		if (mode==1){
			str_image = img_folder + im_name +"." +to_string(i) + ".src.png";
		}
		else{
			str_image = img_folder + im_name +"." +to_string(i) + ".src2.png";
		}
		cout <<str_image<<endl;
        	cv::Mat mat_consulta = get_feature(str_image);
		mat_consultas.push_back(mat_consulta);
	}
	string gt_filename = "/home/sormeno/data/gt2.txt";
        float final_map = 0;
	cout << "---------------- " << endl;
	int denominador = 0;

        for (int id = 1; id < 5; id += 1){

                cout << "Trabajando en video " << id << endl;
                string video_id = to_string(id);
		string shots_img = work_path + "shots" + video_id + "/";
                string f_data = work_path + "features" + video_id + "_vgg.bin";
                string bbox_data = work_path + "test_vgg_" + video_id + ".txt";
                vector <int> r = get_similar(mat_consultas, f_data, bbox_data);
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
		vector <int> gt_list = gt(gt_filename, stoi(video_id), c_id);
		vector <vector<double>> shots_info = get_shot_info(bbox_data);
		if (gt_list.size() == 0){
			string path_out = "/home/sormeno/data/result/top/" + video_id + "/";
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


void evaluacion(int mode){
	vector<int> consultas;
	for(int i = 9069; i < 9129; i+=1){
		consultas.push_back(i);
	}
	float f_result = 0;
	float den = 0;
	vector <float> parcial_result;
	for (int i =0; i < consultas.size(); i += 1){
		string path = "/home/sormeno/data/queries/";
		string im_name = to_string(consultas.at(i));
		int num_img = 4;
		int c_id = 0;//consultas.at(i);
		string path_data = "/home/sormeno/data/ndata/";
		float result_eval = eval_map(path, im_name ,num_img, c_id, path_data, mode);
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
	cout << "------------------------------------"<<endl<<endl;
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
	for(int i = 1; i < 2; i += 1){
		string shots_data = "/home/sormeno/data/ndata/shots"+to_string(i)+"/";
		string bbox_data = "/home/sormeno/data/ndata/test_s3_"+to_string(i)+".txt";
		string out_p = "/home/sormeno/data/ndata/features"+to_string(i)+"_s3.bin";
		get_features(shots_data, bbox_data, out_p);
	}
	*/

	//*
	if (argc != 5){
		cout << "Llamar funcion con nombre de imagen , su identificador y carpeta de trabajo" << endl;
		return 1;
	}
	string im_folder = argv[1];
	string im_name = argv[2];
	int num_images = stoi(argv[3]);
	int c_id = 0;
	string path = argv[4];
	eval_map(im_folder, im_name ,num_images, c_id, path, 1);
	///*/

	//evaluacion(1);

	return 0;
}

