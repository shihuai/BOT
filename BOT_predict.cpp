#include <iostream>
#include <fstream>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
//#include <pair>
#include <map>
#include <algorithm>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace caffe;
using namespace std;
using namespace cv;

typedef struct Elem{
    float prob;
    int idx;
}ProbToIdx;

bool operator < (const ProbToIdx &a, const ProbToIdx &b)
{
    return a.prob < b.prob;
}

void send_to_net(std::vector<std::vector<ProbToIdx> >& result,
        std::vector<cv::Mat>& imgs, std::vector<int>& dvl,
        Net<float>& caffe_test_net)
{
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >
        (caffe_test_net.layers()[0])->AddMatVector(imgs, dvl);

    float loss = 0.0;
    std::vector<Blob<float>* > output_prob = caffe_test_net.Forward(&loss);
    int r = 1;
    //const float* argmaxs = output_prob[r]->cpu_data();

    int length = 12 * imgs.size();
    for(int i = 0; i < length; i += 12)
    {
        std::vector<ProbToIdx> temp_output;
        for(int j = i; j < i + 12; ++ j)
        {
            ProbToIdx temp;
            temp.idx = j - i;
            //temp.prob = argmaxs[j];
            temp.prob = output_prob[r]->cpu_data()[j];

            temp_output.push_back(temp);
        }
        result.push_back(temp_output);
    }
}
std::vector<std::vector<ProbToIdx> > start_predict(
        const std::vector<std::string>& img_list,
        Net<float> &caffe_test_net)
{
    std::vector<cv::Mat> imgs;
    std::vector<int> dvl;
    std::vector<std::vector<ProbToIdx> > result;
    int count = 0;
    cout << img_list.size() << endl;
    for(int i = 0; i < img_list.size(); ++ i)
    {
        ++ count;
        cv::Mat img = imread(img_list[i]);

        if (!img.data)
        {
            LOG(INFO) << img_list[i];

            break;
        }
        //cout << img_list[i] << endl;
        cv::resize(img, img, cv::Size(224, 224));
        imgs.push_back(img);
        dvl.push_back(i);

        if(count % 48 == 0)
        {
            std::cout << "Process " << count << " files" << std::endl;
            send_to_net(result, imgs, dvl, caffe_test_net);
            //count = 0;
            imgs.clear();
            dvl.clear();
        }
    }

   // LOG(INFO) << "Process " << count << " files";
   // if (count != 0 && count < 8)
   // {
   //     send_to_net(result, imgs, dvl, caffe_test_net);
   // }

    return result;
}

std::vector<std::vector<ProbToIdx> > start_predict(
        const std::vector<std::string>& img_list,
        const std::vector<std::vector<float> > &coordinate, Net<float> &caffe_test_net)
{
    std::vector<cv::Mat> imgs;
    std::vector<int> dvl;
    std::vector<std::vector<ProbToIdx> > result;
    int count = 0;
    cout << img_list.size() << endl;
    for(int i = 0; i < img_list.size(); ++ i)
    {
        ++ count;
        cv::Mat img = imread(img_list[i]);

        if (!img.data)
        {
            LOG(INFO) << img_list[i];

            break;
        }
        //cout << img_list[i] << endl;
        cv::Mat crop_img = img(cv::Range(int(coordinate[i][1]), int(coordinate[i][3])),
                cv::Range(int(coordinate[i][0]), int(coordinate[i][2])));
        //cout << crop_img.size() << endl;
        cv::resize(crop_img, crop_img, cv::Size(224, 224));
        imgs.push_back(crop_img);
        dvl.push_back(i);

        if(count % 99 == 0)
        {
            std::cout << "Process " << count << " files" << std::endl;
            send_to_net(result, imgs, dvl, caffe_test_net);
            //count = 0;
            imgs.clear();
            dvl.clear();
        }
    }

   // LOG(INFO) << "Process " << count << " files";
   // if (count != 0 && count < 8)
   // {
   //     send_to_net(result, imgs, dvl, caffe_test_net);
   // }

    return result;
}

std::vector<std::pair<int, int> > select_top2(std::vector<std::vector<ProbToIdx> > &result)
{
    std::vector<std::pair<int, int> > res_idx;

    for(int i = 0; i < result.size(); ++ i)
    {
        vector<ProbToIdx> one_res = result[i];

        sort(one_res.begin(), one_res.end());

        int len = one_res.size();
        std::pair<int, int> temp(one_res[len - 1].idx, one_res[len - 2].idx);

        res_idx.push_back(temp);
    }

    return res_idx;
}

int main(int argc, char** argv)
{
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    std::string deploy_path = "/home/banggui/caffe_software/caffe-master/examples/BOT/GoogLeNet_deploy.prototxt";
    std::string weight_path = "/home/banggui/caffe_software/caffe-master/examples/BOT/BOT_googlenet_v6_iter_42000.caffemodel";
    std::string file_list_path = "/home/banggui/caffe_software/caffe-master/examples/BOT/test7.txt";

    Net<float> caffe_test_net(deploy_path, caffe::TEST);
    caffe_test_net.CopyTrainedLayersFrom(weight_path);

    std::vector<std::string> img_list;
    std::ifstream in(file_list_path.c_str(), std::ios::in);
    std::string temp;
    std::string path_root = "/home/banggui/caffe_software/caffe-master/examples/BOT/";
    std::vector<int> label;
    std::vector<int> hidden;
    int n_temp;
    int n_h_temp;
    while (in >> temp)
    {
        img_list.push_back(path_root + temp);
    }
    in.close();
    //while (in >> temp >> n_temp >> n_h_temp)
    //{
    //    img_list.push_back(path_root + temp);
    //    label.push_back(n_temp);
    //    hidden.push_back(n_h_temp);
    //}
    //in.close();

    //cout << img_list.size() << endl;
    //in.open("examples/BOT/img_labels.txt");
    //std::vector<std::vector<float> > coordinate;
    //while (in >> temp)
    //{
    //    std::vector<float> vec_temp;

    //    for (int i = 0; i < 4; ++ i)
    //    {
    //        float d_temp;
    //        in >> d_temp;
    //        vec_temp.push_back(d_temp);
    //    }

    //    coordinate.push_back(vec_temp);
    //}
    //in.close();

    //cout << coordinate.size() << endl;
    //std::vector<std::vector<ProbToIdx> > result = start_predict(img_list, coordinate, caffe_test_net);
    std::vector<std::vector<ProbToIdx> > result = start_predict(img_list, caffe_test_net);

    std::vector<std::pair<int, int> > res_idx = select_top2(result);

    std::ofstream out("/home/banggui/caffe_software/caffe-master/examples/BOT/result_googlenet_test_7.txt", std::ios::out);
    std::ofstream out_prob("examples/BOT/prob_7_2.txt");

    std::vector<int> error_exam;
    //int count = 0;
    int h_count = 0;
    int top1_count = 0;
    int top2_count = 0;
    double sum = 0.0;
    for (int i = 0; i < res_idx.size(); ++ i)
    {
        size_t start_pos = img_list[i].find_last_of("/");
        size_t end_pos = img_list[i].find_last_of(".");

        out << img_list[i].substr(start_pos + 1, end_pos - start_pos - 1) << "\t";
        out_prob << img_list[i].substr(start_pos + 1, end_pos - start_pos - 1) << " ";
        for (int j = 0; j < result[i].size(); ++ j)
        {
            out_prob << result[i][j].prob << " ";
        }
        out_prob << endl;
        //double t1 = exp(result[i][res_idx[i].first].prob) / (exp(result[i][res_idx[i].first].prob) + exp(result[i][res_idx[i].second].prob));
        //double t2 = exp(result[i][res_idx[i].second].prob) / (exp(result[i][res_idx[i].first].prob) + exp(result[i][res_idx[i].second].prob));
        //out << res_idx[i].first << "\t" << setiosflags(ios::fixed) << setprecision(6)
        //    << t1 << "\t" << res_idx[i].second << "\t" << setiosflags(ios::fixed) << setprecision(6)
        //    << t2 << std::endl;
        out << res_idx[i].first << "\t"  << setiosflags(ios::fixed) << setprecision(6)
            << result[i][res_idx[i].first].prob << "\t" << res_idx[i].second << "\t"
            << setiosflags(ios::fixed) << setprecision(6) << result[i][res_idx[i].second].prob
            << std::endl;

        //if (res_idx[i].first == label[i])
        //{
        //    if (hidden[i])
        //    {
        //        h_count ++;
        //    }
        //    else
        //    {
        //        top1_count ++;
        //    }
        //}
        //if (res_idx[i].second == label[i])
        //{
        //    if (hidden[i])
        //    {
        //        h_count ++;
        //    }
        //    else
        //    {
        //        top2_count ++;
        //    }
        //}
        //if (res_idx[i].first == label[i] || res_idx[i].second == label[i]):wq
        //
        //{
            error_exam.push_back(i);
        //    count ++;
        //}
    }
    out.close();
    out_prob.close();

    std::cout << 1.0 * (top1_count + 0.4 * top2_count + 2 * h_count) / label.size() << std::endl;
    std::vector<std::string> class_names;
    in.open("/home/banggui/caffe_software/caffe-master/examples/BOT/name.txt");

    while(in >> temp)
    {
        class_names.push_back(temp);
    }
    in.close();

    cout << class_names.size() << endl;
    out.open("/home/banggui/caffe_software/caffe-master/examples/BOT/error.txt");
    for (int i = 0; i < error_exam.size(); ++ i)
    {
        std::cout << img_list[error_exam[i]] << std::endl;
       // out << img_list[error_exam[i]] << " " <<
       //     label[error_exam[i]] << " " << res_idx[error_exam[i]].first << std::endl;
        cv::Mat img = imread(img_list[error_exam[i]]);
        cv::resize(img, img, cv::Size(480, 480));
        char buffer[101];
        sprintf(buffer, "%s: %d" , class_names[res_idx[error_exam[i]].first].c_str(),
                res_idx[error_exam[i]].first);
        cv::putText(img, buffer, Point(0, 20), 2, 1,
                cv::Scalar(0, 255, 0));

        sprintf(buffer, "%s: %d", class_names[res_idx[error_exam[i]].second].c_str(),
                res_idx[error_exam[i]].second);
        cv::putText(img, buffer, Point(0, 40), 2, 1,
                cv::Scalar(0,255, 0));
        cv::imshow("img", img);
        cv::waitKey(0);
    }
    out.close();

    return 0;
}
