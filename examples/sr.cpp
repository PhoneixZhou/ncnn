#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "net.h"
#include "mat.h"
#include "benchmark.h"

int run_alexnet(const cv::Mat& bgr,std::vector<float>& cls_scores){
    ncnn::Net net;

    net.load_param("/data/local/ncnn/sr.param");
    net.load_model("/data/local/ncnn/sr.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data,ncnn::Mat::PIXEL_RGB,bgr.cols,bgr.rows,227,227);
    ncnn::Mat out;
    const float mean_vals[3] = {104.f,117.f,123.f};

    in.substract_mean_normalize(mean_vals,0);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_num_threads(4);
    ex.set_light_mode(true);
    ex.input("data",in);

    fprintf(stderr,"run_alexnet begin\n");

    double start = ncnn::get_current_time();
    ex.extract("prob",out);
    double end = ncnn::get_current_time();

    double time = end- start;


    fprintf(stderr,"end of alexnet time = %7.2f\n",time);

    ncnn::Mat out_flatterned = out.reshape(out.w * out.h * out.c);

    cls_scores.resize(out_flatterned.w);
    for(int j = 0;j<out_flatterned.w;j++){
        cls_scores[j] = out_flatterned[j];
    }
    net.clear();
    return 0; 
}

static int print_topk(const std::vector<float>& cls_scores,int topk){
    int size = cls_scores.size();
    std::vector<std::pair <float,int> > vec;

    vec.resize(size);

    for(int i = 0;i<size;i++){
        vec[i] = std::make_pair(cls_scores[i],i);
    }

    std::partial_sort(vec.begin(),vec.begin()+topk,vec.end(),std::greater< std::pair<float,int> >());

    for(int i = 0;i<topk;i++){
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr,"%d = %f \n",index,score);
    }

    return 0;
}

int main(int argc,char** argv){
    if(argc !=2){
        fprintf(stderr,"Usage: %s [imagepath]\n",argv[0]);
        return -1;
    }
    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath,CV_LOAD_IMAGE_COLOR);
    if(m.empty()){
        fprintf(stderr,"cv::imread %s failed\n",imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    run_alexnet(m,cls_scores);

    print_topk(cls_scores,3);

    cls_scores.clear();
    return 0;
    
}