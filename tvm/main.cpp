//#include <iostream>
//#include "include/detection/anchor_generator.hpp"
//#include <sys/time.h>
//#include <opencv2/opencv.hpp>
//#include "./include/misc/time.hpp"
#include "include/detection/detector.hpp"

std::shared_ptr<snr::Tensor> data_blob_;
// 输入图片数量，通道数
int img_num_= 1;
int channel_ = 3;
// 输入到网络的图片高度和宽度
int height_= 1080;
int width_ = 1920;
//int height_= 360;
//int width_ = 640;
// 原图经过 resize_scale_ 之后输入到网络
float resize_scale_width_= 1.0;
float resize_scale_height_= 1.0;
//
//float resize_scale_width_= 0.333333;
//float resize_scale_height_= 0.333333;


//const string &feat_blob_name_,
//const bool ratio_fix_height,
//const int feat_stride,
//const float base_size,
//const float base_scale,
//const vector<float> &scales,
//const vector<float> &ratios)
template<typename T>
inline string convert_to_string(const T &val) {
    ostringstream os;
    os << val;
    return os.str();
}


int main() {

    const int gpu_id = 0;
    const int img_num = 1;
    float input_width_ = 640;
    float input_height_ = 640;

    SSDDetector ssd_detector(gpu_id, 650, 610);     //input 1080p
//    ssd_detector.set_resize_scale(0.3333333, 0.3333333);  //resize to 533*300 无效
    ssd_detector.set_resize(input_width_, input_height_);  //resize to 533*300
    ssd_detector.set_conf_thresh(0.6);
    vector<float> scales, ratios;

    //the anchor setting of p2_conv_1x1
    ratios.push_back(1);
    scales.push_back(1.0);
    scales.push_back(1.259);
    scales.push_back(1.587);
    ssd_detector.add_detect_blob("s8", false, 8, 16, 1, scales, ratios);

    //the anchor setting of p3_conv_1x1
    ratios.clear();
    scales.clear();
    ratios.push_back(1);
    scales.push_back(1.0);
    scales.push_back(1.259);
    scales.push_back(1.587);
    ssd_detector.add_detect_blob("s16", false, 16, 32, 1, scales, ratios);

    ratios.clear();
    scales.clear();
    ratios.push_back(1);
    scales.push_back(1.0);
    scales.push_back(1.259);
    scales.push_back(1.587);
    ssd_detector.add_detect_blob("s32", false, 32, 64, 1, scales, ratios);

    ratios.clear();
    scales.clear();
    ratios.push_back(1);
    scales.push_back(1.0);
    scales.push_back(1.259);
    scales.push_back(1.587);
    ssd_detector.add_detect_blob("s64", false, 64, 128, 1, scales, ratios);

    ratios.clear();
    scales.clear();
    ratios.push_back(1);
    scales.push_back(1.0);
    scales.push_back(1.259);
    scales.push_back(1.587);
    ssd_detector.add_detect_blob("s128", false, 128, 256, 1, scales, ratios);

    ssd_detector.generate_anchors();

    std::string lib_path("/Users/geyongtao/Desktop/RetinaFace.PyTorch/tvm/data/retinaface_small.so");
    std::string graph_path("/Users/geyongtao/Desktop/RetinaFace.PyTorch/tvm/data/retinaface_small.json");
    std::string param_path("/Users/geyongtao/Desktop/RetinaFace.PyTorch/tvm/data/retinaface_small.params");
    std::string image_path("/Users/geyongtao/Desktop/RetinaFace.PyTorch/tvm/data/0_Parade_Parade_0_545.jpg");
//    std::string image_path("/Users/geyongtao/Desktop/RetinaFace.PyTorch/tvm/data/0_Parade_Parade_0_901.jpg");
//    std::string image_path("/Users/geyongtao/CLionProjects/ssd-deploy/data/0_Parade_Parade_0_519.jpg");
    // init end


    // detection begin
    std::vector<std::vector<cv::Rect> > results_boxes;
    std::vector<vector<Point2f> > results_ldmks;
    std::vector<std::vector<float> > scores;
    double start = get_current_time();
    ssd_detector.im_detect(lib_path, graph_path, param_path, image_path, results_boxes, scores, results_ldmks);
    double end = get_current_time();
    std::cout << "Time cost: " << (end - start) << " ms" << std::endl;
    // detection end


    // save or show result begin
    const Scalar color(0, 255, 0);
    ofstream outfile;
    outfile.open("test.txt", std::ofstream::app);

    for (int img_idx = 0; img_idx < img_num; img_idx++) {
        std::vector<Rect> &img_boxes = results_boxes[img_idx];
        std::vector<float> &img_scores = scores[img_idx];
        std::vector<Point2f> &img_ldmks = results_ldmks[img_idx];
//        Mat &img = imgs[img_idx];
        Mat img = cv::imread(image_path);
        cv::resize(img, img, cv::Size(input_width_, input_height_));

        const int box_num = img_boxes.size();
        cout << "total object: " << box_num << endl;

        //！临时调试
        for (int box_idx = 0; box_idx < box_num; box_idx++) {
            Rect &box = img_boxes[box_idx];

            rectangle(img, Point(box.x, box.y),
                      Point(box.x + box.width, box.y + box.height), color, 3);

            circle(img, img_ldmks[box_idx*5], 3, Scalar(0, 0, 255), -1);
            circle(img, img_ldmks[box_idx*5+1], 3, Scalar(0, 0, 255), -1);
            circle(img, img_ldmks[box_idx*5+2], 3, Scalar(0, 0, 255), -1);
            circle(img, img_ldmks[box_idx*5+3], 3, Scalar(0, 0, 255), -1);
            circle(img, img_ldmks[box_idx*5+4], 3, Scalar(0, 0, 255), -1);

            putText(img, convert_to_string(img_scores[box_idx]),
                    Point(box.x, box.y), CV_FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
//            cout << img_scores[box_idx] << "[" << box.x << "," << box.y << "," << box.x + box.width << ","
//                 << box.y + box.height << "]" << endl;
            outfile  << " " << img_scores[box_idx] << " [" << box.x << "," << box.y << ","
                    << box.x + box.width << "," << box.y + box.height << "]" << endl;
        }

        cv::imshow("result", img);

        cv::imwrite("/Users/geyongtao/Desktop/RetinaFace.PyTorch/tvm/data/0_Parade_Parade_0_545_result.jpg", img);
//        cv::imwrite("/Users/geyongtao/Desktop/RetinaFace.PyTorch/tvm/data/0_Parade_Parade_0_901_result.jpg", img);
//        cv::imwrite("/Users/geyongtao/Desktop/RetinaFace.PyTorch/tvm/data/0_Parade_Parade_0_519_result.jpg", img);
        cv::waitKey(0);
    }

}




