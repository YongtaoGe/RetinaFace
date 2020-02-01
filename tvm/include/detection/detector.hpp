//
// Created by geyongtao on 2019-07-07.
//

#ifndef SSD_DEPLOY_DETECTOR_HPP
#define SSD_DEPLOY_DETECTOR_HPP
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../misc/time.hpp"
#include "../engine/tensor.h"

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <fstream>




using namespace std;
using namespace cv;

class SSDDetector {
public:
    SSDDetector(const int gpuid,
                const int src_width,
                const int src_height);


//    void init_model(const string &model_path, int num_threads = 1);
    void generate_anchors();//generate anchors

//    void im_detect(const vector<Mat> &imgs, vector<vector<Rect> > &boxes,
//                   vector<vector<float> > &scores);

    void im_detect(const std::string &lib_path,
                                const std::string &graph_path,
                                const std::string &param_path,
                                const std::string &image_path,
                                vector<vector<Rect> > &boxes,
                                vector<vector<float> > &scores,
                                vector<vector<Point2f> > &ldmks);

    void add_detect_blob(const string &feat_blob_name_, const bool ratio_fix_height,
                         const int feat_stride, const float base_size, const float base_scale,
                         const vector<float> &scales, const vector<float> &ratios);

    void set_resize_scale(const float resize_scale_width, const float resize_scale_height);

    void set_resize(const float resized_width, const float resized_height);

    void set_conf_thresh(const float conf_thresh) { conf_thresh_ = conf_thresh; }

    void set_nms_thresh(const float nms_thresh) { nms_thresh_ = nms_thresh; }

protected:
    struct Box {
        float region[4];
        // for a predicted box, the score is the predicted score
        // for a ground truth box, the score is the area of it
        float score;
        // the label of this box
        int label;

        float get_area() const {
            return (region[2] - region[0] + 1) * (region[3] - region[1] + 1);
        }
    };

    struct Landmark {
        float region[10];
    };

    // 辅助排序
    struct by_box_score {
        const vector<Box> &all_boxes_;

        by_box_score(const vector<Box> &all_boxes) : all_boxes_(all_boxes) {}

        bool operator()(const int &a, const int &b) const {
            return all_boxes_[a].score > all_boxes_[b].score;
        }
    };

    void prepare_input_blob(const vector<Mat> &imgs);

    void parse_detect_result();

    void perform_nms();

private:
    // 类别名
    vector<string> classes_;
    int class_num_;

    // 用来生成 anchor 的特征图 blob
    int feat_blob_num_;
    vector<string> feat_blob_names_;
    vector<bool> ratio_fix_heights_;
    vector<int> feat_strides_;
    vector<float> base_sizes_, base_scales_;
    vector<vector<float> > scales_, ratios_;

    // 生成的 anchor 结果
    vector<int> rois_blob_shape_;
    std::shared_ptr<snr::Tensor> rois_blob_;

    // 输入图片数量，通道数
    int img_num_, channel_;
    // 输入图片的高度和宽度，设置为 float 类型方便后面做浮点数运算
    float src_height_, src_width_;
    // 原图经过 resize_scale_ 之后输入到网络
    float resize_scale_width_;
    float resize_scale_height_;
    // 输入到网络的图片高度和宽度
    int height_, width_; // img_size_;

    // 检测的得分阈值
    float conf_thresh_;

    // nms 的 IOU 阈值
    float nms_thresh_;
    // NMS 之前的最大框的数量
    int max_roi_before_nms_;

    // data 和 im_info 的 blob
    std::shared_ptr<snr::Tensor> data_blob_;
    std::shared_ptr<snr::Tensor> im_info_blob_;

    // 分类和回归的 blob
    std::shared_ptr<snr::Tensor> cls_prob_blob_;
    std::shared_ptr<snr::Tensor> bbox_pred_blob_;
    std::shared_ptr<snr::Tensor> ldmk_pred_blob_;

//    const float *cls_prob_blob_;

    // 临时存储每一张图片的 box
    vector<vector<Box> > result_boxes_;
    vector<vector<Landmark> > result_landmarks_;
    // NMS 后的下标
    vector<vector<int> > keep_inds_;

    std::string model_path_; ////////////////////////////
    int num_threads_ = 2; //!设置使用多少个核数

};

#endif //SSD_DEPLOY_DETECTOR_HPP
