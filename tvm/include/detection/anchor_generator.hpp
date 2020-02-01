//
// Created by geyongtao on 2019-07-06.
//

#ifndef SSD_DEPLOY_ANCHOR_GENERATOR_HPP
#define SSD_DEPLOY_ANCHOR_GENERATOR_HPP
#include <iostream>
#include <memory>
#include "../../include/engine/tensor.h"

using namespace std;
using namespace snr;

class SSDAnchorGenerator {

public:
    SSDAnchorGenerator() : generate_pattern_(true) {
        result_shape_.resize(2, 4);
        anchor_patterns_ = std::make_shared<Tensor>();
        result_blob_ = std::make_shared<Tensor>();
    }

    SSDAnchorGenerator(vector<int> &feat_blob_shape, const bool ratio_fix_height,
                       const int feet_stride, const float base_size, const float base_scale) :
            feat_blob_shape_(feat_blob_shape), feat_stride_(feet_stride),
            base_size_(base_size), base_scale_(base_scale),
            ratio_fix_height_(ratio_fix_height), generate_pattern_(true),
            result_height_(0), result_width_(0) {
        result_shape_.resize(2, 4);
    }

    ~SSDAnchorGenerator() {}

    void set(int src_height_, int src_width_, const bool ratio_fix_height,
             const int feet_stride, const float base_size, const float base_scale);

    void add_scale(const float scale) { scales_.push_back(scale); }

    void add_ratio(const float ratio) { ratios_.push_back(ratio); }

    // 生成 anchor, 返回值表示 anchor 是否有变动
    bool generate_anchor();

    std::shared_ptr<Tensor> get_result_blob() { return result_blob_; }

    // 对指定范围内的点生成 anchor
    // 行号： [ row_start, row_end )
    // 列号： [ col_start, col_end )
    void generate_anchors(const int row_start, const int row_end, const int col_start,
                          const int col_end, const int anchor_height, const int anchor_width,
                          const float *anchor_pattern_data, float *anchor_data);

private:
    // 要用来生成 anchor 的特征图
//    std::shared_ptr<Tensor> feat_blob_;
    vector<int> feat_blob_shape_;
    // 步长
    int feat_stride_;
    // 框的大小和 scale
    float base_size_, base_scale_;
    // 尺度和宽高比
    vector<float> scales_, ratios_;
    // 存储 anchor 的样式
    std::shared_ptr<Tensor> anchor_patterns_;
    int anchor_pattern_num_;
    // 是否按照高度来进行 scale (默认是根据面积进行 scale)
    bool ratio_fix_height_;
    // 是否需要生成 anchor 的样式
    bool generate_pattern_;

    // anchor 样式的坐标数量和指针
    int cor_num_;// 坐标数量 anchor_pattern_num_ * 4
    vector<float *> new_anchor_datas_;

    // 返回结果
    std::shared_ptr<Tensor> result_blob_;//一个 feature map 的 anchor
    vector<int> result_shape_;
    int result_height_, result_width_;
};

#endif //SSD_DEPLOY_ANCHOR_GENERATOR_HPP
