//
// Created by geyongtao on 2019-07-06.
//

#include "../../include/detection/anchor_generator.hpp"
#include <cmath>

void SSDAnchorGenerator::set(const int src_height_, const int src_width_,
                             const bool ratio_fix_height, const int feat_stride,
                             const float base_size, const float base_scale) {
    generate_pattern_ = true;
    feat_blob_shape_.push_back(int(src_height_/feat_stride));
    feat_blob_shape_.push_back(int(src_width_/feat_stride));
    ratio_fix_height_ = ratio_fix_height;
    feat_stride_ = feat_stride;
    base_size_ = base_size;
    base_scale_ = base_scale;
    result_height_ = 0;
    result_width_ = 0;
    anchor_patterns_ = std::make_shared<Tensor>();
    result_blob_ = std::make_shared<Tensor>();
}

bool SSDAnchorGenerator::generate_anchor() {
//    vector<int> shape = feat_blob_shape_;
    if (result_height_ == feat_blob_shape_[0] && result_width_ == feat_blob_shape_[1]) {
        return false;
    }
    if (generate_pattern_) {
        const int scale_num = scales_.size(); // 尺度
        const int ratio_num = ratios_.size(); // 宽高比
        // 每个点生成anchor的个数，代码中都为1,
        anchor_pattern_num_ = scale_num * ratio_num;
        vector<int> anchor_shape(2);
        anchor_shape[0] = 4;
        anchor_shape[1] = anchor_pattern_num_;//anchor_pattern_num_ 指的是同一个位置的anchor数量
        // 4 * （同一个位置的anchor数量）
        anchor_patterns_->Resize(anchor_shape);

        // 生成所有样式的 anchor
        float *anchor_pattern_data = anchor_patterns_->mutable_cpu_data();
        // 计算基准框的中心坐标
//        float base_ctr_x = 0.5 * (base_size_ - 1);
        float base_ctr_x = 0.5 * feat_stride_;
        float base_ctr_y = base_ctr_x;
        // 计算基准框的面积
        float base_area = base_size_ * base_size_;

        // 左上角坐标和右下角坐标的 4 个指针
        vector<float *> anchor_pattern_datas(4);
        anchor_pattern_datas[0] = anchor_pattern_data;
        for (int i = 1; i < 4; i++) {
            anchor_pattern_datas[i] = anchor_pattern_datas[i - 1] + anchor_pattern_num_;
        }

        // 枚举每一种宽高比
        for (int ratio_idx = 0; ratio_idx < ratio_num; ratio_idx++) {
            // the ratio is height / width, so height = width * ratio
            // scale_area = height * width = width * (width * ratio)
            // so the width = sqrt(scale_area / ratio)
            const float &anchor_ratio = ratios_[ratio_idx];

            const float size_ratio = base_area / anchor_ratio;
            const float ratio_width = ratio_fix_height_ ? base_size_ / anchor_ratio
                                                        : round(sqrt(size_ratio));
            const float ratio_height = ratio_fix_height_ ? base_size_
                                                         : round(ratio_width * anchor_ratio);

            // 对调整宽高比之后的框进行缩放
            for (int scale_idx = 0; scale_idx < scale_num; scale_idx++) {
                // pattern 的序号
                const int pattern_id = ratio_idx * scale_num + scale_idx;

                const float anchor_scale = scales_[scale_idx];
                const float new_height = ratio_height * anchor_scale;
                const float new_width = ratio_width * anchor_scale;
//                const float half_height = 0.5 * (new_height - 1);
//                const float half_width = 0.5 * (new_width - 1);
                const float half_height = 0.5 * (new_height);
                const float half_width = 0.5 * (new_width);

                anchor_pattern_datas[0][pattern_id] = base_ctr_x - half_width;
                anchor_pattern_datas[1][pattern_id] = base_ctr_y - half_height;
                anchor_pattern_datas[2][pattern_id] = base_ctr_x + half_width;
                anchor_pattern_datas[3][pattern_id] = base_ctr_y + half_height;
            }
        }

        cout << "totally " << anchor_pattern_num_ << " Anchors: " << endl;
        // const float *anchor_data = anchor_patterns_.cpu_data();
//        for (int i = 0; i < anchor_pattern_num_; i++) {
//            cout << anchor_pattern_datas[0][i] << " " << anchor_pattern_datas[1][i]
//                 << " " << anchor_pattern_datas[2][i] << " " << anchor_pattern_datas[3][i] << endl;
//        }

        // 坐标数量
        cor_num_ = anchor_pattern_num_ * 4;
        new_anchor_datas_.resize(cor_num_);

        generate_pattern_ = false;
    }
    // 先根据ratio和scale生成每个点对应的anchor的模式，再根据feature map的上点生成所有anchor
    // result_shape : 4 * anchor_num
    result_shape_[1] = anchor_pattern_num_ * feat_blob_shape_[0] * feat_blob_shape_[1];
    result_blob_->Resize(result_shape_);
    generate_anchors(0, feat_blob_shape_[0], 0, feat_blob_shape_[1], feat_blob_shape_[0], feat_blob_shape_[1],
                     anchor_patterns_->cpu_data(), result_blob_->mutable_cpu_data());

    return true;
}

// 对指定范围内的点生成 anchor
// 行号： [ row_start, row_end )
// 列号： [ col_start, col_end )
// *anchor_data的顺序是
//void SSDAnchorGenerator::generate_anchors(const int row_start,
//                                          const int row_end, const int col_start, const int col_end,
//                                          const int anchor_height, const int anchor_width,
//                                          const float *anchor_pattern_data, float *anchor_data) {
//
//
//
//    for (int i=0;i<12;i++){
//        cout<<"anchor pattern: "<<i<<":  "<<anchor_pattern_data[i]<<endl;
//    }
//    const int anchor_channel_size = anchor_height * anchor_width;
//    // 对应每一种坐标样式的每一维坐标
//    new_anchor_datas_[0] = anchor_data + row_start * anchor_width + col_start;
//    for (int cor_idx = 1; cor_idx < cor_num_; cor_idx++) {
//        new_anchor_datas_[cor_idx] = new_anchor_datas_[cor_idx - 1] + anchor_channel_size;
//    }
//    // 行号: [ row_start, row_end )
//    for (int row = row_start; row < row_end; row++) {
//        // 计算这一行的 y 在输入图片中的实际偏移量
//        int shift_y = row * feat_stride_;
//        // 列号: [ col_start, col_end )
//        for (int col = col_start; col < col_end; col++) {
//            // 计算该位置的 x 在输入图片中的实际偏移量
//            int shift_x = col * feat_stride_;
//            // 对于每一种坐标样式的每一维坐标
//            for (int cor_idx = 0; cor_idx < cor_num_; cor_idx++) {
//
//                cout<<"((cor_idx / anchor_pattern_num_) & 1)"<<((cor_idx % anchor_pattern_num_)==0)<<endl;
//
//                if ((cor_idx / anchor_pattern_num_) & 1) {
//                    *new_anchor_datas_[cor_idx] = shift_y + anchor_pattern_data[cor_idx];
//                } else {
//                    *new_anchor_datas_[cor_idx] = shift_x + anchor_pattern_data[cor_idx];
//                }
////                cout<<new_anchor_datas_[cor_idx] << endl;
//
//                // 偏移到下一个位置
//                new_anchor_datas_[cor_idx]++;
//            }
//        }
//
////        for (int i=0;i<12;i++){
////            cout<<"new_anchor_datas_: "<<i<<":  "<<*new_anchor_datas_[i]<<endl;
////        }
////        cout<<col_start<<", "<<col_end<<endl;
////        cout<<new_anchor_datas_[0]<<endl;
//        // 从行首偏移到 anchor_blob_shape_[3]
//        for (int cor_idx = 0; cor_idx < cor_num_; cor_idx++) {
////            cout<<"before: "<<new_anchor_datas_[cor_idx]<<endl;
//            new_anchor_datas_[cor_idx] += col_start;
////            cout<<"after: "<<new_anchor_datas_[cor_idx]<<endl;
//        }
//    }
//
//
//    for (int i=0;i<12;i++){
//
//            cout<<"anchor_data: "<<i<<":  "<<anchor_data[i]<<endl;
//    }
//    for (int i=0;i<12;i++){
//        cout<<"result_blob_"<<result_blob_->mutable_cpu_data()[i]<<endl;
//    }
//
//
//}



// 对指定范围内的点生成 anchor
// 行号： [ row_start, row_end )
// 列号： [ col_start, col_end )
// *anchor_data的顺序是
void SSDAnchorGenerator::generate_anchors(const int row_start,
                                          const int row_end, const int col_start, const int col_end,
                                          const int anchor_height, const int anchor_width,
                                          const float *anchor_pattern_data, float *anchor_data) {



//    for (int i=0;i<12;i++){
//        cout<<"anchor pattern: "<<i<<":  "<<anchor_pattern_data[i]<<endl;
//    }
    const int anchor_channel_size = anchor_height * anchor_width;
    // 对应每一种坐标样式的每一维坐标
    new_anchor_datas_[0] = anchor_data + row_start * anchor_width + col_start;
    for (int cor_idx = 1; cor_idx < cor_num_; cor_idx++) {

        if ((cor_idx % anchor_pattern_num_)==0) {

            new_anchor_datas_[cor_idx] = new_anchor_datas_[0] + cor_idx * anchor_channel_size;
//            cout<<"cor_idx * anchor_channel_size: "<<cor_idx * anchor_channel_size<<endl;

        }
        else {

            new_anchor_datas_[cor_idx] = new_anchor_datas_[cor_idx - 1] + 1;
        }

//        cout<<new_anchor_datas_[cor_idx]<<endl;

    }


    // 行号: [ row_start, row_end )
    for (int row = row_start; row < row_end; row++) {
        // 计算这一行的 y 在输入图片中的实际偏移量
        int shift_y = row * feat_stride_;
        // 列号: [ col_start, col_end )
        for (int col = col_start; col < col_end; col++) {
            // 计算该位置的 x 在输入图片中的实际偏移量
            int shift_x = col * feat_stride_;
            // 对于每一种坐标样式的每一维坐标
            for (int cor_idx = 0; cor_idx < cor_num_; cor_idx++) {

//                cout<<"((cor_idx / anchor_pattern_num_) & 1)"<<((cor_idx / anchor_pattern_num_) & 1)<<endl;

                if ((cor_idx / anchor_pattern_num_) & 1) {
                    *new_anchor_datas_[cor_idx] = shift_y + anchor_pattern_data[cor_idx];
                } else {
                    *new_anchor_datas_[cor_idx] = shift_x + anchor_pattern_data[cor_idx];
                }
//                cout<<new_anchor_datas_[cor_idx] << endl;

                // 偏移到下一个位置
                new_anchor_datas_[cor_idx] += anchor_pattern_num_;
            }
        }

//        for (int i=0;i<12;i++){
//            cout<<"new_anchor_datas_: "<<i<<":  "<<*new_anchor_datas_[i]<<endl;
//        }
//        cout<<col_start<<", "<<col_end<<endl;
//        cout<<new_anchor_datas_[0]<<endl;
        // 从行首偏移到 anchor_blob_shape_[3]
        for (int cor_idx = 0; cor_idx < cor_num_; cor_idx++) {
//            cout<<"before: "<<new_anchor_datas_[cor_idx]<<endl;
            new_anchor_datas_[cor_idx] += col_start;
//            cout<<"after: "<<new_anchor_datas_[cor_idx]<<endl;
        }
    }

//    for (int i=0;i<12;i++){
//
//        cout<<"anchor_data: "<<i<<":  "<<anchor_data[i]<<endl;
//    }
//    for (int i=0;i<12;i++){
//        cout<<"result_blob_"<<result_blob_->mutable_cpu_data()[i]<<endl;
//    }


}