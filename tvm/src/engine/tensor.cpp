//
// Created by geyongtao on 2019-07-06.
//

#include "../../include/engine/tensor.h"
#include <climits>

//！这里是用来编译android版本，临时注释掉
#define  USE_HOST 0

#if USE_HOST
#include <glog/logging.h>
#endif

namespace  snr {

    Tensor::Tensor(const std::vector<int> &shape) {
        capacity_ = 0;
        Resize(shape);
    }

    void Tensor::Resize(const std::vector<int> &shape) {
        count_ = 1;
        for (size_t i = 0; i < shape.size(); i++) {

#if USE_HOST
            CHECK_GE(shape[i], 0);
        CHECK_LE(shape[i], INT_MAX / count_);
#endif

            count_ *= shape[i];
        }
        shape_ = shape;
        if (count_ > capacity_) {
            capacity_ = count_;
            data_.reset(new SyncedMemory(capacity_ * sizeof(float)));
        }
    }

    void Tensor::ResizeLike(const Tensor &other) {
        Resize(other.Shape());
    }

    const float* Tensor::cpu_data() const {

#if USE_HOST
        CHECK(data_);
#endif
        return (const float*)data_->cpu_data();
    }

    const float* Tensor::gpu_data() const {

#if USE_HOST
        CHECK(data_);
#endif

        return (const float*)data_->gpu_data();
    }

    float* Tensor::mutable_cpu_data(){

#if USE_HOST
        CHECK(data_);
#endif

        return (float*)data_->mutable_cpu_data();
    }

    float* Tensor::mutable_gpu_data() {

#if USE_HOST
        CHECK(data_);
#endif
        return (float*)data_->mutable_gpu_data();
    }

    void Tensor::ShareData(float *data, int source_flag) {

#if USE_HOST
        CHECK(data);
#endif

        size_t size = count_ * sizeof(float);
        if (data_->size() != size) {
            data_.reset(new SyncedMemory(size));
        }
        if (source_flag == 0) {
            data_->set_cpu_data(data);
        } else {
            data_->set_gpu_data(data);
        }
    }


    void Tensor::TensorToVVector(
            const std::shared_ptr<Tensor> &tensor,
            std::vector<std::vector<float>> &vector_data) {
        vector_data.clear();
        int num = tensor->Shape()[0];
        int output_dim = tensor->Shape()[1];
        const float *blob_data = tensor->cpu_data();
        for (int i = 0; i < num; i++) {
            std::vector<float> output_data(blob_data, blob_data + output_dim);
            vector_data.push_back(output_data);
            blob_data += output_dim;
        }
    }

    void Tensor::MatVectorToTensor(
            const std::vector<cv::Mat> &mats,
            std::shared_ptr<Tensor> &tensor) {
        auto tensor_data = tensor->mutable_cpu_data();
        for (size_t i = 0; i < mats.size(); i++) {
            std::vector<cv::Mat> channels;
            for (int c = 0; c < mats[i].channels(); c++) {
                cv::Mat channel(mats[i].size(), CV_32FC1, tensor_data);
                tensor_data += channel.size().area();
                channels.push_back(channel);
            }
            cv::split(mats[i], channels);
        }
    }


}