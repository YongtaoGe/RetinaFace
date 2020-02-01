//
// Created by geyongtao on 2019-07-06.
//

#ifndef SSD_DEPLOY_TENSOR_H
#define SSD_DEPLOY_TENSOR_H
#include <memory>
#include <vector>
#include <opencv2/core/core.hpp>
#include "syncedmem.h"

namespace snr {

    class Tensor {
    public:
        Tensor():
                data_(), count_(0), capacity_(0) {}
        explicit Tensor(const std::vector<int> &shape);

        void Resize(const std::vector<int> &shape);
        void ResizeLike(const Tensor &other);

        const float *cpu_data() const;
        const float *gpu_data() const;

        float *mutable_cpu_data();
        float *mutable_gpu_data();

        void ShareData(float *data, int source_flag = 0);

        inline const std::vector<int> &Shape() const {
            return shape_;
        }

        static void TensorToVVector(
                const std::shared_ptr<Tensor> &tensor,
                std::vector<std::vector<float>> &vector_data);

        static void MatVectorToTensor(
                const std::vector<cv::Mat> &mats,
                std::shared_ptr<Tensor> &tensor);

        Tensor(const Tensor &) = delete;
        Tensor&operator=(const Tensor &) = delete;

    private:
        std::shared_ptr<SyncedMemory> data_;
        int count_;
        int capacity_;
        std::vector<int> shape_;
    };

} // namespace snalog
#endif //SSD_DEPLOY_TENSOR_H
