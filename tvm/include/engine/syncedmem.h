//
// Created by geyongtao on 2019-07-06.
//

#ifndef SSD_DEPLOY_SYNCEDMEM_H
#define SSD_DEPLOY_SYNCEDMEM_H
#include <cstdlib>
#include <memory.h>

namespace snr {

    class SyncedMemory {
    public:
        SyncedMemory();

        explicit SyncedMemory(size_t size);

        ~SyncedMemory();

        const void *cpu_data();

        void set_cpu_data(void *data);

        void *mutable_cpu_data();

        const void *gpu_data();

        void set_gpu_data(void *data);

        void *mutable_gpu_data();

        size_t size() { return size_; }

        enum SyncedHead {
            UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED
        };

        SyncedHead head() { return head_; }

    private:
        size_t size_;
        SyncedHead head_;
        void *cpu_ptr_;
        void *gpu_ptr_;
        bool own_cpu_data_;
        bool own_gpu_data_;
    private:
        void to_cpu();
        void to_gpu();
    };


} //namespace snr
#endif //SSD_DEPLOY_SYNCEDMEM_H
