//
// Created by geyongtao on 2019-07-06.
//

#include "../../include/engine/syncedmem.h"

#ifndef ANDROID
//#include <glog/logging.h>
#endif

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace snr {

    SyncedMemory::SyncedMemory()
            : cpu_ptr_(nullptr), gpu_ptr_(nullptr), size_(0), head_(UNINITIALIZED),
              own_cpu_data_(false), own_gpu_data_(false) {

    }

    SyncedMemory::SyncedMemory(size_t size)
            : cpu_ptr_(nullptr), gpu_ptr_(nullptr), size_(size), head_(UNINITIALIZED),
              own_cpu_data_(false), own_gpu_data_(false) {
    }

    SyncedMemory::~SyncedMemory() {
        if(cpu_ptr_ && own_cpu_data_) {
            free(cpu_ptr_);
        }
        if (gpu_ptr_ && own_gpu_data_) {
#ifdef WITH_CUDA
            CHECK_EQ(cudaFree(gpu_ptr_), cudaSuccess);
#endif
        }
    }

    inline void SyncedMemory::to_cpu() {
        switch (head_) {
            case UNINITIALIZED:
                cpu_ptr_ = malloc(size_);
                memset(cpu_ptr_, 0, size_);
                own_cpu_data_ = true;
                head_ = HEAD_AT_CPU;
                break;
            case HEAD_AT_GPU:
#ifdef WITH_CUDA
                if (cpu_ptr_ == nullptr) {
                cpu_ptr_ = malloc(size_);
                own_cpu_data_ = true;
            }
            CHECK_EQ(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_, cudaMemcpyDeviceToHost), cudaSuccess);
            head_ = SYNCED;
#endif
                break;
            case HEAD_AT_CPU:
            case SYNCED:
                break;
        }
    }

    inline void SyncedMemory::to_gpu() {
#ifdef WITH_CUDA
        switch (head_) {
        case UNINITIALIZED:
            CHECK_EQ(cudaMalloc(&gpu_ptr_, size_), cudaSuccess);
            CHECK_EQ(cudaMemset(gpu_ptr_, 0, size_), cudaSuccess);
            own_gpu_data_ = true;
            head_ = HEAD_AT_GPU;
            break;
        case HEAD_AT_CPU:
            if (gpu_ptr_ == nullptr) {
                CHECK_EQ(cudaMalloc(&gpu_ptr_, size_), cudaSuccess);
                own_gpu_data_ = true;
            }
            CHECK_EQ(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice), cudaSuccess);
            head_ = SYNCED;
            break;
        case HEAD_AT_GPU:
        case SYNCED:
            break;
    }
#endif
    }

    const void *SyncedMemory::cpu_data() {
        to_cpu();
        return (const void*)cpu_ptr_;
    }

    void SyncedMemory::set_cpu_data(void *data) {

#if USE_HOST
        CHECK(data);
#endif

        if (own_cpu_data_) {
            free(cpu_ptr_);
        }
        cpu_ptr_ = data;
        head_ = HEAD_AT_CPU;
        own_cpu_data_ = false;
    }

    void *SyncedMemory::mutable_cpu_data(){
        to_cpu();
        head_ = HEAD_AT_CPU;
        return (void*)cpu_ptr_;
    }


    const void *SyncedMemory::gpu_data() {
        to_gpu();
        return (const void*)gpu_ptr_;
    }

    void SyncedMemory::set_gpu_data(void *data) {

#if USE_HOST
        CHECK(data);
#endif

        if (own_gpu_data_) {
#ifdef WITH_CUDA
            CHECK_EQ(cudaFree(gpu_ptr_), cudaSuccess);
#endif
        }
        gpu_ptr_ = data;
        head_ = HEAD_AT_GPU;
        own_gpu_data_ = false;
    }

    void *SyncedMemory::mutable_gpu_data(){
        to_gpu();
        head_ = HEAD_AT_GPU;
        return (void*)gpu_ptr_;
    }



} //namespace snr