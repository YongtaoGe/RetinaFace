//
// Created by geyongtao on 2019-07-07.
//
#include "../../include/detection/detector.hpp"
#include "../../include/detection/anchor_generator.hpp"
//#include "net.h"
//#include "benchmark.h"
//#include "quantize.h"
//#include "sn_depthwise_ssd_quantized_no_relu.id.h"

#ifdef ANDROID
#include<android/log.h>
#define TAG "face" // 这个是自定义的LOG的标识
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG ,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG ,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,TAG ,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG ,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,TAG ,__VA_ARGS__) // 定义LOGF类型
#endif


SSDDetector::SSDDetector(const int gpuid, const int src_width, const int src_height) :
        src_height_(src_height), src_width_(src_width) {
    img_num_ = 1;
//    resize_scale_width_ = 0.333333;//0.6;// 0.45; //resize the orignal pic to feed cnn
//    resize_scale_height_ = 0.333333;
    classes_.push_back("__background__");
    classes_.push_back("face");

    conf_thresh_ = 0.8;
    nms_thresh_ = 0.25;
    max_roi_before_nms_ = 400;
    channel_ = 3;

//    height_ = round(src_height_ * resize_scale_height_);
//    width_ = round(src_width_ * resize_scale_width_);

    class_num_ = classes_.size();
    result_boxes_.resize(img_num_);
    result_landmarks_.resize(img_num_);
    keep_inds_.resize(img_num_);
}

void SSDDetector::set_resize_scale(const float resize_scale_width, const float resize_scale_height) {
    resize_scale_width_ = resize_scale_width;
    resize_scale_height_ = resize_scale_height;
    height_ = round(src_height_ * resize_scale_height_);
    width_ = round(src_width_ * resize_scale_width_);
}

void SSDDetector::set_resize(const float resized_width, const float resized_height) {
    resize_scale_width_ = resized_width/src_width_;
    resize_scale_height_ = resized_height/src_height_;
    height_ = round(src_height_ * resize_scale_height_);
    width_ = round(src_width_ * resize_scale_width_);
}

// subtract mean and to chw
void SSDDetector::prepare_input_blob(const vector<cv::Mat> &imgs)
{
    // Reshape 图片 blob
    vector<int> data_shape(4);
    data_shape[0] = img_num_;
    data_shape[1] = channel_;
    data_shape[2] = height_;
    data_shape[3] = width_;
    data_blob_ = std::make_shared<snr::Tensor>(data_shape);

    unsigned long time_0, time_1;

    float *dest = data_blob_->mutable_cpu_data();

    for (int i = 0; i < imgs.size(); i++)
    {
        //Using OpenCV pre-precess API
        cv::Mat src = imgs[i];
        cv::Mat img_float;

        double start = get_current_time();
        //cv::Scalar pixel_mean(102.9801, 115.9465, 122.7717);
//        cv::Scalar pixel_mean(128.0f, 128.0f, 128.0f);
        cv::Scalar pixel_mean(104.0f, 117.0f, 123.0f);
        if (src.channels() == 3)
            src.convertTo(img_float, CV_32FC3);
        else
            src.convertTo(img_float, CV_32FC1);
        double end = get_current_time();
        std::cout << "ConvertTo32F cost: " << (end - start) << " ms" << std::endl;

        start = get_current_time();
        cv::Mat mean_mask = cv::Mat(img_float.rows, img_float.cols, CV_32FC3, pixel_mean);
        cv::Mat img_normalized;
        cv::subtract(img_float, mean_mask, img_normalized);
        end = get_current_time();
        std::cout << "Normaliz cost: " << (end - start) << " ms" << std::endl;

        start = get_current_time();
        std::cout<<"prepare data blob"<<std::endl;
        std::cout<<"resize_scale_width_: "<<resize_scale_width_<<std::endl;
        std::cout<<"resize_scale_height_: "<<resize_scale_height_<<std::endl;

        if(width_!=img_normalized.cols||height_!=img_normalized.rows)
            cv::resize(img_normalized, img_normalized, cv::Size(width_, height_), cv::INTER_LINEAR);
//            cv::resize(img_normalized, img_normalized, cv::Size(), resize_scale_width_, resize_scale_height_, cv::INTER_LINEAR);
        end = get_current_time();
//        cout << "img_normalized: "<< img_normalized << endl;
//        cout << "img_normalized (python)  = " << endl << format(img_normalized, Formatter::FMT_PYTHON) << endl << endl;
//        cout << "img_normalized row: 0~2 = "<< endl << " "  << img_normalized.rowRange(0, 2) << endl << endl;
        std::cout << "Resize cost: " << (end - start) << " ms" << std::endl;

        std::cout << img_normalized.size() << std::endl;

        cout<<"height_: "<<height_<<endl;
        cout<<"width_: "<<width_<<endl;

        start = get_current_time();
        for (int h = 0; h < height_; ++h)
        {
            const float *ptr = img_normalized.ptr<float>(h);
            int img_index = 0;
            for (int w = 0; w < width_; ++w)
            {
                for (int c = 0; c < channel_; ++c)
                {
                    dest[(c * height_ + h) * width_ + w] = static_cast<float>(ptr[img_index++]);
                }
            }
        }

        dest += (data_blob_->Shape()[1] * data_blob_->Shape()[2] * data_blob_->Shape()[3]);

        end = get_current_time();
        std::cout << "Data copy cost: " << (end - start) << " ms" << std::endl;

    }
}


void SSDDetector::generate_anchors(){
    rois_blob_shape_.resize(2);
    rois_blob_shape_[0] = 4;

    // 根据要检测的 blob 名称获取 blob, 计算最终生成的 anchor blob 的大小
    feat_blob_num_ = feat_blob_names_.size();
    vector<SSDAnchorGenerator> ssd_anchor_generators(feat_blob_num_);
    rois_blob_shape_[1] = 0;
    for (int i = 0; i < feat_blob_num_; i++) {
        // 根据要检测的 blob 名称获取 blob
//        std::shared_ptr<Tensor> feat_tensor = outputs[i];
        ssd_anchor_generators[i].set(height_, width_, ratio_fix_heights_[i],
                                     feat_strides_[i], base_sizes_[i], base_scales_[i]);

        vector<float> &scales = scales_[i];
        for (int j = 0; j < scales.size(); j++) {
            ssd_anchor_generators[i].add_scale(scales[j]);
        }

        vector<float> &ratios = ratios_[i];
        for (int j = 0; j < ratios.size(); j++) {
            ssd_anchor_generators[i].add_ratio(ratios[j]);
        }

        // 生成 anchor
        cout << "generating anchor for " << feat_blob_names_[i] << endl;
        ssd_anchor_generators[i].generate_anchor();//产生的结果在 result_blob_
        cout << "generating anchor for " << feat_blob_names_[i] << " sussessfully!" << endl;

        // 累加计算最终的 anchor blob 的大小
        rois_blob_shape_[1] += ssd_anchor_generators[i].get_result_blob()->Shape()[1];
    }

    // 分配 rois_blob_ 的空间
    rois_blob_ = std::make_shared<Tensor>(rois_blob_shape_);
    // 初始化最终的 anchor blob
    float *anchor_blob_data = rois_blob_->mutable_cpu_data();
    for (int i = 0; i < feat_blob_num_; i++) {
        std::shared_ptr<Tensor> sdk_anchor_tensor = ssd_anchor_generators[i].get_result_blob();
        const float *ssd_anchor_data = sdk_anchor_tensor->cpu_data();



        const int count = sdk_anchor_tensor->Shape()[1];//每个点anchor数量*featmap_size
        // 按照 channel 复制数据
        for (int j = 0; j < rois_blob_shape_[0]; j++) {
            memcpy(anchor_blob_data + j * rois_blob_shape_[1], ssd_anchor_data + j * count, count * sizeof(float));
        }

//        if (i==0)
//        {
//            for (int i=0;i<12;i++){
//                cout<<"ssd_anchor_data: "<<ssd_anchor_data[i]<<endl;
//                cout<<"anchor_blob_data: "<<anchor_blob_data[i]<<endl;
//                cout<<"roi blob: "<<rois_blob_->mutable_cpu_data()[i]<<endl;
//            }
//
//        }
        anchor_blob_data += count;
    }

    std::cout<<"rois_blob_shape_[0]: "<<rois_blob_shape_[0]<<std::endl;
    std::cout<<"rois_blob_shape_[1]: "<<rois_blob_shape_[1]<<std::endl;
}

void SSDDetector::add_detect_blob(
        const string &feat_blob_name_,
        const bool ratio_fix_height,
        const int feat_stride,
        const float base_size,
        const float base_scale,
        const vector<float> &scales,
        const vector<float> &ratios) {
    feat_blob_names_.push_back(feat_blob_name_);
    ratio_fix_heights_.push_back(ratio_fix_height);
    feat_strides_.push_back(feat_stride);
    base_sizes_.push_back(base_size);
    base_scales_.push_back(base_scale);
    scales_.push_back(scales);
    ratios_.push_back(ratios);
}


void Mat_to_CHW(float *data, cv::Mat &frame)
{
    assert(data && !frame.empty());
    unsigned int volChl = 128 * 128;

    for(int c = 0; c < 3; ++c)
    {
        for (unsigned j = 0; j < volChl; ++j)
            data[c*volChl + j] = static_cast<float>(float(frame.data[j * 3 + c]) / 255.0);
    }

}


void SSDDetector::im_detect(const std::string &lib_path,
                            const std::string &graph_path,
                            const std::string &param_path,
                            const std::string &image_path,
                            vector<vector<Rect> > &boxes,
                            vector<vector<float> > &scores,
                            vector<vector<Point2f> > &ldmks)
{
    // tvm module for compiled functions
    tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(lib_path);

    // json graph
    std::ifstream json_in(graph_path, std::ios::in);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();

    // parameters in binary
    std::ifstream params_in(param_path, std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    // parameters need to be TVMByteArray type to indicate the binary data
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();

    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;

    // get global function module for graph runtime
    tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);

    const int N = 1;
    const int C = 3;
    const int H = height_;
    const int W = width_;
    DLTensor* x;
    int in_ndim = 4;
    //int64_t in_shape[4] = {1, 3, 224, 224};
    int64_t in_shape[4] = {N, C, H, W};

    TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);

    // load image data saved in binary
//    std::ifstream data_fin(image_path, std::ios::binary);
//    data_fin.read(static_cast<char*>(x->data), C * H * W * sizeof(float));



//    cv::Mat image, frame, input;
//    image = cv::imread(image_path);
//    cv::cvtColor(image, frame, cv::COLOR_BGR2RGB);
//    cv::resize(frame, input,  cv::Size(640,640));
//    float data[640 * 640 * 3];
//    // 在这个函数中 将OpenCV中的图像数据转化为CHW的形式
//    Mat_to_CHW(data, input); //有问题 prepare_data
//    // x为之前的张量类型 data为之前开辟的浮点型空间
//    memcpy(x->data, &data, 3 * 640 * 640 * sizeof(float));

    cv::Mat image;
    std::vector<cv::Mat> img_vector;

    image = cv::imread(image_path);
    img_vector.push_back(image);
    src_height_ = image.rows;
    src_width_ = image.cols;

//    resize_scale_width_=640;
//    resize_scale_height_=640;
    prepare_input_blob(img_vector);
    memcpy(x->data, data_blob_->mutable_cpu_data(), 3 * height_ * width_ * sizeof(float));

    // get the function from the module(set input data)
    tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
    set_input("0", x);

    // get the function from the module(load patameters)
    tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
    load_params(params_arr);

    // get the function from the module(run it)
    tvm::runtime::PackedFunc run = mod.GetFunction("run");
    run();

    DLTensor* loc_out;
    int loc_out_ndim = 2;
    int64_t loc_out_shape[2] = {4, rois_blob_shape_[1]};
    TVMArrayAlloc(loc_out_shape, loc_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &loc_out);

    DLTensor* cls_out;
    int cls_out_ndim = 2;
    int64_t cls_out_shape[2] = {2, rois_blob_shape_[1]};
    TVMArrayAlloc(cls_out_shape, cls_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &cls_out);


    DLTensor* ldmk_out;
    int ldmk_out_ndim = 2;
    int64_t ldmk_out_shape[2] = {10, rois_blob_shape_[1]};
    TVMArrayAlloc(ldmk_out_shape, ldmk_out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &ldmk_out);

    // get the function from the module(get output data)
    tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
    double start = get_current_time();
    get_output(0, loc_out);
    get_output(1, cls_out);
    get_output(2, ldmk_out);
    double end = get_current_time();
    std::cout << "Inference cost: " << (end - start) << " ms" << std::endl;

    // get the maximum position in output vector
//    auto y_iter = static_cast<float*>(loc_out->data);
//    auto max_iter = std::max_element(y_iter, y_iter + 1000);
//    auto max_index = std::distance(y_iter, max_iter);
//    std::cout << "The maximum position in output vector is: " << max_index << std::endl;
//    std::cout << loc_out;
    // 将输出的信息打印出来
//    auto input_tvm = static_cast<float*>(x->data);
//    for (int i = 0; i < 24; i++)
//        std::cout<<input_tvm[i]<<std::endl;

//    auto result = static_cast<float*>(ldmk_out->data);

    vector<int> cls_out_dim(2);
    cls_out_dim[0] = 2;
    cls_out_dim[1] = rois_blob_shape_[1];

    cls_prob_blob_ = std::make_shared<snr::Tensor>(cls_out_dim);

    vector<int> bbox_out_dim(2);
    bbox_out_dim[0] = 4;
    bbox_out_dim[1] = rois_blob_shape_[1];
    bbox_pred_blob_= std::make_shared<snr::Tensor>(bbox_out_dim);

    vector<int> ldmk_out_dim(2);
    ldmk_out_dim[0] = 10;
    ldmk_out_dim[1] = rois_blob_shape_[1];
    ldmk_pred_blob_= std::make_shared<snr::Tensor>(ldmk_out_dim);

    memcpy(cls_prob_blob_->mutable_cpu_data(), cls_out->data, rois_blob_shape_[1] * 2 * sizeof(float));
    memcpy(bbox_pred_blob_->mutable_cpu_data(), loc_out->data, rois_blob_shape_[1] * 4 * sizeof(float));
    memcpy(ldmk_pred_blob_->mutable_cpu_data(), ldmk_out->data, rois_blob_shape_[1] * 10 * sizeof(float));

//    cls_prob_blob_ = outputs[2];
//    bbox_pred_blob_ = outputs[3];
//
//
//    // Reshape 图片 blob
//    vector<int> data_shape(4);
//    data_shape[0] = img_num_;
//    data_shape[1] = channel_;
//    data_shape[2] = height_;
//    data_shape[3] = width_;
//    data_blob_ = std::make_shared<snr::Tensor>(data_shape);
//    float *dest = data_blob_->mutable_cpu_data();



//    std::cout<<(result[0]+result[25575])<<std::endl;
//    std::cout<<(result[1]+result[25576])<<std::endl;
//    std::cout<<(cls_prob_blob_->mutable_cpu_data()[0]+cls_prob_blob_->mutable_cpu_data()[25575])<<std::endl;
//
//    std::cout<<"result[0]: "<<result[0]<<std::endl;
//    std::cout<<"result[25575]: "<<result[25575]<<std::endl;
//    std::cout<<"cls_prob_blob_: "<<cls_prob_blob_->mutable_cpu_data()[0]<<std::endl;
//    std::cout<<"cls_prob_blob_: "<<cls_prob_blob_->mutable_cpu_data()[25575]<<std::endl<<std::endl;
//
//    std::cout<<"bbox_pred_blob_: "<<bbox_pred_blob_->mutable_cpu_data()[0]<<std::endl;
//    std::cout<<"bbox_pred_blob_: "<<bbox_pred_blob_->mutable_cpu_data()[25575]<<std::endl;
//    std::cout<<"bbox_pred_blob_: "<<bbox_pred_blob_->mutable_cpu_data()[51150]<<std::endl;
//    std::cout<<"bbox_pred_blob_: "<<bbox_pred_blob_->mutable_cpu_data()[76725]<<std::endl<<std::endl;
//
//    std::cout<<"cls_prob_blob_: "<<cls_prob_blob_->mutable_cpu_data()[1]<<std::endl;
//    std::cout<<"cls_prob_blob_: "<<cls_prob_blob_->mutable_cpu_data()[25576]<<std::endl<<std::endl;
//
//    std::cout<<"bbox_pred_blob_: "<<bbox_pred_blob_->mutable_cpu_data()[1]<<std::endl;
//    std::cout<<"bbox_pred_blob_: "<<bbox_pred_blob_->mutable_cpu_data()[25576]<<std::endl;
//    std::cout<<"bbox_pred_blob_: "<<bbox_pred_blob_->mutable_cpu_data()[51151]<<std::endl;
//    std::cout<<"bbox_pred_blob_: "<<bbox_pred_blob_->mutable_cpu_data()[76726]<<std::endl<<std::endl;
//
//    std::cout<<"cls_prob_blob_: "<<cls_prob_blob_->mutable_cpu_data()[2]<<std::endl;
//    std::cout<<"cls_prob_blob_: "<<cls_prob_blob_->mutable_cpu_data()[25577]<<std::endl<<std::endl;
//
//    std::cout<<"bbox_pred_blob_: "<<bbox_pred_blob_->mutable_cpu_data()[0]<<std::endl;
//    std::cout<<"bbox_pred_blob_: "<<bbox_pred_blob_->mutable_cpu_data()[25577]<<std::endl;
//    std::cout<<"bbox_pred_blob_: "<<bbox_pred_blob_->mutable_cpu_data()[51152]<<std::endl;
//    std::cout<<"bbox_pred_blob_: "<<bbox_pred_blob_->mutable_cpu_data()[76727]<<std::endl<<std::endl;



//    std::cout<<(result[4]+result[5])<<std::endl;
//    std::cout<<(result[6]+result[7])<<std::endl;

    TVMArrayFree(x);
    TVMArrayFree(loc_out);
    TVMArrayFree(cls_out);
    TVMArrayFree(ldmk_out);


    start = get_current_time();

    parse_detect_result();
    // 进行 NMS
    perform_nms();
//
    // 返回结果
    boxes.resize(img_num_);
    scores.resize(img_num_);
    ldmks.resize(img_num_);
    for (int img_idx = 0; img_idx < img_num_; img_idx++)
    {
        // 获取结果框
        vector<Box> &img_boxes = result_boxes_[img_idx];
        vector<int> img_keep_inds = keep_inds_[img_idx];

        // 获取关键点
        vector<Landmark> &img_ldmks = result_landmarks_[img_idx];

//        for (int i=0; i<img_keep_inds.size();i++){
//            cout<<img_keep_inds.size()<<endl;
//            cout << "img_keep_inds: "<<img_keep_inds[i]<<endl;
//
//        }

        const int img_box_num = img_keep_inds.size();

        if (img_box_num == 0) continue;

        // 为返回结果分配空间
        vector<Rect> &img_rects = boxes[img_idx];
        vector<float> &img_scores = scores[img_idx];
        vector<Point2f> &img_points = ldmks[img_idx];

        img_rects.resize(img_box_num);
        img_scores.resize(img_box_num);
        img_points.resize(img_box_num * 5);

        // 获取每一个结果框
        for (int i = 0; i < img_box_num; i++)
        {
            Box &box = img_boxes[img_keep_inds[i]];
            Rect &rect = img_rects[i];
            Landmark &ldmk_5point = img_ldmks[img_keep_inds[i]];

            // 转化成 (xmin, ymin, w, h) 的形式
            rect.x = box.region[0];
            rect.y = box.region[1];
            rect.width = box.region[2] - box.region[0] + 1;
            rect.height = box.region[3] - box.region[1] + 1;

            img_points[i*5].x = ldmk_5point.region[0];
            img_points[i*5].y = ldmk_5point.region[1];
            img_points[i*5+1].x = ldmk_5point.region[2];
            img_points[i*5+1].y = ldmk_5point.region[3];
            img_points[i*5+2].x = ldmk_5point.region[4];
            img_points[i*5+2].y = ldmk_5point.region[5];
            img_points[i*5+3].x = ldmk_5point.region[6];
            img_points[i*5+3].y = ldmk_5point.region[7];
            img_points[i*5+4].x = ldmk_5point.region[8];
            img_points[i*5+4].y = ldmk_5point.region[9];

            img_scores[i] = box.score;
        }
    }
    end = get_current_time();
    std::cout << "Result cost: " << (end - start) << " ms" << std::endl;



}



void SSDDetector::parse_detect_result()
{
    Box box;
    Landmark ldmk;
    float variances[2] = {0.1, 0.2};
    const int roi_num = rois_blob_->Shape()[1]; //25575

    // 初始化每个 channel 指针
    vector<const float *> roi_datas(4), bbox_datas(4), cls_datas(class_num_), ldmk_datas(10);
    const float *cls_prob_data = cls_prob_blob_->cpu_data();
    const float *bbox_pred_data = bbox_pred_blob_->cpu_data();
    const float *ldmk_pred_data = ldmk_pred_blob_->cpu_data();
    const float *rois_data = rois_blob_->cpu_data();




    // 对于每一张图片
    for (int img_idx = 0; img_idx < img_num_; img_idx++)
    {
        vector<Box> &img_boxes = result_boxes_[img_idx];
        vector<Landmark> &img_landmarks = result_landmarks_[img_idx];
        img_boxes.clear();
        img_landmarks.clear();

        // 所有图片共用同一个 roi blob
        roi_datas[0] = rois_data; //anchor 首地址
        bbox_datas[0] = bbox_pred_data;

        // 4 的含义是 anchor 和 box 需要4维矩阵表示
        for (int i = 1; i < 4; i++)
        {
            roi_datas[i] = roi_datas[i - 1] + roi_num; //anchor 首地址的 roi_num (25575) 个偏移
            bbox_datas[i] = bbox_datas[i - 1] + roi_num;
        }

//        for (int i=0; i<100; i++){
//            cout<<roi_datas[0][i]<<", "<<roi_datas[1][i]<<", "<<roi_datas[2][i]<<", "<<roi_datas[3][i]<<", "<<endl;
//        }


        // 获取分类得分指针
        cls_datas[0] = cls_prob_data;
        for (int i = 1; i < class_num_; i++)
        {
            cls_datas[i] = cls_datas[i - 1] + roi_num;
        }

        // 获取关键点回归指针
        ldmk_datas[0] = ldmk_pred_data;
        for (int i = 1; i < 10; i++)
        {
            ldmk_datas[i] = ldmk_datas[i - 1] + roi_num;
        }


        // 对于每一个 roi
        for (int roi_idx = 0; roi_idx < roi_num; roi_idx++)
        {
            // 获取 roi 并 归一化resize_scale_height_ = 0.333333;
//            const float xmin = std::min(std::max(
//                    float(*(roi_datas[0]) / resize_scale_width_), float(0)), float(src_width_));
//            const float ymin = std::min(std::max(
//                    float(*(roi_datas[1]) / resize_scale_height_), float(0)), float(src_height_));
//            const float xmax = std::min(std::max(
//                    float(*(roi_datas[2]) / resize_scale_width_), float(0)), float(src_width_));
//            const float ymax = std::min(std::max(
//                    float(*(roi_datas[3]) / resize_scale_height_), float(0)), float(src_height_));


            const float xmin = std::min(std::max(
                    float(*(roi_datas[0]) / width_), float(0)), float(src_width_));
            const float ymin = std::min(std::max(
                    float(*(roi_datas[1]) / height_), float(0)), float(src_height_));
            const float xmax = std::min(std::max(
                    float(*(roi_datas[2]) / width_), float(0)), float(src_width_));
            const float ymax = std::min(std::max(
                    float(*(roi_datas[3]) / height_), float(0)), float(src_height_));


            // 转化成 (x, y, w, h) 的形式
            //
//            float width = xmax - xmin + 1;
//            float height = ymax - ymin + 1;
            // anchor [ctr_x, ctr_y, width, height]
            float width = xmax - xmin;
            float height = ymax - ymin;
            float ctr_x = xmin + 0.5 * width;
            float ctr_y = ymin + 0.5 * height;

            // 对于除了背景之外的每一个类别，获取对应的得分和回归后的框
            for (int class_idx = 1; class_idx < class_num_; class_idx++)
            {
                // 获取得分
                box.score = *(cls_datas[class_idx]);
                // 如果得分小于要求的置信度，则忽略这个 roi
                if (box.score < conf_thresh_) continue;
//                cout << box.score << endl;
//                cout << xmin << ", " << ymin << ", " << xmax << ", " << ymax << endl;
                // 获取类别标签
                box.label = class_idx;

                // 对 roi 做回归
//                cout << "anchor center: (" << ctr_x <<", "<< ctr_y <<") " << endl;
//                cout << "bbox_datas: " << bbox_datas[0][0] << endl;
                const float cls_ctr_x = ctr_x + *(bbox_datas[0]) * width * variances[0];
                const float cls_ctr_y = ctr_y + *(bbox_datas[1]) * height* variances[0];
                const float cls_width = width * exp(*(bbox_datas[2]) * variances[1]);
                const float cls_height = height * exp(*(bbox_datas[3]) * variances[1]);

                // 转化成 (xmin, ymin, xmax, ymax) 的形式
                box.region[0] = std::max(cls_ctr_x - 0.5 * cls_width, 0.0);
                box.region[1] = std::max(cls_ctr_y - 0.5 * cls_height, 0.0);
                box.region[2] = std::min(static_cast<float>(cls_ctr_x + 0.5 * cls_width),
                                         src_width_);
                box.region[3] = std::min(static_cast<float>(cls_ctr_y + 0.5 * cls_height),
                                         src_height_);


                box.region[0] = box.region[0] * width_;
                box.region[1] = box.region[1] * height_;
                box.region[2] = box.region[2] * width_;
                box.region[3] = box.region[3] * height_;

                // 对关键点做回归
                const float x1 = ctr_x + *(ldmk_datas[0]) * width * variances[0];
                const float x2 = ctr_x + *(ldmk_datas[2]) * width * variances[0];
                const float x3 = ctr_x + *(ldmk_datas[4]) * width * variances[0];
                const float x4 = ctr_x + *(ldmk_datas[6]) * width * variances[0];
                const float x5 = ctr_x + *(ldmk_datas[8]) * width * variances[0];

                const float y1 = ctr_y + *(ldmk_datas[1]) * height * variances[0];
                const float y2 = ctr_y + *(ldmk_datas[3]) * height * variances[0];
                const float y3 = ctr_y + *(ldmk_datas[5]) * height * variances[0];
                const float y4 = ctr_y + *(ldmk_datas[7]) * height * variances[0];
                const float y5 = ctr_y + *(ldmk_datas[9]) * height * variances[0];

                ldmk.region[0] = x1 * width_;
                ldmk.region[2] = x2 * width_;
                ldmk.region[4] = x3 * width_;
                ldmk.region[6] = x4 * width_;
                ldmk.region[8] = x5 * width_;
                ldmk.region[1] = y1 * height_;
                ldmk.region[3] = y2 * height_;
                ldmk.region[5] = y3 * height_;
                ldmk.region[7] = y4 * height_;
                ldmk.region[9] = y5 * height_;


//                cout<<"box region: "<<box.region[0] << ", " << box.region[1] << ", " << box.region[2] << ", " << box.region[3] << endl;
                // 如果是一个合法的框，那么就添加到临时结果数组里面
                if (box.region[2] > box.region[0] && box.region[3] > box.region[1])
                {
                    img_boxes.push_back(box);
                    img_landmarks.push_back(ldmk);
                }
            }

            // 将指针偏移到下一个 roi
            for (int i = 0; i < 4; i++) {
                roi_datas[i]++;
                bbox_datas[i]++;
            }
            for (int i = 1; i < class_num_; i++) {
                cls_datas[i]++;
            }

            for (int i = 0; i < 10; i++) {
                ldmk_datas[i]++;
            }
        }
    }
}

void SSDDetector::perform_nms() {
    // 对每一张图片做 NMS
    for (int img_idx = 0; img_idx < img_num_; img_idx++) {
        vector<int> &keep_inds = keep_inds_[img_idx];
        keep_inds.clear();
        vector<Box> &img_boxes = result_boxes_[img_idx];
        int box_num = img_boxes.size();
        if (box_num == 0) continue;
        // 用来排序的下标和 NMS 数组
        vector<int> box_inds(box_num);
        for (int i = 0; i < box_num; i++) {
            box_inds[i] = i;
        }

        // 排序
        sort(box_inds.begin(), box_inds.end(), by_box_score(img_boxes));

        // 只保留指定的最大数量的框
        if (box_num > max_roi_before_nms_) {
            box_num = max_roi_before_nms_;
        }

        // 计算每个 roi 的面积
        vector<float> areas(box_num);
        for (int i = 0; i < box_num; i++) {
            areas[i] = img_boxes[i].get_area();
        }

        // 做 NMS
        vector<bool> suppressed(box_num, false);
        for (int i = 0; i < box_num; i++) {
            const int src_box_idx = box_inds[i];
            if (suppressed[i]) continue;
            keep_inds.push_back(src_box_idx);
            Box &src_box = img_boxes[src_box_idx];
            for (int j = i + 1; j < box_num; j++) {
                const int dst_box_idx = box_inds[j];
                if (suppressed[j]) continue;
                Box &dst_box = img_boxes[dst_box_idx];
                // 如果不同 label 则不需要抑制
                if (src_box.label != dst_box.label) continue;

                // 计算交集面积
                const float x1 = std::max(src_box.region[0], dst_box.region[0]);
                const float y1 = std::max(src_box.region[1], dst_box.region[1]);
                const float x2 = std::min(src_box.region[2], dst_box.region[2]);
                const float y2 = std::min(src_box.region[3], dst_box.region[3]);

                // 如果 iou > nms_thresh_，则进行抑制
                if (x2 > x1 && y2 > y1) {
                    const float inter_area = (x2 - x1 + 1) * (y2 - y1 + 1);
                    if (inter_area / (areas[src_box_idx] + areas[dst_box_idx] - inter_area) > nms_thresh_) {
                        suppressed[j] = true;
                    }
                }
            }
        }
    }
}

