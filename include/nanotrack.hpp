#ifndef NANOTRACK_H
#define NANOTRACK_H

#include <vector>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "rknn_api.h"
#include "common.h"

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

#define PI 3.1415926

using namespace cv;

struct Config {
    std::string windowing = "cosine";
    std::vector<float> window;

    int stride = 16;
    float penalty_k = 0.138; //penalty是对于尺度的惩罚
    float window_influence = 0.455; //window_influence是对于位置的惩罚
    float lr = 0.348;
    int exemplar_size = 127;
    int instance_size = 255;
    int total_stride = 16;
    int score_size = 15;
    float context_amount = 0.5;
};

struct State {
    int im_h;
    int im_w;
    cv::Scalar channel_ave;
    cv::Point target_pos;
    cv::Point2f target_sz = {0.f, 0.f};
    float cls_score_max;
};

class NanoTrack {

public:
    NanoTrack();

    ~NanoTrack();

    void init(cv::Mat img, cv::Rect bbox);

    void update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Point2f &target_sz, float scale_z, float &cls_score_max);

    void track(cv::Mat im);

    unsigned char *load_model(const char *filename, int *model_size);

    rknn_app_context_t app_ctx_backbone_t, app_ctx_backbone_x,app_ctx_head;

    int stride = 16;

    State state;

    Config cfg;

private:
    void create_grids();

    void create_window();

    cv::Mat get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz, cv::Scalar channel_ave);

    std::vector<float> grid_to_search_x;
    std::vector<float> grid_to_search_y;
    std::vector<float> window;
    float *zf_data_transpose;
    rknn_output zf[1];
    
};

#endif

