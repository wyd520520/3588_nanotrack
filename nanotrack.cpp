#include <iostream>
#include <cstdlib>
#include <string>
#include "nanotrack.hpp"
#include <malloc.h>
#include <cmath>
//eigen
#include <Eigen/Dense>
#include <Eigen/Core>

void transpose_nchw_to_nhwc(float* input, float* output, int n, int c, int h, int w) {
    // 将输入数组转换为 Eigen 矩阵
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(input, n * c, h * w);

    // 转置矩阵
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> transposed = mat.transpose();

    // 将转置后的矩阵数据复制到输出数组
    std::copy(transposed.data(), transposed.data() + transposed.size(), output);
}
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    //return 1.0f / (1.0f + std::exp(-x));
    return 1.0f / (1.0f + fast_exp(-x));
}

static float sz_whFun(cv::Point2f wh)
{
    float pad = (wh.x + wh.y) * 0.5f;
    float sz2 = (wh.x + pad) * (wh.y + pad);
    return std::sqrt(sz2);
}

static std::vector<float> sz_change_fun(std::vector<float> w, std::vector<float> h,float sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    std::vector<float> pad(rows * cols, 0);
    std::vector<float> sz2;
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            pad[i*cols+j] = (w[i * cols + j] + h[i * cols + j]) * 0.5f;
        }
    }
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            float t = std::sqrt((w[i * rows + j] + pad[i*rows+j]) * (h[i * rows + j] + pad[i*rows+j])) / sz;

            sz2.push_back(std::max(t,(float)1.0/t) );
        }
    }
    return sz2;
}

static std::vector<float> ratio_change_fun(std::vector<float> w, std::vector<float> h, cv::Point2f target_sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    float ratio = target_sz.x / target_sz.y;
    std::vector<float> sz2; 
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float t = ratio / (w[i * cols + j] / h[i * cols + j]);
            sz2.push_back(std::max(t, (float)1.0 / t));
        }
    }

    return sz2; 
}


static float  deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }
//打印模型属性等
int check_model(rknn_context ctx)
{
    // Get Model Input Output Number
    rknn_input_output_num io_num;
    int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        //ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        //ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }
    return 0;
}

NanoTrack::NanoTrack()
{
    create_grids();
    create_window();

    int model_size_back_127_t, model_size_back_255_x, model_size_head;
    unsigned char *back_127_t = load_model("/home/orangepi/Desktop/NanoTrack-master/rknn/backbone_127_q.rknn", &model_size_back_127_t);
    unsigned char *back_255_x = load_model("/home/orangepi/Desktop/NanoTrack-master/rknn/backbone_255_q.rknn", &model_size_back_255_x);
    unsigned char *head = load_model("/home/orangepi/Desktop/NanoTrack-master/rknn/head.rknn", &model_size_head);
    int ret;
    ret = rknn_init(&app_ctx_backbone_t.rknn_ctx, back_127_t, model_size_back_127_t, 0, nullptr);
    if (ret < 0) {
        printf("rknn_init backbone_t error ret=%d\n", ret);
        exit(-1);
    }
    
    ret = rknn_init(&app_ctx_backbone_x.rknn_ctx, back_255_x, model_size_back_255_x, 0, nullptr);
    if (ret < 0) {
        printf("rknn_init backbone_x error ret=%d\n", ret);
        exit(-1);
    }

    ret = rknn_init(&app_ctx_head.rknn_ctx, head, model_size_head, 0, nullptr);
    if (ret < 0) {
        printf("rknn_init neck_head_x error ret=%d\n", ret);
        exit(-1);
    }
    printf("rknn_init success\n");

    ret = check_model(app_ctx_backbone_t.rknn_ctx);
    if (ret < 0) {
        printf("check_model backbone_t error ret=%d\n", ret);
        exit(-1);
    }

    ret = check_model(app_ctx_backbone_x.rknn_ctx);
    if (ret < 0) {
        printf("check_model backbone_x error ret=%d\n", ret);
        exit(-1);
    }

    ret = check_model(app_ctx_head.rknn_ctx);
    if (ret < 0) {
        printf("check_model head error ret=%d\n", ret);
        exit(-1);
    }

}

NanoTrack::~NanoTrack()
{
    rknn_outputs_release(app_ctx_backbone_t.rknn_ctx, 1, nullptr);
    rknn_outputs_release(app_ctx_backbone_x.rknn_ctx, 1, nullptr);
    rknn_outputs_release(app_ctx_head.rknn_ctx, 2, nullptr);
    rknn_destroy(app_ctx_backbone_t.rknn_ctx);
    rknn_destroy(app_ctx_backbone_x.rknn_ctx);
    rknn_destroy(app_ctx_head.rknn_ctx);
    free(zf_data_transpose);
    printf("rknn_destroy success\n");
}

void NanoTrack::init(cv::Mat img, cv::Rect bbox)
{
    //T output Toutput shape: (1, 96, 8, 8)
//     back_X_in shape: (1, 255, 255, 3)
// head_X_in shape: (1, 16, 16, 96)
// head_X_in type: float32
// head_T_in shape: (1, 8, 8, 96)
    state.im_h = img.rows;
    state.im_w = img.cols;
    state.channel_ave = cv::mean(img);
    state.target_pos = cv::Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
    state.target_sz = cv::Point2f(bbox.width, bbox.height);

    float wc_z = state.target_sz.x + cfg.context_amount * (state.target_sz.x + state.target_sz.y);
    float hc_z = state.target_sz.y + cfg.context_amount * (state.target_sz.x + state.target_sz.y);
    float s_z = round(sqrt(wc_z * hc_z));

    cv::Mat z_crop = get_subwindow_tracking(img, state.target_pos, cfg.exemplar_size, int(s_z), state.channel_ave);
    
    cv::Mat rgb;
    cv::cvtColor(z_crop, rgb, cv::COLOR_BGR2RGB);
    rknn_input rknn_img[1];
    memset(rknn_img, 0, sizeof(rknn_img));
    rknn_img[0].index = 0;
    rknn_img[0].type = RKNN_TENSOR_UINT8;
    rknn_img[0].size = rgb.cols * rgb.rows * rgb.channels();
    rknn_img[0].fmt = RKNN_TENSOR_NHWC;
    rknn_img[0].buf = rgb.data;
    rknn_inputs_set(app_ctx_backbone_t.rknn_ctx, 1, rknn_img);

    rknn_run(app_ctx_backbone_t.rknn_ctx, nullptr);

    memset(zf, 0, sizeof(zf));
    for (auto & i : zf) {
        i.want_float = 1;
        i.is_prealloc = 0;
    }
    rknn_outputs_get(app_ctx_backbone_t.rknn_ctx, 1, zf, nullptr);

    float* zf_data = (float*)zf[0].buf; 
    
    std::vector<float> hanning(cfg.score_size, 0);
    window.resize(cfg.score_size * cfg.score_size, 0);
    
    //用eigen实现转置  n c h w -> n h w c
    int n = 1;
    int c = 96;
    int h = 8;
    int w = 8;

    zf_data_transpose = (float*)malloc(n * h * w * c * sizeof(float));
    transpose_nchw_to_nhwc(zf_data, zf_data_transpose, n, c, h, w);

    for (int i = 0; i < cfg.score_size; i++) {
        float w = 0.5f - 0.5f * std::cos(2 * PI * float(i) / float(cfg.score_size - 1)); //
        hanning[i] = w;
    }
    for (int i = 0; i < cfg.score_size; i++) { 
        for (int j = 0; j < cfg.score_size; j++) {
            window[i * cfg.score_size + j] = hanning[i] * hanning[j];
        }
    }
}

void NanoTrack::update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Point2f &target_sz, float scale_z, float &cls_score_max)
{
    rknn_input rknn_img[1];
    memset(rknn_img, 0, sizeof(rknn_img));
    cv::cvtColor(x_crops, x_crops, cv::COLOR_BGR2RGB);
    rknn_img[0].index = 0;
    rknn_img[0].type = RKNN_TENSOR_UINT8;
    rknn_img[0].size = x_crops.cols * x_crops.rows * x_crops.channels();
    rknn_img[0].fmt = RKNN_TENSOR_NHWC;
    rknn_img[0].buf = x_crops.data;
    rknn_inputs_set(app_ctx_backbone_x.rknn_ctx, 1, rknn_img);

    rknn_run(app_ctx_backbone_x.rknn_ctx, nullptr);

    rknn_output xf[1];
    memset(xf, 0, sizeof(xf));
    for (auto & i : xf) {
        i.want_float = 1;
        i.is_prealloc = 0;
    }
    rknn_outputs_get(app_ctx_backbone_x.rknn_ctx, 1, xf, nullptr); //  96 16 16 
    //transpose
    float* xf_data = (float*)xf[0].buf;
    float* xf_data_transpose = (float*)malloc(96 * 16 * 16 * sizeof(float));
    transpose_nchw_to_nhwc(xf_data, xf_data_transpose, 1, 96, 16, 16);

    
    rknn_input zf_xf[2];
    memset(zf_xf, 0, sizeof(zf_xf));
    zf_xf[0].index = 0;
    zf_xf[0].type = RKNN_TENSOR_FLOAT32;
    zf_xf[0].size = zf[0].size;
    zf_xf[0].fmt = RKNN_TENSOR_NHWC;
    //transposed zf
    zf_xf[0].buf = this->zf_data_transpose;
    //zf_xf[0].buf = zf[0].buf;
    zf_xf[0].pass_through = 0;
    zf_xf[1].index = 1;
    zf_xf[1].type = RKNN_TENSOR_FLOAT32;
    zf_xf[1].size = xf[0].size;
    zf_xf[1].fmt = RKNN_TENSOR_NHWC;
    zf_xf[1].buf =  xf_data_transpose;
    //zf_xf[1].buf = xf[0].buf; 
    zf_xf[1].pass_through = 0;
    rknn_inputs_set(app_ctx_head.rknn_ctx, 2, zf_xf); 
    
    rknn_run(app_ctx_head.rknn_ctx, nullptr);
    rknn_output outputs[2];
    memset(outputs, 0, sizeof(outputs));
    for (auto & output : outputs) {
        output.want_float = 1;
        output.is_prealloc = 0;
    }
    rknn_outputs_get(app_ctx_head.rknn_ctx, 2, outputs, nullptr);

    // manually call sigmoid on the output
    std::vector<float> cls_score_sigmoid;
    //std::vector<float> cls_score_softmax;
    float* cls_score_data = (float*)outputs[0].buf;
    float* bbox_pred_data = (float*)outputs[1].buf;
    cls_score_sigmoid.clear();

    int cols = cfg.score_size;
    int rows = cfg.score_size;
    int offset = cols*rows;
    for (int i = 0; i < cols * rows; i++) {
        cls_score_sigmoid.push_back(sigmoid(cls_score_data[i+offset]));
        //printf("cls_score_sigmoid: %f\n", cls_score_sigmoid[i]);
    }

    std::vector<float> pred_x1(cols * rows, 0), pred_y1(cols * rows, 0), pred_x2(cols * rows, 0), pred_y2(cols * rows, 0);

    float* bbox_pred_data1 = (float*)outputs[1].buf;
    float* bbox_pred_data2 = (float*)outputs[1].buf + cols * rows;
    float* bbox_pred_data3 = (float*)outputs[1].buf + 2 * cols * rows;
    float* bbox_pred_data4 = (float*)outputs[1].buf + 3 * cols * rows;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            pred_x1[i * cols + j] = this->grid_to_search_x[i * cols + j] - bbox_pred_data1[i * cols + j];
            pred_y1[i * cols + j] = this->grid_to_search_y[i * cols + j] - bbox_pred_data2[i * cols + j];
            pred_x2[i * cols + j] = this->grid_to_search_x[i * cols + j] + bbox_pred_data3[i * cols + j];
            pred_y2[i * cols + j] = this->grid_to_search_y[i * cols + j] + bbox_pred_data4[i * cols + j];
        }
    }

    // size penalty
    std::vector<float> w(cols * rows, 0), h(cols * rows, 0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            w[i * cols + j] = pred_x2[i * cols + j] - pred_x1[i * cols + j];
            h[i * rows + j] = pred_y2[i * rows + j] - pred_y1[i * cols + j];
        }
    }

    float sz_wh = sz_whFun(target_sz);
    std::vector<float> s_c = sz_change_fun(w, h, sz_wh);
    std::vector<float> r_c = ratio_change_fun(w, h, target_sz);

    std::vector<float> penalty(rows * cols, 0);
    for (int i = 0; i < rows * cols; i++) {
        penalty[i] = std::exp(-1 * (s_c[i] * r_c[i] - 1) * cfg.penalty_k);
    }

    // window penalty
    std::vector<float> pscore(rows * cols, 0);
    int r_max = 0, c_max = 0;
    float maxScore = 0;

    for (int i = 0; i < rows * cols; i++) {  //15*15
        pscore[i] = (penalty[i] * cls_score_sigmoid[i]) * (1 - cfg.window_influence) + this->window[i] * cfg.window_influence;
        if (pscore[i] > maxScore) {
            maxScore = pscore[i];
            r_max = std::floor(i / rows); //行
            c_max = ((float)i / rows - r_max) * rows;
        }
    }
    //printf("r_max: %d, c_max: %d\n", r_max, c_max); 
    // to real size
    float pred_x1_real = pred_x1[r_max * cols + c_max];
    float pred_y1_real = pred_y1[r_max * cols + c_max];
    float pred_x2_real = pred_x2[r_max * cols + c_max];
    float pred_y2_real = pred_y2[r_max * cols + c_max];

    float pred_xs = (pred_x1_real + pred_x2_real) / 2;
    float pred_ys = (pred_y1_real + pred_y2_real) / 2;
    float pred_w = pred_x2_real - pred_x1_real;
    float pred_h = pred_y2_real - pred_y1_real;

    float diff_xs = pred_xs; //- cfg.instance_size / 2;
    float diff_ys = pred_ys; //- cfg.instance_size / 2;

    diff_xs /= scale_z;
    diff_ys /= scale_z;
    pred_w /= scale_z;
    pred_h /= scale_z;
    //printf("diff_xs: %f, diff_ys: %f\n", diff_xs, diff_ys);
    //target_sz.x = target_sz.x / scale_z;
    //target_sz.y = target_sz.y / scale_z;
    // size learning rate
    float lr = penalty[r_max * cols + c_max] * cls_score_sigmoid[r_max * cols + c_max] * cfg.lr;

    // size rate
    //auto res_xs = float(target_pos.x + diff_xs);
    //auto res_ys = float(target_pos.y + diff_ys);
    float res_w = pred_w * lr + (1 - lr) * target_sz.x;
    float res_h = pred_h * lr + (1 - lr) * target_sz.y;

    //target_pos.x = int(res_xs);
    //target_pos.y = int(res_ys);
    
    target_pos.x = int(target_pos.x + diff_xs);
    target_pos.y = int(target_pos.y + diff_ys);
    target_sz.x = target_sz.x * (1 - lr) + lr * res_w;
    target_sz.y = target_sz.y * (1 - lr) + lr * res_h;

    cls_score_max = cls_score_sigmoid[r_max * cols + c_max];
}

void NanoTrack::track(cv::Mat im)
{
    cv::Point target_pos = state.target_pos;
    cv::Point2f target_sz = state.target_sz;

    float hc_z = target_sz.y + cfg.context_amount * (target_sz.x + target_sz.y);
    float wc_z = target_sz.x + cfg.context_amount * (target_sz.x + target_sz.y);
    float s_z = sqrt(wc_z * hc_z);
    float scale_z = cfg.exemplar_size / s_z;

    float d_search = (cfg.instance_size - cfg.exemplar_size) / 2;
    float pad = d_search / scale_z;
    float s_x = s_z + 2 * pad;

    cv::Mat x_crop = get_subwindow_tracking(im, target_pos, cfg.instance_size, int(s_x), state.channel_ave);

    // update
    target_sz.x = target_sz.x * scale_z;
    target_sz.y = target_sz.y * scale_z;

    float cls_score_max;

    this->update(x_crop, target_pos, target_sz, scale_z, cls_score_max);

    target_pos.x = std::max(0, std::min(state.im_w, target_pos.x));
    target_pos.y = std::max(0, std::min(state.im_h, target_pos.y));
    target_sz.x = float(std::max(10, std::min(state.im_w, int(target_sz.x))));
    target_sz.y = float(std::max(10, std::min(state.im_h, int(target_sz.y))) );

    state.target_pos = target_pos;
    state.target_sz = target_sz;
    //printf("target_pos: (%d, %d)\n", target_pos.x, target_pos.y);
    //printf("target_sz: (%f, %f)\n", target_sz.x, target_sz.y);
}

void NanoTrack::create_window()
{
    int score_size = cfg.score_size;
    std::vector<float> hanning(score_size, 0);
    window.resize(score_size * score_size, 0);

    for (int i = 0; i < score_size; i++) {
        float w = 0.5f - 0.5f * std::cos(2 * PI * i / (score_size - 1));
        hanning[i] = w;
    }
    for (int i = 0; i < score_size; i++) {
        for (int j = 0; j < score_size; j++) {
            window[i * score_size + j] = hanning[i] * hanning[j];
        }
    }
}

void NanoTrack::create_grids()
{
    int sz = cfg.score_size;
    float ori = - (sz / 2) * cfg.total_stride;
    grid_to_search_x.resize(sz * sz, 0);
    grid_to_search_y.resize(sz * sz, 0);

    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            grid_to_search_x[i * sz + j] = j * cfg.total_stride + ori;
            grid_to_search_y[i * sz + j] = i * cfg.total_stride + ori;
        }
    }
}

cv::Mat NanoTrack::get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz, cv::Scalar channel_ave)
{
    float c = (float)(original_sz + 1) / 2;
    int context_xmin = std::round(pos.x - c);
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = std::round(pos.y - c);
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad = int(std::max(0, -context_xmin));
    int top_pad = int(std::max(0, -context_ymin));
    int right_pad = int(std::max(0, context_xmax - im.cols + 1));
    int bottom_pad = int(std::max(0, context_ymax - im.rows + 1));

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;
    cv::Mat im_path_original;

    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0) {
        cv::Mat te_im = cv::Mat::zeros(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, CV_8UC3);

        cv::copyMakeBorder(im, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, channel_ave);
        im_path_original = te_im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    } else {
        im_path_original = im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }

    cv::Mat im_path;
    cv::resize(im_path_original, im_path, cv::Size(model_sz, model_sz));

    return im_path;
}


unsigned char *NanoTrack::load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    auto *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    fclose(fp);
    return model;
}