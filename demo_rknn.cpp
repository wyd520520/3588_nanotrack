#include <iostream>
#include <cstdlib>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  

#include "nanotrack.hpp" 

void cxy_wh_2_rect(const cv::Point& pos, const cv::Point2f& sz, cv::Rect &rect) 
{   
    rect.x = std::max(0, pos.x - int(sz.x / 2));
    rect.y = std::max(0, pos.y - int(sz.y / 2));
    rect.width = int(sz.x);   
    rect.height = int(sz.y);    
}

void track(NanoTrack *siam_tracker, const char *video_path)

{
    // Read video 
    cv::VideoCapture capture; 
    bool ret;
    if (strlen(video_path)==1)
        ret = capture.open(atoi(video_path));  
    else
        ret = capture.open(video_path); 

    // Exit if video not opened.
    if (!ret) 
        std::cout << "Open cap failed!" << std::endl;

    // Read first frame. 
    cv::Mat frame; 
    
    bool ok = capture.read(frame);
    if (!ok)
    {
        std::cout<< "Cannot read video file" << std::endl;
        return; 
    }
    
    // Select a rect.
    cv::namedWindow("demo"); 
    //cv::Rect trackWindow = cv::selectROI("demo", frame); // 手动选择
    cv::Rect trackWindow =cv::Rect(244,161,74,70);         // 固定值 
    
    // Initialize tracker with first frame and rect.
    State state; 
    
    //writer
    cv::VideoWriter writer;
    //writer.open("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(640, 480), true);
    //cv::Mat framee = cv::Mat::zeros(640, 640, CV_8UC3);
    //   frame, (cx,cy) , (w,h), state
    siam_tracker->init(frame, trackWindow);
    std::cout << "==========================" << std::endl;
    std::cout << "Init done!" << std::endl; 
    std::cout << std::endl; 
    cv::Mat init_window;
    frame(trackWindow).copyTo(init_window); 
    //计算fps
    int frame_count = 0;
    double now = cv::getTickCount();
    for (;;)
    {
        // Read a new frame.
        bool ok = capture.read(frame);
        if (!ok)
        {
            std::cout << "Cannot read video file" << std::endl;
            break; 
        }
        float cls_score_max;
        //siam_tracker->update(frame, target_pos, target_sz, 1.0, cls_score_max);
        siam_tracker->track(frame);
        // Draw the tracked object.
        cv::Rect result_rect;
        cxy_wh_2_rect(siam_tracker->state.target_pos, siam_tracker->state.target_sz, result_rect);
        //cv::rectangle(frame, result_rect, cv::Scalar(0, 255, 0), 2, 1);
        //cv::imshow("demo", frame);
        //cv::resize(frame, frame, cv::Size(640, 480));
        //writer.write(frame);
        
        // Exit if ESC pressed.
        // int k = cv::waitKey(1);
        // if (k == 27)
        //     break;
        //计算fps
        frame_count++;
        if (frame_count % 100 == 0)
        {
            double time = ((double)cv::getTickCount() - now) / cv::getTickFrequency();
            double fps = 100 / time;
            std::cout << "fps: " << fps << std::endl;
            now = cv::getTickCount();
        }
    }
}

int main(int argc, char **argv)
{
    // if (argc != 2)
    // {
    //     std::cout << "Usage: " << argv[0] << " video_path" << std::endl;
    //     return -1;
    // }
    // string video_path = "girl_dance.mp4"

    std::string video_path;
    video_path = "/home/orangepi/Desktop/nanotrack_cpp/girl_dance.mp4";
    NanoTrack *siam_tracker = new NanoTrack();
    track(siam_tracker, video_path.c_str());


    delete siam_tracker;
    return 0;
}
