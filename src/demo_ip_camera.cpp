#include <glog/logging.h>
#include <iostream>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "SuperPointFast.h"

using namespace std;
using namespace cv;

// Atomic variables for FPS calculation
std::atomic<float> current_fps(0.0f);
std::atomic<int> frame_count(0);
std::atomic<int> keypoints_count(0);

// Trackbar variables (scaled for better usability)
int conf_thresh_trackbar = 10; // Range 0-100, maps to 0.0-0.03
int dist_thresh_trackbar = 2;  // Range 0-10

// Callback functions for trackbars
void on_conf_thresh_change(int pos, void*) {
    float new_val = pos * 0.0003f; // Scale from 0-100 to 0.0-0.03
    g_conf_thresh.store(new_val);
}

void on_dist_thresh_change(int pos, void*) {
    g_dist_thresh.store(pos);
}

void print_usage(char* prog_name) {
  std::cout << "Usage: " << prog_name << " [options] model_name ip_address port fps protocol" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -t <num>      Number of preprocessing/postprocessing threads to use (default: 2)" << std::endl;
  std::cout << "                Note: DPU runners fixed at 4" << std::endl;
  std::cout << "Example: " << prog_name << " -t 4 superpoint_tf.xmodel 192.168.1.100 8554 30 rtsp" << std::endl;
}

int main(int argc, char* argv[]) {
  // Default parameters
  int num_threads = 2;
  std::string model_name;
  std::string ip_address;
  int port = 0;
  int target_fps = 30;
  std::string protocol = "rtsp";  // Default protocol
  
  // Parse command line arguments
  int arg_index = 1;
  while (arg_index < argc) {
    std::string arg = argv[arg_index];
    if (arg == "-t") {
      if (arg_index + 1 < argc) {
        num_threads = std::stoi(argv[arg_index + 1]);
        arg_index += 2;
      } else {
        std::cerr << "Error: Missing value for -t option" << std::endl;
        print_usage(argv[0]);
        return 1;
      }
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else {
      break;
    }
  }
  
  // Get positional arguments
  if (arg_index + 4 < argc) {
    model_name = argv[arg_index];
    ip_address = argv[arg_index + 1];
    port = std::stoi(argv[arg_index + 2]);
    target_fps = std::stoi(argv[arg_index + 3]);
    protocol = argv[arg_index + 4];
  } else {
    std::cerr << "Error: Missing required positional arguments" << std::endl;
    print_usage(argv[0]);
    return 1;
  }
  
  std::cout << "Configuration:" << std::endl;
  std::cout << "- Number of pre/post-processing threads: " << num_threads << std::endl;
  std::cout << "- Number of DPU runners: 4 (fixed)" << std::endl;
  std::cout << "- Model: " << model_name << std::endl;
  std::cout << "- IP Address: " << ip_address << std::endl;
  std::cout << "- Port: " << port << std::endl;
  std::cout << "- Target FPS: " << target_fps << std::endl;
  std::cout << "- Protocol: " << protocol << std::endl;

  try {
    // Initialize SuperPointFast
    auto superpoint = SuperPointFast(model_name, num_threads);
    
    // Create thread-safe queues
    ThreadSafeQueue<InputQueueItem> input_queue(10);  // Smaller queue size to drop frames if processing is too slow
    ThreadSafeQueue<ResultQueueItem> output_queue(10);
    
    // Start the SuperPointFast processor 
    superpoint.run(input_queue, output_queue);
    
    // URL for video stream based on protocol
    std::string stream_url = protocol + "://" + ip_address + ":" + std::to_string(port);

    if (protocol == "rtsp") {
      stream_url += "/h264.sdp";
    } else if (protocol == "http") {
      stream_url += "/video";
    } 
    std::cout << "Connecting to stream: " << stream_url << std::endl;
    
    // Create a window for displaying results and controls
    cv::namedWindow("SuperPoint IP Camera Feed", cv::WINDOW_NORMAL);
    
    // Initialize trackbar values based on current atomic values
    conf_thresh_trackbar = g_conf_thresh.load() * 1000;  // Scale for UI (0.010 -> 10)
    dist_thresh_trackbar = g_dist_thresh.load();
    
    // Create trackbars for parameter adjustments
    cv::createTrackbar("Conf Thresh (x1000)", "SuperPoint IP Camera Feed", 
                      &conf_thresh_trackbar, 100, on_conf_thresh_change);
    cv::createTrackbar("NMS Dist Thresh", "SuperPoint IP Camera Feed", 
                      &dist_thresh_trackbar, 10, on_dist_thresh_change);
    
    // Start producer thread to capture frames from IP camera
    std::thread producer_thread([stream_url, &input_queue, target_fps, protocol]() {
      // Open video stream
      cv::VideoCapture cap(stream_url);
      if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video stream at " << stream_url << std::endl;
        input_queue.shutdown();
        return;
      }
      
      size_t frame_idx = 0;
      size_t frames_dropped = 0;
      auto last_frame_time = std::chrono::high_resolution_clock::now();
      
      while (true) {
        // Calculate time since last frame
        auto now = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_frame_time);
        
        // Enforce target FPS by waiting if needed
        int frame_time_ms = 1000 / target_fps;
        if (frame_duration.count() < frame_time_ms) {
          std::this_thread::sleep_for(std::chrono::milliseconds(frame_time_ms - frame_duration.count()));
          now = std::chrono::high_resolution_clock::now();
        }
        
        last_frame_time = now;
        
        // Capture frame
        cv::Mat frame;
        cap >> frame;
        
        if (frame.empty()) {
            //try once more
            std::this_thread::sleep_for(std::chrono::seconds(frame_time_ms / 4));
            cap >> frame;
            if (frame.empty()) {
              std::cerr << "End of stream or error in video capture" << std::endl;
              break;
            }
        }
        
        // Create input queue item
        InputQueueItem item;
        item.index = frame_idx++;
        
        // Set timestamp in seconds since epoch for compatibility
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count() / 1000.0;
        item.timestamp = timestamp;
        item.image = frame;
        item.filename = "frame_" + std::to_string(item.index);
        
        // Try to add to queue, but don't block if queue is full (drop frames instead)
        if (!input_queue.try_enqueue_for(item, std::chrono::milliseconds(1))) {
          // Frame dropped - queue is full
          // std::cout << "Frame dropped: " << item.index << std::endl;
          frames_dropped++;
          frame_idx--;
        }
      }
      
      // Signal that no more frames will be added
      input_queue.shutdown();
      std::cout << "Producer thread finished, processed " << frame_idx << " frames, dropped " << frames_dropped << " frames" << std::endl;
    });
    
    // Start consumer thread to process results and display
    std::thread consumer_thread([&output_queue]() {
      int fps_update_interval = 10;  // Update FPS every 10 frames
      auto start_time = std::chrono::high_resolution_clock::now();
      int local_frame_count = 0;
      
      ResultQueueItem result;
      while (output_queue.dequeue(result)) {
        // Count keypoints
        keypoints_count.store(result.keypoints.size());
        
        // Use the original image stored in the result
        cv::Mat result_img = result.image.clone();
        
        // Draw keypoints directly on the image
        for (const auto& kp : result.keypoints) {
          // Draw keypoint
          cv::circle(result_img, 
                    cv::Point(kp.pt.x, kp.pt.y), 
                    3, cv::Scalar(0, 255, 0), -1);  // green dots for keypoints
          
        }
        
        // Update FPS calculation
        local_frame_count++;
        if (local_frame_count % fps_update_interval == 0) {
          auto end_time = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
          float fps = (float)(fps_update_interval * 1000) / duration;
          current_fps.store(fps);
          
          // Reset timer for next interval
          start_time = end_time;
        }
        
        // Update frame counter for display
        frame_count.store(result.index);
        
        // Add text overlay with information
        std::string info_text = "Frame: " + std::to_string(result.index) + 
                               " | FPS: " + std::to_string(current_fps.load()) +
                               " | Keypoints: " + std::to_string(result.keypoints.size());
        cv::putText(result_img, info_text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // Add parameter values to display
        std::string param_text = "Conf Thresh: " + std::to_string(g_conf_thresh.load()) +
                               " | NMS Dist: " + std::to_string(g_dist_thresh.load());
        cv::putText(result_img, param_text, cv::Point(10, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
        
        // Display image
        cv::imshow("SuperPoint IP Camera Feed", result_img);
        
        // Exit if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
          std::cout << "User exit requested" << std::endl;
          break;
        }
      }
      
      std::cout << "Consumer thread finished" << std::endl;
      cv::destroyAllWindows();
    });
    
    // Wait for producer thread to complete
    producer_thread.join();
    
    // Wait for consumer thread to complete
    consumer_thread.join();
    
    std::cout << "Processing completed successfully!" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
} 