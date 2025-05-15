// segmentImage.cpp

#include "utils/YOLOSegmenterClient.h"
#include <numeric>
#include <iomanip>

int main(int argc, char** argv) {
    // Check if image path is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [server_url] [iterations]" << std::endl;
        return 1;
    }
    
    // Check if debug mode is enabled
    bool debug_enabled = (std::getenv("DEBUG_SEGMENT") != nullptr);
    if (debug_enabled) {
        std::cout << "[DEBUG_SEGMENT] Segmentation application started" << std::endl;
        std::cout << "[DEBUG_SEGMENT] Command line arguments: " << argc << std::endl;
        for (int i = 0; i < argc; i++) {
            std::cout << "[DEBUG_SEGMENT] Arg[" << i << "]: " << argv[i] << std::endl;
        }
    }
    
    // Load image
    std::string image_path = argv[1];
    if (debug_enabled) {
        std::cout << "[DEBUG_SEGMENT] Loading image from: " << image_path << std::endl;
    }
    
    cv::Mat image = cv::imread(image_path);
    
    if (image.empty()) {
        std::cerr << "Error: Could not load image: " << image_path << std::endl;
        if (debug_enabled) {
            std::cout << "[DEBUG_SEGMENT] Failed to load image: " << image_path << std::endl;
        }
        return 1;
    }
    
    if (debug_enabled) {
        std::cout << "[DEBUG_SEGMENT] Image loaded successfully: " 
                  << image.cols << "x" << image.rows 
                  << ", channels: " << image.channels() << std::endl;
    }
    
    // Set server URL (default or from command line)
    std::string server_url = "http://192.248.10.70:8000/segment";
    if (argc > 2) {
        server_url = argv[2];
    }
    
    // Set number of iterations (default or from command line)
    int num_iterations = 100;
    if (argc > 3) {
        num_iterations = std::stoi(argv[3]);
    }
    
    if (debug_enabled) {
        std::cout << "[DEBUG_SEGMENT] Using server URL: " << server_url << std::endl;
        std::cout << "[DEBUG_SEGMENT] Running " << num_iterations << " iterations" << std::endl;
    }
    
    std::cout << "Starting performance test with " << num_iterations << " iterations..." << std::endl;
    std::cout << "Image size: " << image.cols << "x" << image.rows << ", URL: " << server_url << std::endl;
    
    // Create segmenter client
    YOLOSegmenterClient client(server_url);
    
    // Vectors to store timing results for each stage
    std::vector<double> request_times(num_iterations);
    std::vector<double> total_times(num_iterations);
    
    // Store the last mask for visualization
    cv::Mat last_mask;
    
    // Run the benchmark
    for (int i = 0; i < num_iterations; ++i) {
        std::cout << "Iteration " << (i+1) << "/" << num_iterations << "..." << std::flush;
        
        // Start time measurement
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Send request asynchronously
        auto request_start = std::chrono::high_resolution_clock::now();
        auto future_mask = client.fetchMasksAsync(image);
        auto request_end = std::chrono::high_resolution_clock::now();
        
        // Record request sending time
        request_times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(
            request_end - request_start).count();
        
        // Get the mask
        last_mask = future_mask.get();
        
        // End time measurement
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Record total time
        total_times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        std::cout << " done in " << total_times[i] << "ms" << std::endl;
    }
    
    // Calculate statistics
    // Sort the times to calculate percentiles
    std::vector<double> sorted_request_times = request_times;
    std::vector<double> sorted_total_times = total_times;
    std::sort(sorted_request_times.begin(), sorted_request_times.end());
    std::sort(sorted_total_times.begin(), sorted_total_times.end());
    
    // Calculate mean
    double mean_request_time = std::accumulate(request_times.begin(), request_times.end(), 0.0) / num_iterations;
    double mean_total_time = std::accumulate(total_times.begin(), total_times.end(), 0.0) / num_iterations;
    
    // Calculate median (50th percentile)
    double median_request_time = sorted_request_times[num_iterations / 2];
    double median_total_time = sorted_total_times[num_iterations / 2];
    
    // Calculate 95th percentile
    double p95_request_time = sorted_request_times[static_cast<int>(num_iterations * 0.95)];
    double p95_total_time = sorted_total_times[static_cast<int>(num_iterations * 0.95)];
    
    // Calculate min and max
    double min_request_time = sorted_request_times.front();
    double max_request_time = sorted_request_times.back();
    double min_total_time = sorted_total_times.front();
    double max_total_time = sorted_total_times.back();
    
    // Print statistics
    std::cout << "\n=== Performance Results (" << num_iterations << " iterations) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Request Times (ms):  Mean: " << mean_request_time
              << ", Median: " << median_request_time
              << ", 95th: " << p95_request_time
              << ", Min: " << min_request_time
              << ", Max: " << max_request_time << std::endl;
    
    std::cout << "Total Times (ms):    Mean: " << mean_total_time
              << ", Median: " << median_total_time 
              << ", 95th: " << p95_total_time
              << ", Min: " << min_total_time
              << ", Max: " << max_total_time << std::endl;
    
    // Check if we got a valid mask
    int nonzero_pixels = cv::countNonZero(last_mask);
    double mask_coverage = (nonzero_pixels * 100.0) / (last_mask.rows * last_mask.cols);
    std::cout << "Last mask contains " << nonzero_pixels << " non-zero pixels (" 
              << mask_coverage << "% coverage)" << std::endl;
    
    // Visualize the results if any objects were detected
    if (nonzero_pixels > 0) {
        std::cout << "Visualizing results..." << std::endl;
        
        // Save mask
        std::string mask_path = "segmentation_mask.png";
        cv::imwrite(mask_path, last_mask);
        std::cout << "Saved mask to " << mask_path << std::endl;
        
        // Get mask coordinates
        std::vector<cv::Point> coordinates = getMaskCoordinates(last_mask);
        std::cout << "Found " << coordinates.size() << " mask points" << std::endl;
        
        // Create visualization
        cv::Mat visualization = visualizeMask(image, last_mask);
        cv::imwrite("segmentation_visualization.png", visualization);
        std::cout << "Saved visualization to segmentation_visualization.png" << std::endl;
        
        // Display results
        cv::imshow("Original Image", image);
        cv::imshow("Segmentation Mask", last_mask);
        cv::imshow("Visualization", visualization);
        
        // Wait for key press to close windows
        std::cout << "Press any key to close visualization windows..." << std::endl;
        cv::waitKey(0);
    } else {
        std::cout << "No objects detected in the image." << std::endl;
        
        // Display original image only
        cv::imshow("Original Image (No Objects Detected)", image);
        cv::waitKey(0);
    }
    
    return 0;
}