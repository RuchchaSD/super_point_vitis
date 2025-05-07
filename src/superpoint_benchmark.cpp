/**
 * SuperPoint Benchmark - A lightweight benchmarking tool for SuperPoint on Kria KR260
 * 
 * This benchmark measures:
 * 1. Keypoint detection performance
 * 2. Descriptor computation time
 * 3. Overall processing throughput
 * 4. Matching capability using the HPatches dataset (if specified)
 */

 #include <glog/logging.h>
 #include <iostream>
 #include <string>
 #include <vector>
 #include <chrono>
 #include <filesystem>
 #include <iomanip>
 #include <fstream>
 #include <opencv2/core.hpp>
 #include <opencv2/highgui.hpp>
 #include <opencv2/imgproc.hpp>
 #include <opencv2/calib3d.hpp>
 #include <opencv2/features2d.hpp>
 #include "SuperPointFast.h"
 
 namespace fs = std::filesystem;
 using namespace std;
 using namespace cv;
 
 // Timer utility class
 class Timer {
 public:
     Timer() : m_start(std::chrono::high_resolution_clock::now()) {}
     
     void reset() {
         m_start = std::chrono::high_resolution_clock::now();
     }
     
     double elapsed_ms() {
         auto end = std::chrono::high_resolution_clock::now();
         return std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start).count();
     }
     
     double elapsed_us() {
         auto end = std::chrono::high_resolution_clock::now();
         return std::chrono::duration_cast<std::chrono::microseconds>(end - m_start).count();
     }
     
 private:
     std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
 };
 
 // Simple statistics class
// Statistics class with proper min/max initialization
class Statistics {
    public:
        void add(double value) {
            m_values.push_back(value);
            m_sum += value;
            m_min = std::min(m_min, value);
            m_max = std::max(m_max, value);
        }
        
        double mean() const {
            return m_values.empty() ? 0.0 : m_sum / m_values.size();
        }
        
        double median() const {
            if (m_values.empty()) return 0.0;
            std::vector<double> sorted = m_values;
            std::sort(sorted.begin(), sorted.end());
            size_t mid = sorted.size() / 2;
            if (sorted.size() % 2 == 0) {
                return (sorted[mid-1] + sorted[mid]) / 2.0;
            } else {
                return sorted[mid];
            }
        }
        
        double min() const { return m_min; }
        double max() const { return m_max; }
        size_t count() const { return m_values.size(); }
        
    private:
        std::vector<double> m_values;
        double m_sum = 0.0;
        double m_min = std::numeric_limits<double>::max();
        double m_max = std::numeric_limits<double>::lowest();
    };
 
 // In the repeatability calculation, add boundary checks:
double compute_repeatability(
    const std::vector<cv::KeyPoint>& kpts1,
    const std::vector<cv::KeyPoint>& kpts2,
    const cv::Mat& H,
    const cv::Size& img_size,
    double threshold = 3.0) {
    
    if (kpts1.empty() || kpts2.empty() || H.empty()) {
        return 0.0;
    }
    
    // Convert keypoints to points
    std::vector<cv::Point2f> points1, points2;
    cv::KeyPoint::convert(kpts1, points1);
    cv::KeyPoint::convert(kpts2, points2);
    
    // Warp points from image 1 to image 2
    std::vector<cv::Point2f> points1_warped;
    cv::perspectiveTransform(points1, points1_warped, H);
    
    // Build kd-tree for fast nearest neighbor search
    cv::flann::KDTreeIndexParams indexParams;
    cv::flann::Index kdtree(cv::Mat(points2).reshape(1), indexParams);
    
    // Count matches within threshold
    int matches = 0;
    int valid_points = 0;
    
    for (size_t i = 0; i < points1_warped.size(); i++) {
        // Check if the warped point is inside the image boundaries
        if (points1_warped[i].x >= 0 && points1_warped[i].x < img_size.width &&
            points1_warped[i].y >= 0 && points1_warped[i].y < img_size.height) {
            
            valid_points++;
            
            // Find the nearest keypoint in the second image using kd-tree
            std::vector<float> query = {points1_warped[i].x, points1_warped[i].y};
            std::vector<int> indices(1);
            std::vector<float> dists(1);
            kdtree.knnSearch(query, indices, dists, 1);
            
            if (dists[0] < threshold * threshold) {
                matches++;
            }
        }
    }
    
    // Compute repeatability
    return valid_points > 0 ? static_cast<double>(matches) / valid_points : 0.0;
}
 
 // Extract matches between two sets of descriptors
 std::vector<cv::DMatch> match_descriptors(
     const cv::Mat& desc1,
     const cv::Mat& desc2,
     float ratio_threshold = 0.8f) {
     
     std::vector<std::vector<cv::DMatch>> knn_matches;
     cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
     
     if (desc1.empty() || desc2.empty()) {
         return std::vector<cv::DMatch>();
     }
     
     matcher->knnMatch(desc1, desc2, knn_matches, 2);
     
     // Apply ratio test
     std::vector<cv::DMatch> good_matches;
     for (size_t i = 0; i < knn_matches.size(); i++) {
         if (knn_matches[i].size() >= 2 && 
             knn_matches[i][0].distance < ratio_threshold * knn_matches[i][1].distance) {
             good_matches.push_back(knn_matches[i][0]);
         }
     }
     
     return good_matches;
 }
 
 // Compute matching score
// In the matching score calculation, improve the matching:
double compute_matching_score(
    const std::vector<cv::KeyPoint>& kpts1,
    const std::vector<cv::KeyPoint>& kpts2,
    const cv::Mat& desc1,
    const cv::Mat& desc2,
    const cv::Mat& H,
    double threshold = 3.0) {
    
    if (kpts1.empty() || kpts2.empty() || desc1.empty() || desc2.empty() || H.empty()) {
        return 0.0;
    }
    
    // Ensure descriptors are in the proper format
    cv::Mat desc1_float, desc2_float;
    if (desc1.type() != CV_32F) {
        desc1.convertTo(desc1_float, CV_32F);
    } else {
        desc1_float = desc1;
    }
    
    if (desc2.type() != CV_32F) {
        desc2.convertTo(desc2_float, CV_32F);
    } else {
        desc2_float = desc2;
    }
    
    // Match descriptors using FLANN
    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();
    std::vector<std::vector<cv::DMatch>> knn_matches;
    
    try {
        matcher->knnMatch(desc1_float, desc2_float, knn_matches, 2);
        
        // Apply ratio test
        const float ratio_thresh = 0.7f;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() >= 2 && 
                knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                matches.push_back(knn_matches[i][0]);
            }
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Matching failed: " << e.what() << std::endl;
        return 0.0;
    }
    
    if (matches.empty()) {
        return 0.0;
    }
    
    // Extract matched points
    std::vector<cv::Point2f> matched1, matched2;
    for (const auto& m : matches) {
        matched1.push_back(kpts1[m.queryIdx].pt);
        matched2.push_back(kpts2[m.trainIdx].pt);
    }
    
    // Warp points from image 1 to image 2
    std::vector<cv::Point2f> matched1_warped;
    cv::perspectiveTransform(matched1, matched1_warped, H);
    
    // Count correct matches
    int correct_matches = 0;
    for (size_t i = 0; i < matched1_warped.size(); i++) {
        float dist = cv::norm(matched1_warped[i] - matched2[i]);
        if (dist < threshold) {
            correct_matches++;
        }
    }
    
    // Compute matching score
    return static_cast<double>(correct_matches) / matches.size();
}
 
 void print_usage(char* prog_name) {
     std::cout << "Usage: " << prog_name << " [options] model_name input_folder" << std::endl;
     std::cout << "Options:" << std::endl;
     std::cout << "  -t <num>      Number of preprocessing/postprocessing threads to use (default: 2)" << std::endl;
     std::cout << "  -f <ext>      File extension filter (default: ppm)" << std::endl;
     std::cout << "  -o <dir>      Output directory for visualization (optional)" << std::endl;
     std::cout << "  -h            Homography test mode using HPatches" << std::endl;
     std::cout << "Example: " << prog_name << " -t 4 -h superpoint_tf.xmodel ./HPatches" << std::endl;
 }
 
 int main(int argc, char* argv[]) {
     // Default parameters
     int num_threads = 2;
     std::string file_ext = "ppm";
     std::string model_name;
     std::string input_folder;
     std::string output_folder = "";
     bool homography_test = false;
     
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
         } else if (arg == "-f") {
             if (arg_index + 1 < argc) {
                 file_ext = argv[arg_index + 1];
                 arg_index += 2;
             } else {
                 std::cerr << "Error: Missing value for -f option" << std::endl;
                 print_usage(argv[0]);
                 return 1;
             }
         } else if (arg == "-o") {
             if (arg_index + 1 < argc) {
                 output_folder = argv[arg_index + 1];
                 arg_index += 2;
             } else {
                 std::cerr << "Error: Missing value for -o option" << std::endl;
                 print_usage(argv[0]);
                 return 1;
             }
         } else if (arg == "-h") {
             homography_test = true;
             arg_index += 1;
         } else if (arg == "--help") {
             print_usage(argv[0]);
             return 0;
         } else {
             break;
         }
     }
     
     // Get positional arguments
     if (arg_index + 1 < argc) {
         model_name = argv[arg_index];
         input_folder = argv[arg_index + 1];
     } else {
         std::cerr << "Error: Missing required positional arguments" << std::endl;
         print_usage(argv[0]);
         return 1;
     }
     
     // Create output directory if specified
     if (!output_folder.empty()) {
         try {
             if (!fs::exists(output_folder)) {
                 fs::create_directories(output_folder);
             }
         } catch (const std::exception& e) {
             std::cerr << "Error creating output directory: " << e.what() << std::endl;
             return 1;
         }
     }
     
     // Print configuration
     std::cout << "- Model: " << model_name << std::endl;
    
     if (!output_folder.empty()) {
         std::cout << "- Output folder: " << output_folder << std::endl;
     }
     std::cout << "- File extension: " << file_ext << std::endl;
     std::cout << "- Homography test: " << (homography_test ? "Enabled" : "Disabled") << std::endl;
     
     try {
         // Initialize SuperPointFast
         auto superpoint = SuperPointFast(model_name, num_threads);
         
         // Statistics collectors
         Statistics time_stats, keypoint_stats, throughput_stats;
         Statistics repeatability_stats, matching_score_stats;
         
         if (homography_test) {
             // Process HPatches dataset
             std::cout << "\nProcessing HPatches dataset for homography evaluation..." << std::endl;
             
             for (const auto& scene_dir : fs::directory_iterator(input_folder)) {
                 if (!fs::is_directory(scene_dir)) continue;
                 
                 std::string scene_name = scene_dir.path().filename().string();
                 std::cout << "\nProcessing scene: " << scene_name << std::endl;
                 
                 // Load all images in the scene
                 std::vector<cv::Mat> images;
                 std::vector<std::string> image_names;
                 
                 for (const auto& entry : fs::directory_iterator(scene_dir.path())) {
                     if (entry.is_regular_file() && 
                         entry.path().extension() == "." + file_ext) {
                         
                         std::string filename = entry.path().filename().string();
                         // Check if the file is a numbered image (like 1.ppm, 2.ppm, etc.)
                         if (isdigit(filename[0])) {
                             cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
                             if (!img.empty()) {
                                 images.push_back(img);
                                 image_names.push_back(filename);
                             }
                         }
                     }
                 }
                 
                 if (images.size() < 2) {
                     std::cout << "Skip scene " << scene_name << ": not enough images found" << std::endl;
                     continue;
                 }
                 
                 std::cout << "Found " << images.size() << " images in scene" << std::endl;
                 
                 // Load homography matrices for reference image (1.ppm)
                // Fix the homography file loading section:
                // Load homography matrices for reference image (1.ppm)
                std::vector<cv::Mat> homographies;
                for (size_t i = 1; i < images.size(); i++) {
                    // Extract just the number from image name (e.g., "2" from "2.ppm")
                    std::string img_num = image_names[i].substr(0, image_names[i].find('.'));
                    std::string h_filename = "H_1_" + img_num;
                    fs::path h_path = scene_dir.path() / h_filename;
                    
                    if (fs::exists(h_path)) {
                        cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
                        std::ifstream h_file(h_path);
                        if (h_file.is_open()) {
                            for (int r = 0; r < 3; r++) {
                                for (int c = 0; c < 3; c++) {
                                    h_file >> H.at<double>(r, c);
                                }
                            }
                            homographies.push_back(H);
                            std::cout << "Loaded homography H_1_" << img_num << std::endl;
                        } else {
                            std::cerr << "Warning: Could not open homography file " << h_path << std::endl;
                            // Push identity matrix as fallback
                            homographies.push_back(cv::Mat::eye(3, 3, CV_64F));
                        }
                    } else {
                        std::cerr << "Warning: Homography file not found " << h_path << std::endl;
                        // Push identity matrix as fallback
                        homographies.push_back(cv::Mat::eye(3, 3, CV_64F));
                    }
                }

                    
                 
                 if (homographies.size() != images.size() - 1) {
                     std::cout << "Warning: Number of homographies (" << homographies.size() 
                               << ") doesn't match number of images - 1 (" << images.size() - 1 << ")" << std::endl;
                 }
                 
                 // Extract features from all images
                 std::vector<SuperPointResult> all_results;
                 
                 Timer batch_timer;
                 all_results = superpoint.run(images);
                 double batch_time = batch_timer.elapsed_ms();
                 
                 std::cout << "Batch processing time: " << batch_time << " ms" << std::endl;
                 std::cout << "Per image time: " << batch_time / images.size() << " ms" << std::endl;
                 
                 // Evaluate repeatability and matching
                 if (!homographies.empty()) {
                     cv::Mat& ref_img = images[0];
                     SuperPointResult& ref_result = all_results[0];
                     //hi
                     // Evaluate repeatability and matching
                    for (size_t i = 1; i < images.size() && (i-1) < homographies.size(); i++) {
                        cv::Mat& ref_img = images[0];
                        SuperPointResult& ref_result = all_results[0];
                        
                        if (!homographies[i-1].empty() && 
                            ref_result.keypoints_cv.size() > 0 && 
                            all_results[i].keypoints_cv.size() > 0) {
                            
                            double repeatability = compute_repeatability(
                                ref_result.keypoints_cv,
                                all_results[i].keypoints_cv,
                                homographies[i-1],
                                ref_img.size()
                            );
                            
                            // Only calculate matching score if we have descriptors
                            double matching_score = 0.0;
                            if (!ref_result.descriptors_cv.empty() && !all_results[i].descriptors_cv.empty()) {
                                matching_score = compute_matching_score(
                                    ref_result.keypoints_cv,
                                    all_results[i].keypoints_cv,
                                    ref_result.descriptors_cv,
                                    all_results[i].descriptors_cv,
                                    homographies[i-1]
                                );
                            }
                            
                            std::cout << "Image 1 -> " << i+1 << ": "
                                    << "Keypoints: " << ref_result.keypoints_cv.size() << "/" << all_results[i].keypoints_cv.size()
                                    << ", Repeatability = " << repeatability 
                                    << ", Matching Score = " << matching_score 
                                    << std::endl;
                                    
                            repeatability_stats.add(repeatability);
                            matching_score_stats.add(matching_score);
                         

                            // In the homography test section, add this output:
                            for (size_t i = 0; i < all_results.size(); i++) {
                                std::cout << "Image " << i+1 << ": " << all_results[i].keypoints_cv.size() 
                                        << " keypoints extracted" << std::endl;
                                // Track statistics
                                keypoint_stats.add(all_results[i].keypoints_cv.size());
                            }

                            // Then add this to the final summary output for homography mode
                            std::cout << "Feature metrics:" << std::endl;
                            std::cout << "- Average keypoints per image: " << keypoint_stats.mean() << std::endl;
                            std::cout << "- Median keypoints per image: " << keypoint_stats.median() << std::endl;
                            std::cout << "- Min/Max keypoints: " << keypoint_stats.min() << " / " << keypoint_stats.max() << std::endl;
                         // Visualize matches if output folder is specified
                         if (!output_folder.empty()) {
                             // Match descriptors
                             std::vector<cv::DMatch> matches = match_descriptors(
                                 ref_result.descriptors_cv,
                                 all_results[i].descriptors_cv
                             );
                             
                             // Draw matches
                             cv::Mat img_matches;
                             cv::drawMatches(ref_img, ref_result.keypoints_cv,
                                          images[i], all_results[i].keypoints_cv,
                                          matches, img_matches);
                             
                             // Save image
                             std::string out_path = output_folder + "/matches_" + 
                                                   scene_name + "_1_" + std::to_string(i+1) + ".jpg";
                             cv::imwrite(out_path, img_matches);
                         } else {
                            std::cout << "Image 1 -> " << i+1 << ": Skipping (missing homography or no keypoints)" << std::endl;
                        }
                     }
                 }
             }
             } 
         } else {
             // Basic benchmark mode: process all images in the directory
             std::vector<fs::path> image_paths;
             
             try {
                 for (const auto& entry : fs::directory_iterator(input_folder)) {
                     if (entry.is_regular_file() && entry.path().extension() == "." + file_ext) {
                         image_paths.push_back(entry.path());
                     }
                 }
             } catch (const std::exception& e) {
                 std::cerr << "Error reading input directory: " << e.what() << std::endl;
                 return 1;
             }
             
             if (image_paths.empty()) {
                 std::cerr << "No image files found in " << input_folder << " with extension ." << file_ext << std::endl;
                 return 1;
             }
             
             std::cout << "\nRunning basic benchmark on " << image_paths.size() << " images..." << std::endl;
             
             // Process each image individually for detailed timings
             for (size_t i = 0; i < image_paths.size(); i++) {
                 std::string filename = image_paths[i].filename().string();
                 cv::Mat img = cv::imread(image_paths[i].string(), cv::IMREAD_COLOR);
                 
                 if (img.empty()) {
                     std::cerr << "Warning: Could not read image: " << image_paths[i].string() << std::endl;
                     continue;
                 }
                 
                 // Create a vector with just one image
                 std::vector<cv::Mat> single_image = {img};
                 
                 // Process image and measure time
                 Timer timer;
                 auto results = superpoint.run(single_image);
                 double elapsed_ms = timer.elapsed_ms();
                 
                 // Gather statistics
                 time_stats.add(elapsed_ms);
                 keypoint_stats.add(results[0].keypoints_cv.size());
                 throughput_stats.add(1000.0 / elapsed_ms); // images per second
                 
                 std::cout << "Image " << i+1 << "/" << image_paths.size() 
                           << " (" << filename << "): "
                           << elapsed_ms << " ms, "
                           << results[0].keypoints_cv.size() << " keypoints" 
                           << std::endl;
                 
                 // Save visualization if output folder is specified
                 if (!output_folder.empty()) {
                     cv::Mat vis_img = img.clone();
                     
                     // Draw keypoints
                     for (const auto& kp : results[0].keypoints_cv) {
                         cv::circle(vis_img, kp.pt, 3, cv::Scalar(0, 0, 255), -1);
                         cv::circle(vis_img, kp.pt, 5, cv::Scalar(0, 255, 0), 1);
                     }
                     
                     // Save image
                     std::string out_path = output_folder + "/keypoints_" + filename;
                     cv::imwrite(out_path, vis_img);
                 }
             }
         }
         
         // Print summary statistics
         std::cout << "\n===== Benchmark Summary =====" << std::endl;
         
         if (!homography_test) {
             std::cout << "Performance metrics:" << std::endl;
             std::cout << "- Average processing time: " << time_stats.mean() << " ms" << std::endl;
             std::cout << "- Median processing time: " << time_stats.median() << " ms" << std::endl;
             std::cout << "- Min/Max processing time: " << time_stats.min() << " / " << time_stats.max() << " ms" << std::endl;
             std::cout << "- Average throughput: " << throughput_stats.mean() << " images/sec" << std::endl;
             std::cout << "- Median throughput: " << throughput_stats.median() << " images/sec" << std::endl;
             
             std::cout << "\nFeature metrics:" << std::endl;
             std::cout << "- Average keypoints per image: " << keypoint_stats.mean() << std::endl;
             std::cout << "- Median keypoints per image: " << keypoint_stats.median() << std::endl;
             std::cout << "- Min/Max keypoints: " << keypoint_stats.min() << " / " << keypoint_stats.max() << std::endl;
         } else {
             std::cout << "Homography evaluation metrics:" << std::endl;
             std::cout << "- Average repeatability: " << repeatability_stats.mean() << std::endl;
             std::cout << "- Median repeatability: " << repeatability_stats.median() << std::endl;
             std::cout << "- Min/Max repeatability: " << repeatability_stats.min() << " / " << repeatability_stats.max() << std::endl;
             
             std::cout << "\nMatching metrics:" << std::endl;
             std::cout << "- Average matching score: " << matching_score_stats.mean() << std::endl;
             std::cout << "- Median matching score: " << matching_score_stats.median() << std::endl;
             std::cout << "- Min/Max matching score: " << matching_score_stats.min() << " / " << matching_score_stats.max() << std::endl;
         }
         
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
     
     return 0;
 }