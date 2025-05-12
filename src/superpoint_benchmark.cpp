#include <glog/logging.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>
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

// Statistics class with proper min/max initialization
class Statistics {
public:
    Statistics() : m_sum(0.0), m_min(std::numeric_limits<double>::max()), m_max(std::numeric_limits<double>::lowest()) {}
    
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
    
    double min() const { return m_values.empty() ? 0.0 : m_min; }
    double max() const { return m_values.empty() ? 0.0 : m_max; }
    size_t count() const { return m_values.size(); }
    bool empty() const { return m_values.empty(); }
    
private:
    std::vector<double> m_values;
    double m_sum;
    double m_min;
    double m_max;
};

// Accuracy statistics class for collecting new metrics
class AccuracyStatistics {
public:
    Statistics repeatability_stats;
    Statistics matching_score_stats;
    Statistics map_stats;
    Statistics aucpr_stats;
    Statistics localization_error_stats;
    
    void printSummary() const {
        std::cout << "\n===== Accuracy Metrics Summary =====" << std::endl;
        
        std::cout << "Repeatability metrics:" << std::endl;
        printStats(repeatability_stats, "repeatability");
        
        std::cout << "\nMatching score metrics:" << std::endl;
        printStats(matching_score_stats, "matching score");
        
        std::cout << "\nMean Average Precision (mAP) metrics:" << std::endl;
        printStats(map_stats, "mAP");
        
        std::cout << "\nArea Under Precision-Recall Curve (AUC-PR) metrics:" << std::endl;
        printStats(aucpr_stats, "AUC-PR");
        
        std::cout << "\nLocalization error metrics (lower is better):" << std::endl;
        printStats(localization_error_stats, "localization error (px)");
    }
    
private:
    void printStats(const Statistics& stats, const std::string& name) const {
        if (stats.empty()) {
            std::cout << "- No " << name << " data collected" << std::endl;
            return;
        }
        std::cout << "- Average " << name << ": " << stats.mean() << std::endl;
        std::cout << "- Median " << name << ": " << stats.median() << std::endl;
        std::cout << "- Min/Max " << name << ": " << stats.min() << " / " << stats.max() << std::endl;
    }
};

// Repeatability calculation

// Fixed repeatability calculation - using simple distance calculation approach
double compute_repeatability(
    const std::vector<cv::KeyPoint>& kpts1,
    const std::vector<cv::KeyPoint>& kpts2,
    const cv::Mat& H,
    const cv::Size& img_size,
    double threshold = 3.0) {
    
    // Input validation
    if (kpts1.empty() || kpts2.empty() || H.empty()) {
        std::cout << "Warning: Empty keypoints or homography matrix in repeatability calculation" << std::endl;
        return 0.0;
    }
    
    // Convert keypoints to points
    std::vector<cv::Point2f> points1, points2;
    cv::KeyPoint::convert(kpts1, points1);
    cv::KeyPoint::convert(kpts2, points2);
    
    // Warp points from image 1 to image 2
    std::vector<cv::Point2f> points1_warped;
    try {
        cv::perspectiveTransform(points1, points1_warped, H);
    } catch (const cv::Exception& e) {
        std::cerr << "Error in perspective transform: " << e.what() << std::endl;
        return 0.0;
    }
    
    // Validate there are points in the second image
    if (points2.empty()) {
        std::cout << "Warning: No keypoints in the second image" << std::endl;
        return 0.0;
    }
    
    // Count matches within threshold using direct distance calculation
    int matches = 0;
    int valid_points = 0;
    
    for (size_t i = 0; i < points1_warped.size(); i++) {
        // Check if the warped point is inside the image boundaries
        if (points1_warped[i].x >= 0 && points1_warped[i].x < img_size.width &&
            points1_warped[i].y >= 0 && points1_warped[i].y < img_size.height) {
            
            valid_points++;
            
            // Find the nearest keypoint in the second image
            float min_dist = threshold * threshold; // Initialize with threshold squared
            bool found_match = false;
            
            for (size_t j = 0; j < points2.size(); j++) {
                float dx = points1_warped[i].x - points2[j].x;
                float dy = points1_warped[i].y - points2[j].y;
                float dist_squared = dx*dx + dy*dy;
                
                if (dist_squared < min_dist) {
                    min_dist = dist_squared;
                    found_match = true;
                }
            }
            
            if (found_match) {
                matches++;
            }
        }
    }
    
    // Compute repeatability
    double repeatability = valid_points > 0 ? static_cast<double>(matches) / valid_points : 0.0;
    std::cout << "Valid points: " << valid_points << ", Matches: " << matches 
              << ", Repeatability: " << repeatability << std::endl;
    return repeatability;
}

// Fixed extract matches function - more robust implementation
std::vector<cv::DMatch> match_descriptors(
    const cv::Mat& desc1,
    const cv::Mat& desc2,
    float ratio_threshold = 0.8f) {
    
    std::vector<cv::DMatch> good_matches;
    
    // Input validation
    if (desc1.empty() || desc2.empty()) {
        std::cout << "Warning: Empty descriptors in match_descriptors" << std::endl;
        return good_matches;
    }
    
    if (desc1.rows == 0 || desc2.rows == 0) {
        std::cout << "Warning: Descriptors with zero rows in match_descriptors" << std::endl;
        return good_matches;
    }
    
    cv::Mat desc1_float, desc2_float;
    
    // Convert descriptors to CV_32F if needed
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
    
    // Use BruteForce matcher with L2 norm for better stability
    try {
        // Try FLANN-based matcher first
        std::vector<std::vector<cv::DMatch>> knn_matches;
        cv::Ptr<cv::DescriptorMatcher> matcher;
        
        if (desc1_float.rows >= 1 && desc2_float.rows >= 1) {
            try {
                matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
                matcher->knnMatch(desc1_float, desc2_float, knn_matches, 2);
            } catch (const cv::Exception& e) {
                std::cout << "FLANN matching failed, falling back to BruteForce: " << e.what() << std::endl;
                matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
                matcher->knnMatch(desc1_float, desc2_float, knn_matches, 2);
            }
            
            // Apply ratio test
            for (size_t i = 0; i < knn_matches.size(); i++) {
                if (knn_matches[i].size() >= 2 &&
                    knn_matches[i][0].distance < ratio_threshold * knn_matches[i][1].distance) {
                    good_matches.push_back(knn_matches[i][0]);
                }
            }
        } else {
            std::cout << "Warning: Not enough rows in descriptors for matching" << std::endl;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error in descriptor matching: " << e.what() << std::endl;
    }
    
    return good_matches;
}

// Fixed compute_matching_score function
double compute_matching_score(
    const std::vector<cv::KeyPoint>& kpts1,
    const std::vector<cv::KeyPoint>& kpts2,
    const cv::Mat& desc1,
    const cv::Mat& desc2,
    const cv::Mat& H,
    double threshold = 3.0) {
    
    // Input validation
    if (kpts1.empty() || kpts2.empty() || desc1.empty() || desc2.empty() || H.empty()) {
        std::cout << "Warning: Empty keypoints, descriptors, or homography matrix in matching score calculation" << std::endl;
        return 0.0;
    }
    
    // Match descriptors using the fixed function
    std::vector<cv::DMatch> matches = match_descriptors(desc1, desc2, 0.8f);
    
    if (matches.empty()) {
        std::cout << "Warning: No matches found in matching score calculation" << std::endl;
        return 0.0;
    }
    
    // Extract matched points with bounds checking
    std::vector<cv::Point2f> matched1, matched2;
    for (const auto& m : matches) {
        if (m.queryIdx < static_cast<int>(kpts1.size()) && 
            m.trainIdx < static_cast<int>(kpts2.size())) {
            matched1.push_back(kpts1[m.queryIdx].pt);
            matched2.push_back(kpts2[m.trainIdx].pt);
        }
    }
    
    if (matched1.empty() || matched2.empty()) {
        std::cout << "Warning: No valid matched points" << std::endl;
        return 0.0;
    }
    
    // Warp points from image 1 to image 2
    std::vector<cv::Point2f> matched1_warped;
    try {
        cv::perspectiveTransform(matched1, matched1_warped, H);
    } catch (const cv::Exception& e) {
        std::cerr << "Error in perspective transform: " << e.what() << std::endl;
        return 0.0;
    }
    
    // Count correct matches
    int correct_matches = 0;
    for (size_t i = 0; i < matched1_warped.size(); i++) {
        float dist = cv::norm(matched1_warped[i] - matched2[i]);
        if (dist < threshold) {
            correct_matches++;
        }
    }
    
    // Compute matching score
    double matching_score = static_cast<double>(correct_matches) / matches.size();
    std::cout << "Total matches: " << matches.size() << ", Correct matches: " << correct_matches 
              << ", Matching score: " << matching_score << std::endl;
    return matching_score;
}

// Fixed compute_mAP function
double compute_mAP(
    const std::vector<cv::KeyPoint>& kpts1,
    const std::vector<cv::KeyPoint>& kpts2,
    const cv::Mat& desc1,
    const cv::Mat& desc2,
    const cv::Mat& H,
    const cv::Size& img_size,
    double threshold = 3.0) {
    
    // Input validation
    if (kpts1.empty() || kpts2.empty() || desc1.empty() || desc2.empty() || H.empty()) {
        std::cout << "Warning: Empty keypoints, descriptors, or homography matrix in mAP calculation" << std::endl;
        return 0.0;
    }
    
    // Convert keypoints to points
    std::vector<cv::Point2f> points1, points2;
    cv::KeyPoint::convert(kpts1, points1);
    cv::KeyPoint::convert(kpts2, points2);
    
    // Warp points from image 1 to image 2
    std::vector<cv::Point2f> points1_warped;
    try {
        cv::perspectiveTransform(points1, points1_warped, H);
    } catch (const cv::Exception& e) {
        std::cerr << "Error in perspective transform: " << e.what() << std::endl;
        return 0.0;
    }
    
    // Match descriptors with the fixed function
    std::vector<cv::DMatch> matches = match_descriptors(desc1, desc2, 0.8f);
    
    if (matches.empty()) {
        std::cout << "Warning: No matches found in mAP calculation" << std::endl;
        return 0.0;
    }
    
    // Calculate distances for each match with bounds checking
    std::vector<std::pair<float, bool>> match_distances;
    for (const auto& m : matches) {
        if (m.queryIdx < static_cast<int>(kpts1.size()) && 
            m.trainIdx < static_cast<int>(kpts2.size()) &&
            m.queryIdx < static_cast<int>(points1_warped.size())) {
            
            // Calculate Euclidean distance between warped point and matched point
            cv::Point2f warped_pt = points1_warped[m.queryIdx];
            cv::Point2f matched_pt = points2[m.trainIdx];
            
            // Check if the warped point is inside the image
            if (warped_pt.x >= 0 && warped_pt.x < img_size.width &&
                warped_pt.y >= 0 && warped_pt.y < img_size.height) {
                
                float dist = cv::norm(warped_pt - matched_pt);
                bool is_correct = (dist < threshold);
                match_distances.push_back(std::make_pair(m.distance, is_correct));
            }
        }
    }
    
    if (match_distances.empty()) {
        std::cout << "Warning: No valid matches for mAP calculation" << std::endl;
        return 0.0;
    }
    
    // Sort matches by descriptor distance (ascending)
    std::sort(match_distances.begin(), match_distances.end(),
              [](const std::pair<float, bool>& a, const std::pair<float, bool>& b) {
                  return a.first < b.first;
              });
    
    // Calculate precision at each recall point
    int total_correct = 0;
    double ap = 0.0;
    int total_gt = 0;
    
    // Count total correct matches (ground truth)
    for (const auto& md : match_distances) {
        if (md.second) total_gt++;
    }
    
    if (total_gt == 0) {
        std::cout << "Warning: No correct matches found for mAP calculation" << std::endl;
        return 0.0;
    }
    
    // Calculate average precision
    for (size_t i = 0; i < match_distances.size(); i++) {
        if (match_distances[i].second) {
            total_correct++;
            // Precision at this point
            double precision = static_cast<double>(total_correct) / (i + 1);
            // Add to AP (each correct match contributes equally)
            ap += precision;
        }
    }
    
    ap /= total_gt;
    
    std::cout << "Total matches: " << match_distances.size() 
              << ", Correct matches: " << total_gt
              << ", mAP: " << ap << std::endl;
    
    return ap;
}
// Compute Precision-Recall curve
std::pair<std::vector<double>, std::vector<double>> compute_precision_recall(
    const std::vector<cv::KeyPoint>& kpts1,
    const std::vector<cv::KeyPoint>& kpts2,
    const cv::Mat& desc1,
    const cv::Mat& desc2,
    const cv::Mat& H,
    const cv::Size& img_size,
    double threshold = 3.0,
    int num_points = 11) {
    
    std::vector<double> precisions, recalls;
    
    if (kpts1.empty() || kpts2.empty() || desc1.empty() || desc2.empty() || H.empty()) {
        std::cout << "Warning: Empty keypoints, descriptors, or homography matrix in PR calculation" << std::endl;
        return {precisions, recalls};
    }
    
    // Convert keypoints to points
    std::vector<cv::Point2f> points1, points2;
    cv::KeyPoint::convert(kpts1, points1);
    cv::KeyPoint::convert(kpts2, points2);
    
    // Warp points from image 1 to image 2
    std::vector<cv::Point2f> points1_warped;
    try {
        cv::perspectiveTransform(points1, points1_warped, H);
    } catch (const cv::Exception& e) {
        std::cerr << "Error in perspective transform: " << e.what() << std::endl;
        return {precisions, recalls};
    }
    
    // Match descriptors with low ratio threshold to get more matches
    std::vector<cv::DMatch> matches = match_descriptors(desc1, desc2, 0.9f);
    
    if (matches.empty()) {
        std::cout << "Warning: No matches found in PR calculation" << std::endl;
        return {precisions, recalls};
    }
    
    // Get distances and correctness of matches
    std::vector<std::pair<float, bool>> match_distances;
    for (const auto& m : matches) {
        if (m.queryIdx < kpts1.size() && m.trainIdx < kpts2.size()) {
            cv::Point2f warped_pt = points1_warped[m.queryIdx];
            cv::Point2f matched_pt = points2[m.trainIdx];
            
            if (warped_pt.x >= 0 && warped_pt.x < img_size.width &&
                warped_pt.y >= 0 && warped_pt.y < img_size.height) {
                
                float dist = cv::norm(warped_pt - matched_pt);
                bool is_correct = (dist < threshold);
                match_distances.push_back(std::make_pair(m.distance, is_correct));
            }
        }
    }
    
    if (match_distances.empty()) {
        std::cout << "Warning: No valid matches for PR calculation" << std::endl;
        return {precisions, recalls};
    }
    
    // Sort matches by descriptor distance (ascending)
    std::sort(match_distances.begin(), match_distances.end(),
              [](const std::pair<float, bool>& a, const std::pair<float, bool>& b) {
                  return a.first < b.first;
              });
    
    // Count total correct matches (ground truth)
    int total_gt = 0;
    for (const auto& md : match_distances) {
        if (md.second) total_gt++;
    }
    
    if (total_gt == 0) {
        std::cout << "Warning: No correct matches found for PR calculation" << std::endl;
        return {precisions, recalls};
    }
    
    // Calculate precision and recall at different thresholds
    for (int i = 0; i <= num_points; i++) {
        // Take i/num_points of matches (from best to worst)
        int num_to_take = i * match_distances.size() / num_points;
        if (num_to_take == 0 && i > 0) num_to_take = 1;
        if (num_to_take > match_distances.size()) num_to_take = match_distances.size();
        
        int correct_matches = 0;
        for (int j = 0; j < num_to_take; j++) {
            if (match_distances[j].second) correct_matches++;
        }
        
        double precision = num_to_take > 0 ? static_cast<double>(correct_matches) / num_to_take : 1.0;
        double recall = total_gt > 0 ? static_cast<double>(correct_matches) / total_gt : 0.0;
        
        precisions.push_back(precision);
        recalls.push_back(recall);
    }
    
    return {precisions, recalls};
}

// Calculate the Area Under Precision-Recall Curve (AUC-PR)
double compute_aucpr(
    const std::vector<double>& precisions,
    const std::vector<double>& recalls) {
    
    if (precisions.size() != recalls.size() || precisions.empty()) {
        std::cout << "Warning: Invalid precision-recall data for AUC-PR calculation" << std::endl;
        return 0.0;
    }
    
    double auc = 0.0;
    for (size_t i = 1; i < recalls.size(); i++) {
        // Trapezoidal rule for area calculation
        double width = recalls[i] - recalls[i-1];
        double height = (precisions[i] + precisions[i-1]) / 2.0;
        auc += width * height;
    }
    
    return auc;
}

// Compute localization error
double compute_localization_error(
    const std::vector<cv::KeyPoint>& kpts1,
    const std::vector<cv::KeyPoint>& kpts2,
    const cv::Mat& desc1,
    const cv::Mat& desc2,
    const cv::Mat& H,
    const cv::Size& img_size) {
    
    if (kpts1.empty() || kpts2.empty() || desc1.empty() || desc2.empty() || H.empty()) {
        std::cout << "Warning: Empty keypoints, descriptors, or homography matrix in localization error calculation" << std::endl;
        return std::numeric_limits<double>::max();
    }
    
    // Convert keypoints to points
    std::vector<cv::Point2f> points1, points2;
    cv::KeyPoint::convert(kpts1, points1);
    cv::KeyPoint::convert(kpts2, points2);
    
    // Warp points from image 1 to image 2
    std::vector<cv::Point2f> points1_warped;
    try {
        cv::perspectiveTransform(points1, points1_warped, H);
    } catch (const cv::Exception& e) {
        std::cerr << "Error in perspective transform: " << e.what() << std::endl;
        return std::numeric_limits<double>::max();
    }
    
    // Match descriptors
    std::vector<cv::DMatch> matches = match_descriptors(desc1, desc2, 0.8f);
    
    if (matches.empty()) {
        std::cout << "Warning: No matches found in localization error calculation" << std::endl;
        return std::numeric_limits<double>::max();
    }
    
    // Calculate localization error for each match
    std::vector<double> errors;
    for (const auto& m : matches) {
        if (m.queryIdx < kpts1.size() && m.trainIdx < kpts2.size()) {
            cv::Point2f warped_pt = points1_warped[m.queryIdx];
            cv::Point2f matched_pt = points2[m.trainIdx];
            
            if (warped_pt.x >= 0 && warped_pt.x < img_size.width &&
                warped_pt.y >= 0 && warped_pt.y < img_size.height) {
                
                double error = cv::norm(warped_pt - matched_pt);
                errors.push_back(error);
            }
        }
    }
    
    if (errors.empty()) {
        std::cout << "Warning: No valid matches for localization error calculation" << std::endl;
        return std::numeric_limits<double>::max();
    }
    
    // Calculate mean localization error
    double total_error = 0.0;
    for (const auto& err : errors) {
        total_error += err;
    }
    
    double mean_error = total_error / errors.size();
    std::cout << "Mean localization error: " << mean_error 
              << " pixels over " << errors.size() << " matches" << std::endl;
    
    return mean_error;
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
    int num_threads = 4;
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
    std::cout << "- Input folder: " << input_folder << std::endl;
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
        AccuracyStatistics accuracy_stats;
        
        // Create thread-safe queues for SuperPoint processing
        ThreadSafeQueue<InputQueueItem> input_queue(20);
        ThreadSafeQueue<SuperPointResult> output_queue(50);
        std::vector<SuperPointResult> all_results;
        std::mutex results_mutex;
        std::atomic<int> processed_count(0);
        std::atomic<int> expected_count(0);
        
        if (homography_test) {
            // Process HPatches dataset
            std::cout << "\nProcessing HPatches dataset for homography evaluation..." << std::endl;
            
            bool processed_any_scene = false;
            
            for (const auto& scene_dir_entry : fs::directory_iterator(input_folder)) {
                if (!fs::is_directory(scene_dir_entry)) continue;
                
                fs::path scene_dir = scene_dir_entry.path();
                std::string scene_name = scene_dir.filename().string();
                std::cout << "\nProcessing scene: " << scene_name << std::endl;
                
                // Load all images in the scene
                std::vector<cv::Mat> images;
                std::vector<std::string> image_names;
                
                for (const auto& entry : fs::directory_iterator(scene_dir)) {
                    if (entry.is_regular_file() && 
                        entry.path().extension() == "." + file_ext) {
                        
                        std::string filename = entry.path().filename().string();
                        // Check if the file is a numbered image (like 1.ppm, 2.ppm, etc.)
                        if (isdigit(filename[0])) {
                            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
                            if (!img.empty()) {
                                images.push_back(img);
                                image_names.push_back(filename);
                                std::cout << "Loaded image: " << filename << " (" << img.size() << ")" << std::endl;
                            } else {
                                std::cerr << "Warning: Failed to load image: " << entry.path().string() << std::endl;
                            }
                        }
                    }
                }
                
                if (images.size() < 2) {
                    std::cout << "Skip scene " << scene_name << ": not enough images found" << std::endl;
                    continue;
                }
                
                std::cout << "Found " << images.size() << " images in scene" << std::endl;
                processed_any_scene = true;
                
                // Load homography matrices for reference image (1.ppm)
                std::vector<cv::Mat> homographies;
                for (size_t i = 1; i < images.size(); i++) {
                    // Extract just the number from image name (e.g., "2" from "2.ppm")
                    std::string img_num = image_names[i].substr(0, image_names[i].find('.'));
                    std::string h_filename = "H_1_" + img_num;
                    fs::path h_path = scene_dir / h_filename;
                    
                    if (fs::exists(h_path)) {
                        cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
                        std::ifstream h_file(h_path);
                        if (h_file.is_open()) {
                            for (int r = 0; r < 3; r++) {
                                for (int c = 0; c < 3; c++) {
                                    double val;
                                    h_file >> val;
                                    if (h_file.good()) {
                                        H.at<double>(r, c) = val;
                                    } else {
                                        std::cerr << "Warning: Error reading value at " << r << "," << c 
                                                  << " from homography file " << h_path << std::endl;
                                    }
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
                
                // Extract features from all images using queue-based approach
                all_results.clear();
                all_results.resize(images.size());
                processed_count = 0;
                expected_count = images.size();
                
                // Start the SuperPointFast processor
                superpoint.run(input_queue, output_queue);
                
                // Start consumer thread to collect results
                std::thread consumer_thread([&output_queue, &all_results, &processed_count, &results_mutex]() {
                    SuperPointResult result;
                    while (output_queue.dequeue(result)) {
                        // Store result in the right position
                        {
                            std::lock_guard<std::mutex> lock(results_mutex);
                            all_results[result.index] = result;
                        }
                        processed_count++;
                    }
                });
                
                // Start producer thread - feed images
                Timer batch_timer;
                std::thread producer_thread([&images, &input_queue]() {
                    for (size_t i = 0; i < images.size(); i++) {
                        InputQueueItem item;
                        item.index = i;
                        item.image = images[i];
                        item.name = std::to_string(i+1);  // name as index+1 to match 1.ppm, 2.ppm format
                        input_queue.enqueue(item);
                    }
                    input_queue.shutdown();
                });
                
                // Wait for producer to finish
                producer_thread.join();
                
                // Wait for consumer to finish
                consumer_thread.join();
                
                double batch_time = batch_timer.elapsed_ms();
                
                std::cout << "Batch processing completed: " << processed_count << "/" << expected_count << " images" << std::endl;
                std::cout << "Batch processing time: " << batch_time << " ms" << std::endl;
                std::cout << "Per image time: " << batch_time / images.size() << " ms" << std::endl;
                
                // Track keypoint statistics
                for (size_t i = 0; i < all_results.size(); i++) {
                    std::cout << "Image " << i+1 << ": " << all_results[i].keypoints_cv.size() 
                              << " keypoints extracted" << std::endl;
                    keypoint_stats.add(all_results[i].keypoints_cv.size());
                }
                
                // Evaluate repeatability and matching
                if (!homographies.empty()) {
                    cv::Mat& ref_img = images[0];
                    SuperPointResult& ref_result = all_results[0];
                    
                    for (size_t i = 1; i < images.size() && (i-1) < homographies.size(); i++) {
                        if (ref_result.keypoints_cv.empty() || all_results[i].keypoints_cv.empty()) {
                            std::cout << "Image 1 -> " << i+1 << ": Skipping (no keypoints)" << std::endl;
                            continue;
                        }
                        
                        if (homographies[i-1].empty()) {
                            std::cout << "Image 1 -> " << i+1 << ": Skipping (empty homography)" << std::endl;
                            continue;
                        }
                        
                        std::cout << "Evaluating Image 1 -> " << i+1 << ":" << std::endl;
                        
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
                        } else {
                            std::cout << "Warning: Empty descriptors, skipping matching score calculation" << std::endl;
                        }
                        
                        // Calculate mAP
                        double map = compute_mAP(
                            ref_result.keypoints_cv,
                            all_results[i].keypoints_cv,
                            ref_result.descriptors_cv,
                            all_results[i].descriptors_cv,
                            homographies[i-1],
                            ref_img.size()
                        );
                        accuracy_stats.map_stats.add(map);
                        
                        // Calculate precision-recall curve and AUC-PR
                        auto pr_data = compute_precision_recall(
                            ref_result.keypoints_cv,
                            all_results[i].keypoints_cv,
                            ref_result.descriptors_cv,
                            all_results[i].descriptors_cv,
                            homographies[i-1],
                            ref_img.size()
                        );
                        double aucpr = compute_aucpr(pr_data.first, pr_data.second);
                        accuracy_stats.aucpr_stats.add(aucpr);
                        
                        // Calculate localization error
                        double loc_error = compute_localization_error(
                            ref_result.keypoints_cv,
                            all_results[i].keypoints_cv,
                            ref_result.descriptors_cv,
                            all_results[i].descriptors_cv,
                            homographies[i-1],
                            ref_img.size()
                        );
                        accuracy_stats.localization_error_stats.add(loc_error);
                        
                        std::cout << "Image 1 -> " << i+1 << ": "
                                  << "Keypoints: " << ref_result.keypoints_cv.size() << "/" << all_results[i].keypoints_cv.size()
                                  << ", Repeatability = " << repeatability 
                                  << ", Matching Score = " << matching_score 
                                  << ", mAP = " << map
                                  << ", AUC-PR = " << aucpr
                                  << ", Loc. Error = " << loc_error << " px"
                                  << std::endl;
                                  
                        repeatability_stats.add(repeatability);
                        matching_score_stats.add(matching_score);
                        
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
                            
                            // Create and save PR curve
                            int width = 600, height = 600;
                            cv::Mat pr_curve(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
                            
                            // Draw axes
                            cv::line(pr_curve, cv::Point(50, height - 50), cv::Point(width - 50, height - 50), cv::Scalar(0, 0, 0), 2);
                            cv::line(pr_curve, cv::Point(50, height - 50), cv::Point(50, 50), cv::Scalar(0, 0, 0), 2);
                            
                            // Draw labels
                            cv::putText(pr_curve, "Recall", cv::Point(width/2, height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
                            cv::putText(pr_curve, "Precision", cv::Point(10, height/2), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2, cv::LINE_AA, false);
                            
                            // Draw grid
                            for (int j = 1; j < 10; j++) {
                                int x = 50 + j * (width - 100) / 10;
                                int y = height - 50 - j * (height - 100) / 10;
                                cv::line(pr_curve, cv::Point(x, height - 50), cv::Point(x, height - 45), cv::Scalar(0, 0, 0), 1);
                                cv::line(pr_curve, cv::Point(50, y), cv::Point(45, y), cv::Scalar(0, 0, 0), 1);
                            }
                            
                            // Plot PR curve
                            std::vector<cv::Point> curve_points;
                            for (size_t j = 0; j < pr_data.first.size(); j++) {
                                int x = 50 + static_cast<int>(pr_data.second[j] * (width - 100));
                                int y = height - 50 - static_cast<int>(pr_data.first[j] * (height - 100));
                                curve_points.push_back(cv::Point(x, y));
                            }
                            
                            // Draw curve
                            for (size_t j = 1; j < curve_points.size(); j++) {
                                cv::line(pr_curve, curve_points[j-1], curve_points[j], cv::Scalar(0, 0, 255), 2);
                            }
                            
                            // Add AUC-PR value
                            std::stringstream ss;
                            ss << "AUC-PR = " << std::fixed << std::setprecision(4) << aucpr;
                            cv::putText(pr_curve, ss.str(), cv::Point(width - 250, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
                            
                            // Save PR curve
                            std::string pr_path = output_folder + "/pr_curve_" + scene_name + "_1_" + std::to_string(i+1) + ".jpg";
                            cv::imwrite(pr_path, pr_curve);
                        }
                    }
                } else {
                    std::cout << "Warning: No homographies found for scene " << scene_name << std::endl;
                }
            }
            
            if (!processed_any_scene) {
                std::cerr << "Error: No valid scenes processed in homography test mode" << std::endl;
            }
            
            // Update accuracy statistics and print summary
            accuracy_stats.repeatability_stats = repeatability_stats;
            accuracy_stats.matching_score_stats = matching_score_stats;
            accuracy_stats.printSummary();
            
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

                // Create fresh queues for each image
                ThreadSafeQueue<InputQueueItem> single_input_queue(5);
                ThreadSafeQueue<SuperPointResult> single_output_queue(5);
                processed_count = 0;
                
                // Start the processor with fresh queues
                superpoint.run(single_input_queue, single_output_queue);
                
                // Process a single image and measure time
                Timer timer;
                
                // Create a single result container
                SuperPointResult single_result;
                
                // Start consumer thread
                std::thread consumer_thread([&single_output_queue, &single_result, &processed_count]() {
                    if (single_output_queue.dequeue(single_result)) {
                        processed_count++;
                    }
                });
                
                // Start producer thread with a single image
                std::thread producer_thread([&img, &single_input_queue, i, &filename]() {
                    InputQueueItem item;
                    item.index = 0; // Only one image
                    item.image = img;
                    item.name = filename;
                    single_input_queue.enqueue(item);
                    single_input_queue.shutdown();
                });
                
                producer_thread.join();
                consumer_thread.join();
                
                double elapsed_ms = timer.elapsed_ms();
                
                // Gather statistics if we got a result
                if (processed_count > 0) {
                    time_stats.add(elapsed_ms);
                    keypoint_stats.add(single_result.keypoints_cv.size());
                    throughput_stats.add(1000.0 / elapsed_ms); // images per second
                    
                    std::cout << "Image " << i+1 << "/" << image_paths.size() 
                              << " (" << filename << "): "
                              << elapsed_ms << " ms, "
                              << single_result.keypoints_cv.size() << " keypoints" 
                              << std::endl;
                    
                    // Save visualization if output folder is specified
                    if (!output_folder.empty()) {
                        cv::Mat vis_img = img.clone();
                        
                        // Draw keypoints
                        for (const auto& kp : single_result.keypoints_cv) {
                            cv::circle(vis_img, kp.pt, 3, cv::Scalar(0, 0, 255), -1);
                            cv::circle(vis_img, kp.pt, 5, cv::Scalar(0, 255, 0), 1);
                        }
                        
                        // Save image
                        std::string out_path = output_folder + "/keypoints_" + filename;
                        cv::imwrite(out_path, vis_img);
                    }
                } else {
                    std::cerr << "Warning: No result received for image " << filename << std::endl;
                }
            }
        }
        
        // Print summary statistics for non-homography test
        if (!homography_test) {
            std::cout << "\n===== Benchmark Summary =====" << std::endl;
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
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}