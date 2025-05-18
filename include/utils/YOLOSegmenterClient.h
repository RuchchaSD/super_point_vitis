#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <future>
#include <thread>
#include <vector>
#include <cstdlib> // For getenv
#include <filesystem>
#include <random>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using json = nlohmann::json;

// Helper macro for debug printing
#define DEBUG_PRINT(x) if (std::getenv("DEBUG_SEGMENT") != nullptr) { std::cout << "[DEBUG_SEGMENT] " << x << std::endl; }

class YOLOSegmenterClient {
private:
    std::string server_url;
    std::mt19937 rng; // Random number generator for temp filenames

    // Generate a unique temporary filename to avoid conflicts
    std::string generateTempFilename(const std::string& extension = ".png") {
        static std::uniform_int_distribution<int> dist(10000, 99999);
        auto now = std::chrono::system_clock::now();
        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
        auto timestamp = now_ms.time_since_epoch().count();
        return "temp_image_" + std::to_string(timestamp) + "_" + std::to_string(dist(rng)) + extension;
    }

    // Function to decode base64 string to binary data (optimized)
    std::vector<uchar> decodeBase64(const std::string& encoded_string) {
        static const std::string base64_chars = 
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        
        std::vector<uchar> decoded;
        if (encoded_string.empty()) return decoded;
        
        decoded.reserve(encoded_string.length() * 3 / 4); // Pre-allocate memory
        
        int i = 0, j = 0;
        unsigned char char_array_4[4], char_array_3[3];
        std::string input = encoded_string;
        
        DEBUG_PRINT("Decoding base64 string, length: " << encoded_string.length());
        
        // Remove any non-base64 characters like whitespace
        input.erase(std::remove_if(input.begin(), input.end(), [&](char c) {
            return base64_chars.find(c) == std::string::npos && c != '=';
        }), input.end());
        
        int in_len = input.size();
        
        while (in_len-- && input[i] != '=') {
            char_array_4[j++] = input[i++];
            if (j == 4) {
                for (j = 0; j < 4; j++)
                    char_array_4[j] = base64_chars.find(char_array_4[j]);
                
                char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
                
                for (j = 0; j < 3; j++)
                    decoded.push_back(char_array_3[j]);
                j = 0;
            }
        }
        
        if (j) {
            for (int k = j; k < 4; k++)
                char_array_4[k] = 0;
            
            for (int k = 0; k < 4; k++)
                char_array_4[k] = base64_chars.find(char_array_4[k]);
            
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
            
            for (int k = 0; k < j - 1; k++)
                decoded.push_back(char_array_3[k]);
        }
        
        return decoded;
    }

public:
    YOLOSegmenterClient(const std::string& url) : server_url(url), rng(std::random_device{}()) {
        DEBUG_PRINT("YOLOSegmenterClient initialized with URL: " << url);
    }

    // Fetch masks asynchronously and return the combined mask directly
    std::future<cv::Mat> fetchMasksAsync(const cv::Mat& image, cv::Size temp_size = cv::Size(480, 320)) {
        DEBUG_PRINT("fetchMasksAsync called with image size: " << image.size().width << "x" << image.size().height 
                  << ", target size: " << temp_size.width << "x" << temp_size.height);
        
        // Make a copy of the image to avoid reference issues with async execution
        cv::Mat image_copy = image.clone();
        
        return std::async(std::launch::async, [this, image_copy, temp_size]() {
            // Start timing
            auto start_time = std::chrono::high_resolution_clock::now();
            
            DEBUG_PRINT("Async function started for image size: " << image_copy.size().width << "x" << image_copy.size().height);
            
            // Create empty mask as default (for when no objects are detected)
            cv::Mat combined_mask = cv::Mat::zeros(image_copy.size(), CV_8UC1);
            cv::Size original_size = image_copy.size();
            
            // Generate unique temporary filename
            std::string temp_file = generateTempFilename();
            
            try {
                // Resize image for faster processing
                cv::Mat resized_image;
                DEBUG_PRINT("Resizing image from " << original_size.width << "x" << original_size.height 
                          << " to " << temp_size.width << "x" << temp_size.height);
                cv::resize(image_copy, resized_image, temp_size, 0, 0, cv::INTER_AREA);
                
                // Use PNG format for better quality (especially for masks with sharp edges)
                std::vector<int> compression_params;
                compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
                compression_params.push_back(9); // Maximum compression
                
                DEBUG_PRINT("Saving temporary image to: " << temp_file);
                bool write_success = cv::imwrite(temp_file, resized_image, compression_params);
                if (!write_success) {
                    DEBUG_PRINT("ERROR: Failed to write temporary image: " << temp_file);
                    return combined_mask;
                }
                
                // Prepare multipart request for the YOLO server
                DEBUG_PRINT("Creating multipart request with temp file: " << temp_file);
                cpr::Multipart multipart{
                    {"image", cpr::File{temp_file}}
                };
                
                // Make POST request with timeout
                DEBUG_PRINT("Sending POST request to: " << server_url);
                auto request_start = std::chrono::high_resolution_clock::now();
                
                auto response = cpr::Post(
                    cpr::Url{server_url},
                    multipart,
                    cpr::Timeout{10000} // 10 second timeout
                );
                
                auto request_end = std::chrono::high_resolution_clock::now();
                auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(request_end - request_start).count();
                DEBUG_PRINT("HTTP request completed in " << duration_ms << " ms with status code: " << response.status_code);
                
                // Clean up temporary file
                std::remove(temp_file.c_str());
                DEBUG_PRINT("Removed temporary file: " << temp_file);
                
                // Check if request was successful
                if (response.status_code != 200) {
                    DEBUG_PRINT("ERROR: Server returned non-200 status code: " << response.status_code);
                    DEBUG_PRINT("Response: " << response.text);
                    return combined_mask;
                }
                
                // Parse JSON response
                DEBUG_PRINT("Parsing JSON response, length: " << response.text.length());
                
                auto parse_start = std::chrono::high_resolution_clock::now();
                auto j = json::parse(response.text);
                
                // Check for merged_mask first (preferred approach)
                if (j.contains("merged_mask") && !j["merged_mask"].empty()) {
                    DEBUG_PRINT("Found merged_mask in response");
                    std::string merged_mask_b64 = j["merged_mask"].get<std::string>();
                    
                    // Decode base64 to binary
                    std::vector<uchar> mask_data = decodeBase64(merged_mask_b64);
                    if (!mask_data.empty()) {
                        // Decode mask from binary data
                        cv::Mat mask = cv::imdecode(mask_data, cv::IMREAD_UNCHANGED);
                        if (!mask.empty()) {
                            // Resize mask to original image size
                            cv::resize(mask, combined_mask, original_size, 0, 0, cv::INTER_NEAREST);
                            DEBUG_PRINT("Successfully processed merged mask");
                        } else {
                            DEBUG_PRINT("Failed to decode merged mask");
                        }
                    } else {
                        DEBUG_PRINT("Empty decoded data for merged mask");
                    }
                }
                // Fall back to processing individual masks if no merged mask or merged mask processing failed
                else if (j.contains("masks") && !j["masks"].empty()) {
                    auto masks_array = j["masks"].get<std::vector<std::string>>();
                    DEBUG_PRINT("No merged mask found. Received " << masks_array.size() << " individual masks from server");
                    
                    // Process each mask and combine them
                    for (size_t i = 0; i < masks_array.size(); ++i) {
                        DEBUG_PRINT("Processing mask " << (i+1) << "/" << masks_array.size());
                        
                        // Decode base64 to binary
                        std::vector<uchar> mask_data = decodeBase64(masks_array[i]);
                        if (mask_data.empty()) {
                            DEBUG_PRINT("Empty decoded data for mask " << (i+1));
                            continue;
                        }
                        
                        // Decode mask from binary data
                        cv::Mat mask = cv::imdecode(mask_data, cv::IMREAD_UNCHANGED);
                        if (mask.empty()) {
                            DEBUG_PRINT("Failed to decode mask " << (i+1));
                            continue;
                        }
                        
                        // Resize mask to original image size
                        cv::Mat resized_mask;
                        cv::resize(mask, resized_mask, original_size, 0, 0, cv::INTER_NEAREST);
                        
                        // Combine masks (logical OR)
                        cv::bitwise_or(combined_mask, resized_mask, combined_mask);
                    }
                } else {
                    DEBUG_PRINT("No masks or merged_mask in response");
                }
                
                auto parse_end = std::chrono::high_resolution_clock::now();
                auto parse_duration = std::chrono::duration_cast<std::chrono::milliseconds>(parse_end - parse_start).count();
                DEBUG_PRINT("Mask processing completed in " << parse_duration << " ms");
                
                // Count non-zero pixels in the combined mask
                int nonzero_count = cv::countNonZero(combined_mask);
                DEBUG_PRINT("Combined mask contains " << nonzero_count << " non-zero pixels ("
                        << (nonzero_count * 100.0 / (original_size.width * original_size.height)) << "% of image)");
                
            } catch (const std::exception& e) {
                DEBUG_PRINT("ERROR in processing: " << e.what());
                // Clean up temporary file if an exception occurred
                std::remove(temp_file.c_str());
            }
            
            // End timing
            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            DEBUG_PRINT("Total segmentation processing time: " << total_duration << " ms");
            
            return combined_mask;
        });
    }
};

// Helper function to extract mask coordinates
std::vector<cv::Point> getMaskCoordinates(const cv::Mat& mask, int threshold = 128) {
    std::vector<cv::Point> coordinates;
    
    if (std::getenv("DEBUG_SEGMENT") != nullptr) {
        std::cout << "[DEBUG_SEGMENT] Extracting coordinates from mask, size: " 
                  << mask.size().width << "x" << mask.size().height 
                  << ", threshold: " << threshold << std::endl;
    }
    
    // Make sure the mask is valid
    if (mask.empty()) {
        if (std::getenv("DEBUG_SEGMENT") != nullptr) {
            std::cout << "[DEBUG_SEGMENT] ERROR: Empty mask" << std::endl;
        }
        std::cerr << "Error: Empty mask" << std::endl;
        return coordinates;
    }
    
    // Convert mask to binary if needed
    cv::Mat binary_mask;
    if (mask.channels() > 1) {
        if (std::getenv("DEBUG_SEGMENT") != nullptr) {
            std::cout << "[DEBUG_SEGMENT] Converting " << mask.channels() 
                      << "-channel mask to grayscale" << std::endl;
        }
        cv::cvtColor(mask, binary_mask, cv::COLOR_BGR2GRAY);
    } else {
        binary_mask = mask.clone();
    }
    
    if (binary_mask.type() != CV_8U) {
        if (std::getenv("DEBUG_SEGMENT") != nullptr) {
            std::cout << "[DEBUG_SEGMENT] Converting mask to 8-bit unsigned (CV_8U)" << std::endl;
        }
        binary_mask.convertTo(binary_mask, CV_8U);
    }
    
    // Threshold the mask if needed
    cv::Mat thresholded_mask;
    if (std::getenv("DEBUG_SEGMENT") != nullptr) {
        std::cout << "[DEBUG_SEGMENT] Applying threshold at " << threshold << std::endl;
    }
    cv::threshold(binary_mask, thresholded_mask, threshold, 255, cv::THRESH_BINARY);
    
    // Extract coordinates of non-zero pixels - this can be optimized for large masks
    auto start = std::chrono::high_resolution_clock::now();
    
    // Fast way to find non-zero pixels
    std::vector<cv::Point> nonzero_points;
    cv::findNonZero(thresholded_mask, nonzero_points);
    coordinates = std::move(nonzero_points);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    if (std::getenv("DEBUG_SEGMENT") != nullptr) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[DEBUG_SEGMENT] Found " << coordinates.size() << " non-zero pixels in " 
                  << duration << " ms" << std::endl;
    }
    
    return coordinates;
}

// Visualize mask overlaid on the original image
cv::Mat visualizeMask(const cv::Mat& image, const cv::Mat& mask, const cv::Scalar& color = cv::Scalar(0, 0, 255)) {
    if (std::getenv("DEBUG_SEGMENT") != nullptr) {
        std::cout << "[DEBUG_SEGMENT] Visualizing mask, image size: " << image.size().width << "x" << image.size().height
                  << ", mask size: " << mask.size().width << "x" << mask.size().height << std::endl;
    }
    
    cv::Mat visualization;
    image.copyTo(visualization);
    
    // Create a colored overlay
    cv::Mat colored_mask = cv::Mat::zeros(image.size(), CV_8UC3);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Optimized mask coloring using OpenCV functions
    colored_mask.setTo(cv::Scalar(color[0], color[1], color[2]), mask);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Blend the original image with the colored mask
    if (std::getenv("DEBUG_SEGMENT") != nullptr) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[DEBUG_SEGMENT] Created colored mask in " << duration << " ms" << std::endl;
        std::cout << "[DEBUG_SEGMENT] Blending with alpha=0.7 for original, alpha=0.3 for mask" << std::endl;
    }
    
    cv::addWeighted(visualization, 0.7, colored_mask, 0.3, 0, visualization);
    
    return visualization;
}