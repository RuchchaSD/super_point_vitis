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
#include <opencv2/features2d.hpp>
#include "SuperPointFast.h"
#include "FeatureIO.h"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

void print_usage(char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options] model_name input_folder output_folder" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -t <num>      Number of preprocessing/postprocessing threads to use (default: 2)" << std::endl;
    std::cout << "                Note: DPU runners fixed at 4" << std::endl;
    std::cout << "  -f <ext>      File extension filter (default: jpg)" << std::endl;
    std::cout << "Example: " << prog_name << " -t 4 superpoint_tf.xmodel ./imgs ./features" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default parameters
    int num_threads = 2;
    std::string file_ext = "jpg";
    std::string model_name;
    std::string input_folder;
    std::string output_folder;
    
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
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            break;
        }
    }
    
    // Get positional arguments
    if (arg_index + 2 < argc) {
        model_name = argv[arg_index];
        input_folder = argv[arg_index + 1];
        output_folder = argv[arg_index + 2];
    } else {
        std::cerr << "Error: Missing required positional arguments" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Create output directory structure
    std::string kpts_folder = output_folder + "/kpts";
    std::string desc_folder = output_folder + "/desc";
    
    try {
        if (!fs::exists(output_folder)) {
            fs::create_directories(output_folder);
        }
        if (!fs::exists(kpts_folder)) {
            fs::create_directories(kpts_folder);
        }
        if (!fs::exists(desc_folder)) {
            fs::create_directories(desc_folder);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error creating output directories: " << e.what() << std::endl;
        return 1;
    }
    
    // Get list of image files
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
    
    std::cout << "SuperPoint Feature Extraction:" << std::endl;
    std::cout << "- Number of pre/post-processing threads: " << num_threads << std::endl;
    std::cout << "- Number of DPU runners: 4 (fixed)" << std::endl;
    std::cout << "- Model: " << model_name << std::endl;
    std::cout << "- Input folder: " << input_folder << std::endl;
    std::cout << "- Output folder: " << output_folder << std::endl;
    std::cout << "- Keypoints folder: " << kpts_folder << std::endl;
    std::cout << "- Descriptors folder: " << desc_folder << std::endl;
    std::cout << "- File extension: " << file_ext << std::endl;
    std::cout << "- Total images found: " << image_paths.size() << std::endl;

    try {
        // Initialize SuperPointFast
        auto superpoint = SuperPointFast(model_name, num_threads);
        
        // Create thread-safe queues
        ThreadSafeQueue<InputQueueItem> input_queue(20);
        ThreadSafeQueue<SuperPointResult> output_queue(50);
        
       
        // Start the SuperPointFast processor 
        superpoint.run(input_queue, output_queue);
        
        // Start producer thread to feed images when rate limiting allows
        std::thread producer_thread([&image_paths, &input_queue, &kpts_folder, &desc_folder]() {
            size_t count = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            
            for (const auto& img_path : image_paths) {
                // Extract base filename from the image path
                std::string base_filename = fs::path(img_path).stem().string();
                
                // Define filenames for keypoints and descriptors
                std::string kpts_filename = kpts_folder + "/" + base_filename + ".kp";
                std::string desc_filename = desc_folder + "/" + base_filename + ".desc";
                
                // Check if both files already exist
                if (fs::exists(kpts_filename) && fs::exists(desc_filename)) {
                    std::cout << "Skipping " << base_filename << " - feature files already exist" << std::endl;
                    continue;  // Skip to next image
                }
                
                // Read image
                cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
                if (img.empty()) {
                    std::cerr << "Warning: Could not read image: " << img_path.string() << std::endl;
                    continue;
                }
                
                // Create input queue item (store filename in the data field)
                InputQueueItem item;
                item.index = count++;
                item.image = img;
                item.name = img_path.filename().string();
                
                // Add to queue
                input_queue.enqueue(item);
                //there should be some delay to preserve order
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "All " << count << " images queued in " << duration.count() << " ms" << std::endl;
            
            // Signal that no more images will be added
            input_queue.shutdown();
        });
        
        // Start consumer thread to process results and save features
        std::thread consumer_thread([&output_queue, &kpts_folder, &desc_folder]() {
            size_t result_count = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            
            SuperPointResult result;
            while (output_queue.dequeue(result)) {
                result_count++;
                
                // Extract base filename without extension
                std::string base_filename = fs::path(result.name).stem().string();
                if (base_filename.empty()) {
                    // If name wasn't passed through, use the index
                    base_filename = "image_" + std::to_string(result.index);
                }
                
                // Define filenames for keypoints and descriptors
                std::string kpts_filename = kpts_folder + "/" + base_filename + ".kp";
                std::string desc_filename = desc_folder + "/" + base_filename + ".desc";
                
                // Use keypoints_cv and descriptors_cv directly
                bool saved = FeatExtraction::FeatureIO::saveFeatures(result.keypoints_cv, result.descriptors_cv, 
                                                                     kpts_filename, desc_filename);
                
                if (saved) {
                    std::cout << "Saved features for " << base_filename 
                              << " (keypoints: " << result.keypoints_cv.size() << ")" << std::endl;
                } else {
                    std::cerr << "Error saving features for " << base_filename << std::endl;
                }
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "All " << result_count << " feature sets extracted and saved in " 
                      << duration.count() << " ms" << std::endl;
        });
        
        // Wait for all threads to complete
        producer_thread.join();
        consumer_thread.join();
        
        // Calculate statistics
        std::cout << "Feature extraction completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}