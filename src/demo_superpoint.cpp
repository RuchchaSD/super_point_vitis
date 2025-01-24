/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <glog/logging.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "./superpoint.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  if(argc < 3) {
    cout << "Usage: " << argv[0] << " model_name image" << endl;
    char* default_argv[] = {(char*)"demo", (char*)"superpoint_tf.xmodel", (char*)"test.jpg"};
    argv = default_argv;
    // argc = 3;
  }
  int nIter = 1;
  string model_name = argv[1];
  Mat img = imread(argv[2], cv::IMREAD_GRAYSCALE);
  {
    auto superpoint = vitis::ai::SuperPoint::create(model_name, 5);
    if (!superpoint) { // supress coverity complain
       std::cerr <<"create error\n";
       abort();
    }

    vector<Mat> imgs;

    cout << "input batch: " << superpoint->get_input_batch() << endl;
    for(size_t i = 0; i < superpoint->get_input_batch(); ++i)
      imgs.push_back(img);
    
    auto start = chrono::high_resolution_clock::now();
    auto result = superpoint->run(imgs);
    for(int i = 1; i < nIter; ++i){
      result = superpoint->run(imgs);
    }
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>((end - start));
    for(size_t i = 0; i < superpoint->get_input_batch(); ++i) {
      LOG(INFO) << "res scales: " << result[i].scale_h << " " << result[i].scale_w;
      for(size_t k = 0; k < result[i].keypoints.size(); ++k)
        circle(imgs[i], Point(result[i].keypoints[k].first*result[i].scale_w,
               result[i].keypoints[k].second*result[i].scale_h), 1, Scalar(0, 0, 255), -1);
      imwrite(string("result_superpoint_")+to_string(i)+".jpg", imgs[i]);
      //imshow(std::string("result ") + std::to_string(c), result[c]);
      //waitKey(0);
    }

    std::cout << "Time: " << duration.count()/nIter << " ms" << endl;
  }

  
  LOG(INFO) << "BYEBYE";
  return 0;
}
