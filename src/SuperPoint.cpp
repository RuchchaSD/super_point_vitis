// SuperPoint.cpp
#include "SuperPoint.hpp"
#include "SuperPointSingleImp.h"
#include "SuperPointMultiImp.h"
namespace vitis {
  namespace ai {
    SuperPoint::SuperPoint(const std::string& model_name) {}

    SuperPoint::~SuperPoint() {}

    std::unique_ptr<SuperPoint> SuperPoint::create(const std::string& model_name, 
                                                  ImplType impl_type,
                                                  int num_runners) {
      if (impl_type == ImplType::SINGLE_THREADED) {
        return std::unique_ptr<SuperPointSingleImp>(new SuperPointSingleImp(model_name));
      } else {
        return std::unique_ptr<SuperPointMultiImp>(new SuperPointMultiImp(model_name, num_runners));
      }
    }

  }  // namespace ai
}  // namespace vitis
