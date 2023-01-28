//
// Created by timber3252 on 2023/1/28.
//

#ifndef ESLAB_ASCEND_PRESENTER_HPP
#define ESLAB_ASCEND_PRESENTER_HPP

#include <opencv2/opencv.hpp>
#include "ascenddk/presenter/agent/presenter_channel.h"

class Presenter {

public:
  explicit Presenter(const std::string &conf);
  void SendImage(cv::Mat &image);

private:
  ascend::presenter::Channel* channel_;
};

#endif //ESLAB_ASCEND_PRESENTER_HPP
