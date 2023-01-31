//
// Created by timber3252 on 2023/1/31.
//

#ifndef ESLAB_ASCEND_PCA_9557_HPP
#define ESLAB_ASCEND_PCA_9557_HPP

#include <thread>

class Pca9557 {
public:
  Pca9557();
  void show_number(int x);

private:
  std::thread thread_pca9557_;
};

#endif //ESLAB_ASCEND_PCA_9557_HPP
