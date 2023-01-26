# eslab

An assignment for Embedded Systems

**Please note that this project could only be deployed on Huawei Ascend-based platforms, and has only been tested on Huawei Atlas200.**

## Face Recognition

Inspired by [Ascend Community Samples](https://gitee.com/ascend/samples), re-implement to solve [this issue](https://toscode.gitee.com/ascend/samples/issues/I51GHH).

### Models

- [face_detection](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/facedetection/ATC_resnet10-SSD_caffe_AE): A model for face detection. It is a converted network model based on Caffe's Resnet10-SSD300 model.
- [vanillacnn](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/vanillacnn/ATC_vanillacnn_caffe_AE): A model of face feature point masking. It is a converted network model based on Caffe's VanillaCNN model.
- [sphereface](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/sphereface/ATC_sphereface_caffe_AE): A model to get the face feature vector. It is a converted network model based on Caffe's SphereFace model. 

## License

The project is licensed under [MIT License](license) if no special instructions are given.