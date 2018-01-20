#ifndef CAFFE_ONE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_ONE_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class OneImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit OneImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~OneImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OneImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);
  vector<std::pair<cv::Mat, int> > dataSet;
  int iImageIdx;
  int iWidth;
  int iHeight;
  int iCols;
  int iRows;
  bool bGray;
  bool bSave;
  string strLoadAddress;
  string strSaveAddress;
  int iBatchSize;

};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
