#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/one_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

namespace caffe {

template <typename Dtype>
OneImageDataLayer<Dtype>::~OneImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void OneImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	iWidth = this->layer_param_.oneimage_param().iwidth();
	iHeight = this->layer_param_.oneimage_param().iheight();
	iCols = this->layer_param_.oneimage_param().icols();
	iRows = this->layer_param_.oneimage_param().irows();
	bGray = this->layer_param_.oneimage_param().bgray();
	bSave = this->layer_param_.oneimage_param().bsave();
	strLoadAddress = this->layer_param_.oneimage_param().strloadaddress();
	strSaveAddress = this->layer_param_.oneimage_param().strsaveaddress();
	iBatchSize = this->layer_param_.oneimage_param().ibatchsize();
	dataSet.clear();
	//载入图像
	cv::Mat fullImage = cv::imread(strLoadAddress, 0);
	CHECK(fullImage.data) << "Load Image Failed!";
	for (int idxY = 0; idxY < iRows; idxY++)
	{
		for (int idxX = 0; idxX < iCols; idxX++)
		{
			cv::Rect rcPatch = cv::Rect(idxX * iWidth, idxY * iHeight, iWidth, iHeight);
			cv::Mat matPatch = cv::Mat(fullImage, rcPatch);
			cv::resize(matPatch, matPatch, cv::Size(10, 10));
			int label = idxY / 5;
			dataSet.push_back(make_pair(matPatch.clone(), label)); 
		}
	}
	LOG(INFO) << "Total Image Number: " << dataSet.size();
	ShuffleImages();
	iImageIdx = 0;
	// data
	vector<int> top_shape(4);
	top_shape[0] = iBatchSize;
	top_shape[1] = 1;
	top_shape[2] = iWidth;
	top_shape[3] = iHeight;
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
	{
		this->prefetch_[i].data_.Reshape(top_shape);
	}
	top[0]->Reshape(top_shape);

	LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
	// label
	vector<int> label_shape(1, iBatchSize);
	top[1]->Reshape(label_shape);
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) 
	{
		this->prefetch_[i].label_.Reshape(label_shape);
	}
}

template <typename Dtype>
void OneImageDataLayer<Dtype>::ShuffleImages() {
  //shuffle(lines_.begin(), lines_.end(), prefetch_rng);
  shuffle(dataSet.begin(), dataSet.end());
}

// This function is called on prefetch thread
template <typename Dtype>
void OneImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
	CPUTimer batch_timer;
	batch_timer.Start();
	double read_time = 0;
	double trans_time = 0;
	CPUTimer timer;
	vector<int> top_shape(4);
	top_shape[0] = iBatchSize;
	top_shape[1] = 1;
	top_shape[2] = iWidth;
	top_shape[3] = iHeight;
	this->transformed_data_.Reshape(top_shape);
	batch->data_.Reshape(top_shape);
	Dtype* prefetch_data = batch->data_.mutable_cpu_data();
	Dtype* prefetch_label = batch->label_.mutable_cpu_data();
	for (int item_id = 0; item_id < iBatchSize; ++item_id)
	{
	  // get a blob
	  timer.Start();
	  CHECK_GT(dataSet.size(), iImageIdx);
	  //取出图像数据和标签数据
	  cv::Mat cv_img = dataSet[iImageIdx].first;
	  int label = dataSet[iImageIdx].second;
	  iImageIdx++;
	  read_time += timer.MicroSeconds();
	  timer.Start();
	  //传输图像数据和标签
	  int offset = batch->data_.offset(item_id);
	  this->transformed_data_.set_cpu_data(prefetch_data + offset);
	  this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
	  prefetch_label[item_id] = label;
	  trans_time += timer.MicroSeconds();
	  if (iImageIdx >= dataSet.size()) {
	    DLOG(INFO) << "Restarting data prefetching from start.";
		iImageIdx = 0;
	     ShuffleImages();
	  }
	}
	batch_timer.Stop();
	DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
	DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(OneImageDataLayer);
REGISTER_LAYER_CLASS(OneImageData);

}  // namespace caffe
#endif  // USE_OPENCV
