#include <vector>

#include "caffe/layers/square_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SquareLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
  power = this->layer_param_.square_param().power();
}

template <typename Dtype>
void SquareLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  //���򴫲������Ǵ�bottom_data����top_data��y = x^a�� �����x����bottom_data,y��top_data
  caffe_powx(count, bottom_data, Dtype(power), top_data);
}

template <typename Dtype>
void SquareLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	//���򴫲������Ǽ���bottom_diff
	//1.�ȼ��㱾����ݶ�dy/dx, dy/dx = ax^(a-1)�������x����bottom_data
	caffe_powx(count, bottom_data, Dtype(power - 1.0), bottom_diff);
	caffe_scal(count, Dtype(power), bottom_diff);
	//2.�ٽ�������ݶ��봫�����������ˣ���bottom_diff
	caffe_mul(count, bottom_diff, top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SquareLayer);
#endif

INSTANTIATE_CLASS(SquareLayer);
REGISTER_LAYER_CLASS(Square);

}  // namespace caffe
