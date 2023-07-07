#include "vision/VisionDetector.h"

namespace rt_vision {

cv::Mat VisionDetector::staticResize(const cv::Mat &img) {
  float r = std::min(_input_w / (img.cols * 1.0), _input_h / (img.rows * 1.0));
  int unpad_w = r * img.cols;
  int unpad_h = r * img.rows;
  cv::Mat re(unpad_h, unpad_w, CV_8UC3);
  cv::resize(img, re, re.size());
  cv::Mat out(_input_h, _input_w, CV_8UC3, cv::Scalar(114, 114, 114));
  re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
  return out;
}

void VisionDetector::blobFromImage(const cv::Mat &img, float *blob_data) {
  std::size_t channels = 3;
  std::size_t img_h = img.rows;
  std::size_t img_w = img.cols;

  for (std::size_t c = 0; c < channels; c++) {
    for (std::size_t h = 0; h < img_h; h++) {
      for (std::size_t w = 0; w < img_w; w++) {
        blob_data[c * img_w * img_h + h * img_w + w] =
            (float)img.at<cv::Vec3b>(h, w)[c];
      }
    }
  }
}

void VisionDetector::generateGridsAndStride(
    std::vector<std::size_t> &strides,
    std::vector<GridAndStride> &gridStrides) {
  for (auto stride : strides) {
    std::size_t num_grid_y = _input_h / stride;
    std::size_t num_grid_x = _input_w / stride;

    for (std::size_t g1 = 0; g1 < num_grid_y; g1++) {
      for (std::size_t g0 = 0; g0 < num_grid_x; g0++) {
        GridAndStride GridAndStride{g0, g1, stride};
        gridStrides.push_back(GridAndStride);
      }
    }
  }
}

void VisionDetector::decodeOutput(const float *prob,
                                  std::vector<Object> &objects,
                                  const float bboxConfThresh, float scale,
                                  const int imgWidth, const int imgHeight) {
  std::vector<Object> proposals;
  std::vector<GridAndStride> gridStrides;

  /// Prepare GridAndStrides
  generateGridsAndStride(_strides, gridStrides);

  generateYoloxProposals(gridStrides, prob, bboxConfThresh, proposals);

  qsortDescentInplace(proposals);

  std::vector<int> picked;
  nmsSortedBboxes(proposals, picked, _nmsThresh);

  int count = picked.size();
  objects.resize(count);

  for (int i = 0; i < count; ++i) {
    objects[i] = proposals[picked[i]];

    /// adjust offset to original unpadded
    float x0 = (objects[i].rect.x) / scale;
    float y0 = (objects[i].rect.y) / scale;
    float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
    float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

    /// clip
    x0 = std::max(std::min(x0, (float)(imgWidth - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(imgHeight - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(imgWidth - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(imgHeight - 1)), 0.f);

    objects[i].rect.x = x0;
    objects[i].rect.y = y0;
    objects[i].rect.width = x1 - x0;
    objects[i].rect.height = y1 - y0;
  }
}

void VisionDetector::generateYoloxProposals(
    const std::vector<GridAndStride> &gridStrides, const float *featBlob,
    const float probThreshold, std::vector<Object> &objects) {
  const int numAnchors = gridStrides.size();

  for (int anchorIdx = 0; anchorIdx < numAnchors; ++anchorIdx) {
    const int grid0 = gridStrides[anchorIdx].grid0;
    const int grid1 = gridStrides[anchorIdx].grid1;
    const int stride = gridStrides[anchorIdx].stride;

    const int basic_pos = anchorIdx * (_numClasses + 5);

    /// yolox/models/yolo_head.py decode logic
    float x_center = (featBlob[basic_pos + 0] + grid0) * stride;
    float y_center = (featBlob[basic_pos + 1] + grid1) * stride;
    float w = exp(featBlob[basic_pos + 2]) * stride;
    float h = exp(featBlob[basic_pos + 3]) * stride;
    float x0 = x_center - w * 0.5f;
    float y0 = y_center - h * 0.5f;

    float box_objectness = featBlob[basic_pos + 4];
    for (std::size_t class_idx = 0; class_idx < _numClasses; class_idx++) {
      float box_cls_score = featBlob[basic_pos + 5 + class_idx];
      float box_prob = box_objectness * box_cls_score;
      if (box_prob > probThreshold) {
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = w;
        obj.rect.height = h;
        obj.label = class_idx;
        obj.prob = box_prob;

        objects.push_back(obj);
      }
    }
  }
}

void VisionDetector::qsortDescentInplace(std::vector<Object> &objects) {
  if (objects.empty())
    return;

  qsortDescentInplace(objects, 0, objects.size() - 1);
}

void VisionDetector::qsortDescentInplace(std::vector<Object> &faceobjects,
                                         int left, int right) {
  int i = left;
  int j = right;
  float p = faceobjects[(left + right) / 2].prob;

  while (i <= j) {
    while (faceobjects[i].prob > p)
      i++;

    while (faceobjects[j].prob < p)
      j--;

    if (i <= j) {
      /// swap
      std::swap(faceobjects[i], faceobjects[j]);

      i++;
      j--;
    }
  }

  if (left < j) {
    qsortDescentInplace(faceobjects, left, j);
  } else if (i < right) {
    qsortDescentInplace(faceobjects, i, right);
  }
}

void VisionDetector::nmsSortedBboxes(const std::vector<Object> &faceobjects,
                                     std::vector<int> &picked,
                                     const float nmsThreshold) {
  picked.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; ++i) {
    areas[i] = faceobjects[i].rect.area();
  }

  for (int i = 0; i < n; ++i) {
    const Object &a = faceobjects[i];
    const int picked_size = picked.size();

    int keep = 1;
    for (int j = 0; j < picked_size; ++j) {
      const Object &b = faceobjects[picked[j]];

      /// intersection over union
      float inter_area = intersectionArea(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      if (inter_area / union_area > nmsThreshold)
        keep = 0;
    }

    if (keep)
      picked.push_back(i);
  }
}

float VisionDetector::intersectionArea(const Object &a, const Object &b) {
  cv::Rect_<float> inter = a.rect & b.rect;
  return inter.area();
}

} // namespace rt_vision