
#include "ma_nms.h"

#include <algorithm>
#include <cmath>
#include <forward_list>
#include <vector>

namespace ma::utils {

template <typename Container,
          typename T = typename Container::value_type,
          std::enable_if_t<has_iterator_support_v<Container> && std::is_base_of_v<ma_bbox_t, T>, bool> = true>
constexpr void nms(Container bboxes, float threshold_iou, float threshold_score, bool soft_nms, bool multi_target) {
    std::sort(bboxes.begin(), bboxes.end(), [](const auto& box1, const auto& box2) { return box1.score > box2.score; });

    static const auto compute_iou = [](const auto& box1, const auto& box2) {
        float x1    = std::max(box1.x, box2.x);
        float y1    = std::max(box1.y, box2.y);
        float x2    = std::min(box1.x + box1.w, box2.x + box2.w);
        float y2    = std::min(box1.y + box1.h, box2.y + box2.h);
        float w     = std::max(0.0f, x2 - x1);
        float h     = std::max(0.0f, y2 - y1);
        float inter = w * h;
        float iou   = inter / (box1.w * box1.h + box2.w * box2.h - inter);
        return iou;
    };

    for (auto it = bboxes.begin(); it != bboxes.end(); ++it) {
        if (it->score == 0) continue;
        for (auto it2 = std::next(it); it2 != bboxes.end(); ++it2) {
            if (it2->score == 0) continue;
            if (multi_target && it->target != it2->target) continue;
            const auto iou = compute_iou(*it, *it2);
            if (iou > threshold_iou) {
                if (soft_nms) {
                    it2->score = it2->score * (1 - iou);
                    if (it2->score < threshold_score) it2->score = 0;
                } else {
                    it2->score = 0;
                }
            }
        }
    }

    if constexpr (std::is_same_v<Container, std::forward_list<T>>) {
        bboxes.remove_if([](const auto& box) { return box.score == 0; });
    } else {
        bboxes.erase(std::remove_if(bboxes.begin(), bboxes.end(), [](const auto& box) { return box.score == 0; }),
                     bboxes.end());
    }
}

template void nms(std::forward_list<ma_bbox_ext_t>&, float, float, bool, bool);
template void nms(std::vector<ma_bbox_t>&, float, float, bool, bool);

}  // namespace ma::utils