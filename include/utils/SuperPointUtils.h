#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <iostream>
#include <opencv2/core/core.hpp>

// L2 Normalization utility function
inline void L2_normalization(const int8_t* input, float scale, int channel, int group, float* output) {
#ifdef __ARM_NEON
    // NEON vectorized L2 normalization with fixed indices
    const size_t blk = 32;
    for (size_t g = 0; g < group; ++g) {
    const int8_t* src = input + g * channel;
    float32x4_t sumv = vdupq_n_f32(0.f);
    
    for (size_t c = 0; c < channel; c += blk) {
        int8x16_t v0 = vld1q_s8(src + c);
        int8x16_t v1 = vld1q_s8(src + c + 16);
        int16x8_t s0 = vmovl_s8(vget_low_s8(v0));
        int16x8_t s1 = vmovl_s8(vget_high_s8(v0));
        int16x8_t s2 = vmovl_s8(vget_low_s8(v1));
        int16x8_t s3 = vmovl_s8(vget_high_s8(v1));
        
        sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))), vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0))));
        sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0))));
        sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))), vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1))));
        sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1))));
        sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s2))), vcvtq_f32_s32(vmovl_s16(vget_low_s16(s2))));
        sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s2))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(s2))));
        sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s3))), vcvtq_f32_s32(vmovl_s16(vget_low_s16(s3))));
        sumv = vmlaq_f32(sumv, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s3))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(s3))));
    }
    
    // Extract the sum using fixed lane indices
    float sum = vgetq_lane_f32(sumv, 0) + vgetq_lane_f32(sumv, 1) + 
                vgetq_lane_f32(sumv, 2) + vgetq_lane_f32(sumv, 3);
    
    float norm = 1.f / std::sqrt(sum) * scale;
    float32x4_t nrm = vdupq_n_f32(norm);
    
    for (size_t c = 0; c < channel; c += 16) {
        int8x16_t v = vld1q_s8(src + c);
        int16x8_t lo = vmovl_s8(vget_low_s8(v));
        int16x8_t hi = vmovl_s8(vget_high_s8(v));
        
        float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), nrm);
        float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), nrm);
        float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), nrm);
        float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), nrm);
        
        vst1q_f32(output + g * channel + c + 0, f0);
        vst1q_f32(output + g * channel + c + 4, f1);
        vst1q_f32(output + g * channel + c + 8, f2);
        vst1q_f32(output + g * channel + c + 12, f3);
    }
    }
#else
    // Scalar L2 normalization
    for (int i = 0; i < group; ++i) {
    float sum = 0.0;
    for (int j = 0; j < channel; ++j) {
        int pos = i * channel + j;
        float temp = input[pos] * scale;
        sum += temp * temp;
    }
    float var = sqrt(sum);
    for (int j = 0; j < channel; ++j) {
        int pos = i * channel + j;
        output[pos] = (input[pos] * scale) / var;
    }
    }
#endif
}

// Bilinear interpolation helper function
inline float bilinear_interpolation(float v00, float v01, float v10, float v11,
                                    int x0, int y0, int x1, int y1,
                                    float x, float y, bool border_check) {
    if (border_check) {
    // Out of bounds check
    if (x0 < 0 || y0 < 0 || x1 < 0 || y1 < 0) {
        return 0;
    }
    }
    
    float dx = (x - x0) / static_cast<float>(x1 - x0);
    float dy = (y - y0) / static_cast<float>(y1 - y0);
    
    float val = (1 - dx) * (1 - dy) * v00 +
                dx * (1 - dy) * v01 +
                (1 - dx) * dy * v10 +
                dx * dy * v11;
    
    return val;
}

// Optimized bilinear sampling for descriptor maps
inline void bilinear_sample(const float* map, size_t h, size_t w, size_t ch,
                            const std::vector<std::pair<float, float>>& pts,
                            std::vector<std::vector<float>>& descs) {
    descs.resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
    int x0 = floor(pts[i].first / 8.f);
    int y0 = floor(pts[i].second / 8.f);
    int x1 = std::min<int>(x0 + 1, w - 1);
    int y1 = std::min<int>(y0 + 1, h - 1);
    float dx = pts[i].first / 8.f - x0;
    float dy = pts[i].second / 8.f - y0;
    float w00 = (1 - dx) * (1 - dy);
    float w01 = dx * (1 - dy);
    float w10 = (1 - dx) * dy;
    float w11 = dx * dy;
    
    descs[i].resize(ch);
    float norm = 0.f;
    for (size_t c = 0; c < ch; ++c) {
        float val = map[c + ch * (y0 * w + x0)] * w00 +
                map[c + ch * (y0 * w + x1)] * w01 +
                map[c + ch * (y1 * w + x0)] * w10 +
                map[c + ch * (y1 * w + x1)] * w11;
        descs[i][c] = val;
        norm += val * val;
    }
    
    // Normalize the descriptor
    norm = 1.f / std::sqrt(norm);
    for (auto& v : descs[i]) v *= norm;
    }
}

// Optimized NMS implementation from the optimized code
inline void nms_fast(const std::vector<int>& xs, const std::vector<int>& ys,
                    const std::vector<float>& score, int w, int h,
                    std::vector<size_t>& keep) {
    const int radius = 4;
    std::vector<int> grid(w * h, 0);
    std::vector<size_t> order(xs.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b){ return score[a] > score[b]; });
    for (auto idx : order) {
    int x = xs[idx], y = ys[idx];
    if (x < radius || x >= w - radius || y < radius || y >= h - radius) continue;
    bool skip = false;
    for (int i = -radius; i <= radius && !skip; ++i)
        for (int j = -radius; j <= radius; ++j)
        if (grid[(y + i) * w + (x + j)] == 1) { skip = true; break; }
    if (!skip) {
        keep.push_back(idx);
        for (int i = -radius; i <= radius; ++i)
        for (int j = -radius; j <= radius; ++j)
            grid[(y + i) * w + (x + j)] = 1;
    }
    }
}

// Helper function to match existing interface with the optimized NMS
inline void nms_fast(const std::vector<int>& xs, const std::vector<int>& ys,
                    const std::vector<float>& score, std::vector<size_t>& keep_inds, int w, int h) {
    nms_fast(xs, ys, score, w, h, keep_inds);
}

inline void nms_mask(std::vector<std::vector<int>>& grid, int x, int y, int dist_thresh) {
    int h = grid.size();
    int w = grid[0].size();
    for (int i = std::max(0, x - dist_thresh); i < std::min(h, x + dist_thresh + 1); ++i) {
        for (int j = std::max(0, y - dist_thresh); j < std::min(w, y + dist_thresh + 1); ++j) {
            grid[i][j] = -1;
        }
    }
    grid[x][y] = 1;
}

inline void nms_old(const std::vector<int>& xs, const std::vector<int>& ys, const std::vector<float>& ptscore,
                std::vector<size_t>& keep_inds, const int inputW, const int inputH) {
    std::vector<std::vector<int>> grid(inputW, std::vector<int>(inputH, 0));
    std::vector<std::pair<float, size_t>> order;

    //Check Here: Higher means more aggressive NMS
    int dist_thresh = 1; // Helitha
    // int dist_thresh = 4; // Xilinx
    for (size_t i = 0; i < ptscore.size(); ++i) {
    order.push_back({ptscore[i], i});
    }
    std::stable_sort(order.begin(), order.end(),
                    [](const std::pair<float, size_t>& ls, const std::pair<float, size_t>& rs) {
                    return ls.first > rs.first;
                    });
    std::vector<size_t> ordered;
    std::transform(order.begin(), order.end(), std::back_inserter(ordered),
              [](auto& km) { return km.second; });

    for (size_t _i = 0; _i < ordered.size(); ++_i) {
    size_t i = ordered[_i];
    int x = xs[i];
    int y = ys[i];
    if (grid[x][y] == 0 && x >= dist_thresh && x < inputW - dist_thresh && y >= dist_thresh &&
        y < inputH - dist_thresh) {
        keep_inds.push_back(i);
        nms_mask(grid, x, y, dist_thresh);
    }
    }
}

inline std::vector<std::vector<float>> grid_sample(const float* desc_map, const std::vector<std::pair<float, float>>& coarse_pts,
                                const size_t channel, const size_t outputH, const size_t outputW) {
    std::vector<std::vector<float>> descs(coarse_pts.size());
    for (size_t i = 0; i < coarse_pts.size(); ++i) {
    float x = (coarse_pts[i].first + 1) / 8 - 0.5;
    float y = (coarse_pts[i].second + 1) / 8 - 0.5;
    int xmin = floor(x);
    int ymin = floor(y);
    int xmax = xmin + 1;
    int ymax = ymin + 1;

    xmin = std::max(0, std::min(xmin, static_cast<int>(outputW) - 1));
    xmax = std::max(0, std::min(xmax, static_cast<int>(outputW) - 1));
    ymin = std::max(0, std::min(ymin, static_cast<int>(outputH) - 1));
    ymax = std::max(0, std::min(ymax, static_cast<int>(outputH) - 1));

    // Bilinear interpolation
    {
        float divisor = 0.0;
        for (size_t j = 0; j < channel; ++j) {
        float value = bilinear_interpolation(
            desc_map[j + (ymin * outputW + xmin) * channel],
            desc_map[j + (ymin * outputW + xmax) * channel],
            desc_map[j + (ymax * outputW + xmin) * channel],
            desc_map[j + (ymax * outputW + xmax) * channel], xmin, ymin, xmax, ymax, x, y, false);
        divisor += value * value;
        descs[i].push_back(value);
        }
        for (size_t j = 0; j < channel; ++j) {
        descs[i][j] /= sqrt(divisor);  // L2 normalize
        }
    }
    }
    return descs;
}