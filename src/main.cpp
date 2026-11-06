#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// 大きすぎるとオーバーフローする。
// Census変換だと最大でも62コストなので、500くらいにしておけば十分。
const int kCostInf = 500;

const int kMinDisparity = 1;  // 最小視差
const int kMaxDisparity = 63; // 最大視差

const int kWindowHeight = 7; // ウィンドウの高さ
const int kWindowWidth = 9;  // ウィンドウの幅

// Census変換に使うウィンドウが画像内に収まっているかを判定する。
bool is_window_inside_image(const cv::Mat &image, const cv::Point &pixel_pos)
{
    const int window_top_left_column = pixel_pos.x - kWindowWidth / 2;
    const int window_top_left_row = pixel_pos.y - kWindowHeight / 2;
    const int window_bottom_right_column = pixel_pos.x + kWindowWidth / 2;
    const int window_bottom_right_row = pixel_pos.y + kWindowHeight / 2;

    const bool is_inside = (window_top_left_column >= 0) && (window_top_left_row >= 0) &&
                           (window_bottom_right_column < image.cols) && (window_bottom_right_row < image.rows);
    return is_inside;
}

cv::Mat extract_window(const cv::Mat &image, const cv::Point &pixel_pos)
{
    const int window_top_left_column = pixel_pos.x - kWindowWidth / 2;
    const int window_top_left_row = pixel_pos.y - kWindowHeight / 2;
    const cv::Rect window_rect{window_top_left_column, window_top_left_row, kWindowWidth, kWindowHeight};
    return image(window_rect);
}

std::uint64_t convert_census_bits(const cv::Mat &window)
{
    const int center_row = window.rows / 2;
    const int center_col = window.cols / 2;
    const int center_pixel_value = window.at<uchar>(center_row, center_col);

    std::uint64_t bits = 0;
    int bit_index = 0;
    for (int row = 0; row < window.rows; ++row)
    {
        for (int col = 0; col < window.cols; ++col)
        {
            if (row == center_row && col == center_col)
            {
                // 中心ピクセルはスキップする。
                continue;
            }

            const int pixel_value = window.at<uchar>(row, col);
            if (pixel_value < center_pixel_value)
            {
                bits |= std::uint64_t{1} << bit_index;
            }
            ++bit_index;
        }
    }
    return bits;
}

// census_mapの添字を計算する。
std::size_t calc_census_map_index(const cv::Mat &gray_image, const cv::Point &pixel_pos)
{
    return static_cast<std::size_t>(pixel_pos.y) * gray_image.cols + pixel_pos.x;
}

// 画像全体のCensus変換の結果を計算する。
// cv::Matはstd::uint64_tを要素にできないので、std::vectorに保存する。
// ウィンドウが画像をはみ出している画素の値は使わないこと。is_window_inside_imageで判定できる。
std::vector<std::uint64_t> calc_census_map(const cv::Mat &gray_image)
{
    std::vector<std::uint64_t> census_map(static_cast<std::size_t>(gray_image.rows) * gray_image.cols, 0);

    for (int row = 0; row < gray_image.rows; ++row)
    {
        for (int column = 0; column < gray_image.cols; ++column)
        {
            const cv::Point pixel_pos{column, row};
            if (not is_window_inside_image(gray_image, pixel_pos))
            {
                continue;
            }

            const cv::Mat window = extract_window(gray_image, pixel_pos);
            census_map[calc_census_map_index(gray_image, pixel_pos)] = convert_census_bits(window);
        }
    }

    return census_map;
}

// コストを計算するため。
int calc_hamming_distance(std::uint64_t left_census_bits, std::uint64_t right_census_bits)
{
    const std::bitset<64> xor_result{left_census_bits ^ right_census_bits};
    const int hamming_distance = static_cast<int>(xor_result.count());
    return hamming_distance;
}

// TODO: ネストが深くて分かりづらいし、行数も長い。
cv::Mat_<int> calc_local_cost_map(const cv::Mat &left_gray_image, const cv::Mat &right_gray_image)
{
    const int disparity_range = kMaxDisparity - kMinDisparity + 1;
    const int sizes[] = {left_gray_image.rows, left_gray_image.cols, disparity_range};
    cv::Mat_<int> local_cost_map(3, sizes);

    const std::vector<std::uint64_t> left_census_map = calc_census_map(left_gray_image);
    const std::vector<std::uint64_t> right_census_map = calc_census_map(right_gray_image);

    for (int row = 0; row < left_gray_image.rows; ++row)
    {
        for (int column = 0; column < left_gray_image.cols; ++column)
        {
            const cv::Point left_pixel_pos{column, row};
            if (not is_window_inside_image(left_gray_image, left_pixel_pos))
            {
                // 左画像でウィンドウが画像をはみ出しているとき、その画素はコスト不明として扱う。
                for (int disparity = kMinDisparity; disparity <= kMaxDisparity; ++disparity)
                {
                    const int disparity_index = disparity - kMinDisparity;
                    local_cost_map.at<int>(row, column, disparity_index) = kCostInf;
                }
                continue;
            }

            for (int disparity = kMinDisparity; disparity <= kMaxDisparity; ++disparity)
            {
                const int disparity_index = disparity - kMinDisparity;

                const cv::Point right_pixel_pos{column - disparity, row};
                if (not is_window_inside_image(right_gray_image, right_pixel_pos))
                {
                    // 右画像でウィンドウが画像をはみ出しているときは、その画素はコスト不明として扱う。
                    local_cost_map.at<int>(row, column, disparity_index) = kCostInf;
                    continue;
                }

                const std::uint64_t left_census_bits = left_census_map[calc_census_map_index(left_gray_image, left_pixel_pos)];
                const std::uint64_t right_census_bits = right_census_map[calc_census_map_index(right_gray_image, right_pixel_pos)];
                const int local_cost = calc_hamming_distance(left_census_bits, right_census_bits);
                local_cost_map.at<int>(row, column, disparity_index) = local_cost;
            }
        }
    }
    return local_cost_map;
}

int calc_penalty(const int current_disparity, const int previous_disparity)
{
    // ペナルティの値はlibSGMの実装を参考にしている。
    // AI調査により、ペナルティのデフォルト値が10, 120らしい。
    if (current_disparity == previous_disparity)
    {
        return 0;
    }
    else if (std::abs(current_disparity - previous_disparity) == 1)
    {
        return 10;
    }
    else
    {
        return 120;
    }
}

// TODO: 左から右のみをまずは実装する。
// TODO: 読みづらい。
cv::Mat_<int> calc_line_cost_map(const cv::Mat_<int> local_cost_map)
{
    cv::Mat_<int> line_cost_map(3, local_cost_map.size);

    for (int row = 0; row < local_cost_map.size[0]; ++row)
    {
        for (int column = 0; column < local_cost_map.size[1]; ++column)
        {
            if (column == 0)
            {
                // 最初の列は、コストマップの値をそのままコピーする。
                for (int disparity_index = 0; disparity_index < local_cost_map.size[2]; ++disparity_index)
                {
                    line_cost_map.at<int>(row, column, disparity_index) = local_cost_map.at<int>(row, column, disparity_index);
                }
                continue;
            }

            for (int disparity_index = 0; disparity_index < local_cost_map.size[2]; ++disparity_index)
            {
                int min_cost = std::numeric_limits<int>::max();
                for (int previous_disparity_index = 0; previous_disparity_index < local_cost_map.size[2]; ++previous_disparity_index)
                {
                    const int previous_line_cost = line_cost_map.at<int>(row, column - 1, previous_disparity_index);
                    const int penalty = calc_penalty(disparity_index, previous_disparity_index);
                    min_cost = std::min(min_cost, previous_line_cost + penalty);
                }

                const int local_cost = local_cost_map.at<int>(row, column, disparity_index);
                line_cost_map.at<int>(row, column, disparity_index) = min_cost + local_cost;
            }
        }
    }
    return line_cost_map;
}

// 視差マップを計算する。
// 左右画像のサイズが同じであることを前提にしている。
// 0の要素を持つ画素は視差が不明であることを意味する。
// TODO: 関数の中身が長いのとネストが深くて、コードが見づらい。
cv::Mat_<int> calc_disparity_map(const cv::Mat &left_gray_image, const cv::Mat &right_gray_image)
{
    cv::Mat_<int> disparity_map(left_gray_image.rows, left_gray_image.cols);

    const cv::Mat_<int> cost_map = calc_local_cost_map(left_gray_image, right_gray_image);
    // TODO: ラインコストマップの計算は、左から右だけでなく、右から左、上から下、下から上の4方向を考慮する必要がある。
    const cv::Mat_<int> line_cost_map = calc_line_cost_map(cost_map);

    for (int row = 0; row < left_gray_image.rows; ++row)
    {
        for (int column = 0; column < left_gray_image.cols; ++column)
        {
            const bool is_left_pixel_inside_image = is_window_inside_image(left_gray_image, cv::Point{column, row});
            const bool is_right_pixel_inside_image = is_window_inside_image(right_gray_image, cv::Point{column - kMinDisparity, row});
            if (not is_left_pixel_inside_image || not is_right_pixel_inside_image)
            {
                // ウィンドウが画像をはみ出しているとき、その画素は視差不明として扱う。
                disparity_map.at<int>(row, column) = 0;
                continue;
            }

            int min_cost_disparity_index = 0;

            for (int disparity_index = 1; disparity_index < line_cost_map.size[2]; ++disparity_index)
            {
                const int current_cost = line_cost_map.at<int>(row, column, disparity_index);
                const int min_cost = line_cost_map.at<int>(row, column, min_cost_disparity_index);
                if (current_cost < min_cost)
                {
                    min_cost_disparity_index = disparity_index;
                }
            }
            disparity_map.at<int>(row, column) = min_cost_disparity_index + kMinDisparity;
        }
    }
    return disparity_map;
}

// 視差マップをグレースケール画像に変換する。
// 視差dに4を掛けた値をグレースケール画像の輝度値として扱う。
cv::Mat convert_disparity_map_to_gray_image(const cv::Mat_<int> &disparity_map)
{
    cv::Mat gray_image(disparity_map.rows, disparity_map.cols, CV_8UC1);

    for (int row = 0; row < disparity_map.rows; ++row)
    {
        for (int column = 0; column < disparity_map.cols; ++column)
        {
            const int disparity = disparity_map.at<int>(row, column);
            const int gray_value = disparity * 4;
            gray_image.at<uchar>(row, column) = static_cast<uchar>(gray_value);
        }
    }

    return gray_image;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        const std::string error_message = "引数の数が間違っています。使い方: ./main <左画像> <右画像>";
        throw std::runtime_error(error_message);
    }

    const std::string left_image_path = argv[1];
    const std::string right_image_path = argv[2];

    const cv::Mat left_gray_image = cv::imread(left_image_path, cv::IMREAD_GRAYSCALE);
    const cv::Mat right_gray_image = cv::imread(right_image_path, cv::IMREAD_GRAYSCALE);

    if (left_gray_image.empty())
    {
        const std::string error_message = "画像 " + left_image_path + " の読み込みに失敗しました";
        throw std::runtime_error(error_message);
    }
    if (right_gray_image.empty())
    {
        const std::string error_message = "画像 " + right_image_path + " の読み込みに失敗しました";
        throw std::runtime_error(error_message);
    }

    const cv::Mat_<int> disparity_map = calc_disparity_map(left_gray_image, right_gray_image);
    const cv::Mat gray_image = convert_disparity_map_to_gray_image(disparity_map);
    cv::imshow("Disparity Map", gray_image);
    cv::waitKey(0);

    return 0;
}
