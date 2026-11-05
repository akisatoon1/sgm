#include <algorithm>
#include <bitset>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// Census変換に使うウィンドウが画像内に収まっているかを判定する。
bool is_window_inside_image(const cv::Mat &image, const cv::Point &pixel_pos)
{
    // TODO: 重複有り
    const int window_height = 7;
    const int window_width = 9;

    const int window_top_left_column = pixel_pos.x - window_width / 2;
    const int window_top_left_row = pixel_pos.y - window_height / 2;
    const int window_bottom_right_column = pixel_pos.x + window_width / 2;
    const int window_bottom_right_row = pixel_pos.y + window_height / 2;

    const bool is_inside = (window_top_left_column >= 0) && (window_top_left_row >= 0) &&
                           (window_bottom_right_column < image.cols) && (window_bottom_right_row < image.rows);
    return is_inside;
}

cv::Mat extract_window(const cv::Mat &image, const cv::Point &pixel_pos)
{
    // 左右画像で共通のウィンドウサイズを定義する。
    // TODO: 重複有り
    const int window_height = 7;
    const int window_width = 9;

    const int window_top_left_column = pixel_pos.x - window_width / 2;
    const int window_top_left_row = pixel_pos.y - window_height / 2;
    const cv::Rect window_rect{window_top_left_column, window_top_left_row, window_width, window_height};
    return image(window_rect);
}

std::bitset<64> convert_census_bits(const cv::Mat &window)
{
    const int center_row = window.rows / 2;
    const int center_col = window.cols / 2;
    const int center_pixel_value = window.at<uchar>(center_row, center_col);

    std::bitset<64> bits{0};
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
            bits.set(bit_index, pixel_value < center_pixel_value);
            ++bit_index;
        }
    }
    return bits;
}

// ピクセルごとのコストを計算する。
// Censusをピクセルのコストとする。
int calc_pixel_cost(const cv::Mat &left_window, const cv::Mat &right_window)
{
    const std::bitset<64> left_image_bits = convert_census_bits(left_window);
    const std::bitset<64> right_image_bits = convert_census_bits(right_window);

    const std::bitset<64> xor_result = left_image_bits ^ right_image_bits;
    const int hamming_distance = static_cast<int>(xor_result.count());
    return hamming_distance;
}

// 視差マップを計算する。
// 左右画像のサイズが同じであることを前提にしている。
// 0の要素を持つ画素は視差が不明であることを意味する。
// TODO: 関数の中身が長いのとネストが深くて、コードが見づらい。
cv::Mat_<int> calc_disparity_map(const cv::Mat &left_gray_image, const cv::Mat &right_gray_image)
{
    const int kMinDisparity = 1;  // 最小視差
    const int kMaxDisparity = 63; // 最大視差
    cv::Mat_<int> disparity_map(left_gray_image.rows, left_gray_image.cols);

    for (int row = 0; row < left_gray_image.rows; ++row)
    {
        for (int column = 0; column < left_gray_image.cols; ++column)
        {
            // それぞれの画素にたいして、最もふさわしい視差を求めるため。
            // 視差を変化させて、最もコストが小さい視差を選択する。
            // 各視差に対するコストを格納する配列を用意する。
            const int pixel_costs_length = kMaxDisparity - kMinDisparity + 1;
            std::vector<int> pixel_costs(pixel_costs_length);

            if (not is_window_inside_image(left_gray_image, cv::Point{column, row}))
            {
                // 左画像でウィンドウが画像をはみ出しているとき、その画素はコスト不明として扱う。
                // コスト不明なので視差不明として扱う。
                disparity_map.at<int>(row, column) = 0;
                continue;
            }
            const cv::Mat left_image_window = extract_window(left_gray_image, cv::Point{column, row});

            for (int disparity = kMinDisparity; disparity <= kMaxDisparity; ++disparity)
            {
                // kMinDisparityが0ではないため、配列のインデックスを計算する必要がある。
                const int index = disparity - kMinDisparity;

                const int right_column = column - disparity;
                if (not is_window_inside_image(right_gray_image, cv::Point{right_column, row}))
                {
                    // 右画像のウィンドウを計算して、右画像のウィンドウが画像をはみ出しているときは、無視するためにコスト最大値として扱うようにする。
                    pixel_costs.at(index) = std::numeric_limits<int>::max();
                    continue;
                }
                const cv::Mat right_image_window = extract_window(right_gray_image, cv::Point{right_column, row});

                pixel_costs.at(index) = calc_pixel_cost(left_image_window, right_image_window);
            }

            const auto min_element_iterator = std::min_element(pixel_costs.begin(), pixel_costs.end());
            if (*min_element_iterator == std::numeric_limits<int>::max())
            {
                // 全てが最大コストの場合は視差不明なので、視差を0に設定する。
                disparity_map.at<int>(row, column) = 0;
                continue;
            }
            const int min_cost_disparity = min_element_iterator - pixel_costs.begin() + kMinDisparity;
            disparity_map.at<int>(row, column) = min_cost_disparity;
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
