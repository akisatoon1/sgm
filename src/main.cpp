#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

const int kMinDisparity = 1;  // 最小視差
const int kMaxDisparity = 63; // 最大視差

// ピクセルごとのコストを計算する。
// BGRの各チャンネルの差の絶対値の合計をそのピクセルのコストとする。
int calc_pixel_cost(const cv::Vec3b &left_pixel, const cv::Vec3b &right_pixel)
{
    const int abs_diff_B = std::abs(static_cast<int>(left_pixel[0]) - static_cast<int>(right_pixel[0]));
    const int abs_diff_G = std::abs(static_cast<int>(left_pixel[1]) - static_cast<int>(right_pixel[1]));
    const int abs_diff_R = std::abs(static_cast<int>(left_pixel[2]) - static_cast<int>(right_pixel[2]));
    return abs_diff_B + abs_diff_G + abs_diff_R;
}

// 視差マップを計算する。
// 左右画像のサイズが同じであることを前提にしている。
// 0の要素を持つ画素は視差が不明であることを意味する。
// TODO: 関数の中身が長いのとネストが深くて、コードが見づらい。
cv::Mat_<int> calc_disparity_map(const cv::Mat &left_color_image, const cv::Mat &right_color_image)
{
    cv::Mat_<int> disparity_map(left_color_image.rows, left_color_image.cols);

    for (int row = 0; row < left_color_image.rows; ++row)
    {
        for (int column = 0; column < left_color_image.cols; ++column)
        {
            // それぞれの画素にたいして、最もふさわしい視差を求めるため。
            // 視差を変化させて、最もコストが小さい視差を選択する。
            // 各視差に対するコストを格納する配列を用意する。
            const int pixel_costs_length = kMaxDisparity - kMinDisparity + 1;
            std::vector<int> pixel_costs(pixel_costs_length);

            for (int disparity = kMinDisparity; disparity <= kMaxDisparity; ++disparity)
            {
                // kMinDisparityが0ではないため、配列のインデックスを計算する必要がある。
                const int index = disparity - kMinDisparity;

                const int right_column = column - disparity;
                if (right_column < 0)
                {
                    // 右画像の列が負の値になる場合はコスト不明なので、
                    // ピクセルコストを最大値に設定して、コストの最小値を選択する際に選ばれないようにする。
                    pixel_costs.at(index) = std::numeric_limits<int>::max();
                    continue;
                }
                const cv::Vec3b left_pixel = left_color_image.at<cv::Vec3b>(row, column);
                const cv::Vec3b right_pixel = right_color_image.at<cv::Vec3b>(row, right_column);

                const int pixel_cost = calc_pixel_cost(left_pixel, right_pixel);
                pixel_costs.at(index) = pixel_cost;
            }

            const auto min_element_iterator = std::min_element(pixel_costs.begin(), pixel_costs.end());
            if (*min_element_iterator == std::numeric_limits<int>::max())
            {
                // 右画像の列が負の値になる場合はコスト不明なので、視差を0に設定する。
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

    const cv::Mat left_color_image = cv::imread(left_image_path);
    const cv::Mat right_color_image = cv::imread(right_image_path);

    if (left_color_image.empty())
    {
        const std::string error_message = "画像 " + left_image_path + " の読み込みに失敗しました";
        throw std::runtime_error(error_message);
    }
    if (right_color_image.empty())
    {
        const std::string error_message = "画像 " + right_image_path + " の読み込みに失敗しました";
        throw std::runtime_error(error_message);
    }

    const cv::Mat_<int> disparity_map = calc_disparity_map(left_color_image, right_color_image);
    const cv::Mat gray_image = convert_disparity_map_to_gray_image(disparity_map);
    cv::imshow("Disparity Map", gray_image);
    cv::waitKey(0);

    return 0;
}
