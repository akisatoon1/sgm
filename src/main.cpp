#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/opencv.hpp>

cv::Mat calc_disparity_map(const cv::Mat &left_image, const cv::Mat &right_image)
{
    // ここに視差マップ計算のコードを追加する
    return cv::Mat();
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

    const cv::Mat left_image = cv::imread(left_image_path);
    const cv::Mat right_image = cv::imread(right_image_path);

    if (left_image.empty())
    {
        const std::string error_message = "画像 " + left_image_path + " の読み込みに失敗しました";
        throw std::runtime_error(error_message);
    }
    if (right_image.empty())
    {
        const std::string error_message = "画像 " + right_image_path + " の読み込みに失敗しました";
        throw std::runtime_error(error_message);
    }

    const cv::Mat disparity_map = calc_disparity_map(left_image, right_image);
    // 視差マップの可視化を行うかも

    return 0;
}
