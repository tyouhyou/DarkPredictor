#include "opencv2/opencv.hpp"
#include "cxxlog.hpp"
#include "dpredictor.h"

using namespace zb;

int main()
{
    DarkPredictor predictor;
    predictor.Load("cfg\\yolov4.cfg", "weights\\yolov4.weights");

    auto results = predictor.Predict("test_data\\dog.jpg");
    for (int i = 0; i < results.size(); i++)
    {
        DE << "class id: " << results[i].class_id << ". probability: " << results[i].probability;
    }

    DE << "////////////////////////";

    auto im = cv::imread("test_data\\horses.jpg");
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    results = predictor.Predict((char*)im.data, im.cols, im.rows, im.channels());
    for (int i = 0; i < results.size(); i++)
    {
        DE << "class id: " << results[i].class_id << ". probability: " << results[i].probability;
    }

    DE << "////////////////////////";

    im = cv::imread("test_data\\person.jpg");
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    results = predictor.Predict((char*)im.data, im.cols, im.rows, im.channels());
    for (int i = 0; i < results.size(); i++)
    {
        DE << "class id: " << results[i].class_id << ". probability: " << results[i].probability;
    }

    predictor.Destroy();
}
