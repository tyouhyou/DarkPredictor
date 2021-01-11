#pragma once

#include <vector>
#include <string>
#include "darknet.h"

// let's be c-like, there have compilers not recognizing #pragma once
#ifndef _DPREDICTOR_H
#define _DPREDICTOR_H

#ifndef EXPORT
#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

struct predict_result
{
	int class_id;
	float x, y, w, h;		// topleft (x, y), width and height
	float probability;
};

#endif

namespace zb
{
	class EXPORT DarkPredictor
	{
	public:
		DarkPredictor();

		~DarkPredictor();

		void Destroy();

		void SetLog(const std::string &log_file);

		void Load(const std::string &config_file, const std::string &weights_file);

		/** The image data format should be the same as trained images.
		* When pass image data from cv::Mat, make sure the color is rgb or bgr which should be the same as trained images.
		*/
		std::vector<predict_result> Predict(const char* image_data, const int image_width, const int image_height, const int channels);

		/* rgb image only */
		std::vector<predict_result> Predict(const std::string& image_file);

	private:
		network* net;

		std::vector<predict_result> Predict(const image &img);
	};

}

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*predict_result_handler)(predict_result*, int);

EXPORT void* create_predictor();
EXPORT void load(void* predictor, char* config_file, char* weights_file);
EXPORT void set_log(void* predictor, char* log_file);
EXPORT void destroy_predictor(void* predictor);
EXPORT void predict_image_file(void* predictor, char* image_file, predict_result_handler result_handler);
EXPORT void predict_image(void* predictor, char* image_data, int image_width, int image_height, int channels, predict_result_handler result_handler);

#ifdef __cplusplus
}
#endif

#endif
