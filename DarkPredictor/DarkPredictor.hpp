#pragma once

#include <vector>
#include <string>
#include "darknet.h"
#include "dpredictor.h"

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
