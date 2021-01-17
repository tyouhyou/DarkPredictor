#include "dpredictor.h"

void* create_predictor()
{
	return new zb::DarkPredictor();
}

void load(void* predictor, char* config_file, char* weights_file)
{
	if (nullptr != predictor)
	{
		static_cast<zb::DarkPredictor*>(predictor)->Load(config_file, weights_file);
	}
}

void set_log(void* predictor, char* log_file)
{
	if (nullptr != predictor)
	{
		static_cast<zb::DarkPredictor*>(predictor)->SetLog(log_file);
	}
}

void destroy_predictor(void* predictor)
{
	if (nullptr != predictor)
	{
		static_cast<zb::DarkPredictor *>(predictor)->Destroy();
		delete predictor;
		predictor = nullptr;
	}
}

void* predict_image_file(void *predictor, char* image_file, predict_result_handler result_handler)
{
	auto rsts = static_cast<zb::DarkPredictor*>(predictor)->Predict(image_file);
	return result_handler(rsts.data(), (int)rsts.size());
}

void* predict_image(void* predictor, char* image_data, int image_width, int image_height, int channels, predict_result_handler result_handler)
{
	auto rsts = static_cast<zb::DarkPredictor*>(predictor)->Predict(image_data, image_width, image_height, channels);
	return result_handler(rsts.data(), (int)rsts.size());
}
