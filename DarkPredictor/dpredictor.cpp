#include "dpredictor.h"
#include "DarkPredictor.hpp"

typedef void (*predict_result_handler)(predict_result*, int);

#ifdef __cplusplus
extern "C" {
#endif

EXPORT void* create_predictor()
{
	return new zb::DarkPredictor();
}

EXPORT void load(void* predictor, char* config_file, char* weights_file)
{
	if (nullptr != predictor)
	{
		static_cast<zb::DarkPredictor*>(predictor)->Load(config_file, weights_file);
	}
}

EXPORT void set_log(void* predictor, char* log_file)
{
	if (nullptr != predictor)
	{
		static_cast<zb::DarkPredictor*>(predictor)->SetLog(log_file);
	}
}

EXPORT void destroy_predictor(void* predictor)
{
	if (nullptr != predictor)
	{
		static_cast<zb::DarkPredictor *>(predictor)->Destroy();
		delete predictor;
		predictor = nullptr;
	}
}

EXPORT void predict_image_file(void *predictor, char* image_file, predict_result_handler result_handler)
{
	auto rsts = static_cast<zb::DarkPredictor*>(predictor)->Predict(image_file);
	result_handler(rsts.data(), (int)rsts.size());
}

EXPORT void predict_image(void* predictor, char* image_data, int image_width, int image_height, int channels, predict_result_handler result_handler)
{
	auto rsts = static_cast<zb::DarkPredictor*>(predictor)->Predict(image_data, image_width, image_height, channels);
	result_handler(rsts.data(), (int)rsts.size());
}

#ifdef __cplusplus
}
#endif