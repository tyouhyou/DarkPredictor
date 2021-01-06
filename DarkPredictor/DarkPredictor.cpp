#include "DarkPredictor.hpp"
#include "cxxlog.hpp"
#include "stopwatch.hpp"

using namespace zb;

DarkPredictor::DarkPredictor()
	:net{nullptr}
{
	// DO NOTHING
}

void DarkPredictor::SetLog(const std::string& log_file)
{
	SET_LOG_FILE(log_file);
}

void DarkPredictor::Load(const std::string &config_file, const std::string &weights_file)
{
	StopWatch sw;
	sw.start();
	net = load_network_custom(const_cast<char*>(config_file.c_str()), const_cast<char*>(weights_file.c_str()), 1, 1);
	if (nullptr == net)
	{
		auto msg = "Failed to load network.";
		EL << msg;
		throw std::exception(msg);
	}
	fuse_conv_batchnorm(*net);
	calculate_binary_weights(*net);
	IL << "Loading network elapsed: " << sw.elaspsed() << " micro seconds";
}

DarkPredictor::~DarkPredictor()
{
	Destroy();
}

void DarkPredictor::Destroy()
{
	if (nullptr != net)
	{
		free_network_ptr(net);
		net = nullptr;
	}
}

std::vector<predict_result> DarkPredictor::Predict(const char* image_data, const int image_width, const int image_height, const int channels)
{
	image img = make_image(image_width, image_height, channels);
	copy_image_from_bytes(img, const_cast<char*>(image_data));
	auto ret = Predict(img);
	free_image(img);
	return ret;
}

std::vector<predict_result> DarkPredictor::Predict(const std::string& image_file)
{
	image img = load_image_color(const_cast<char*>(image_file.c_str()), 0, 0);
	auto ret = Predict(img);
	free_image(img);
	return ret;
}

std::vector<predict_result> DarkPredictor::Predict(const image &img)
{
	StopWatch sw;

	image sized;
	if (net->letter_box)
	{
		sized = letterbox_image(img, net->w, net->h);
	}
	else
	{
		sized = resize_image(img, net->w, net->h);
	}

	sw.start();

	network_predict_ptr(net, sized.data);
	IL << "Predicting elapsed: " << sw.wrap() << " micro seconds.";

	// TODO: move the following values to setting ?
	float thresh = 0.5f;
	float hier_thresh = 0.5f;
	float nms = 0.45f;

	int nboxes = 0;
	auto dets = get_network_boxes(net, img.w, img.h, thresh, hier_thresh, 0, 1, &nboxes, net->letter_box);


	layer l = net->layers[net->n - 1];
	int k;
	for (k = net->n - 1; k >= 0; k--) {
		layer lk = net->layers[k];
		if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
			l = lk;
			IL << " Detection layer: " << k << " - type = " << l.type;
			break;
		}
	}

	if (nms)
	{
		if (l.nms_kind == DEFAULT_NMS)
		{
			do_nms_sort(dets, nboxes, l.classes, nms);
		}
		else
		{
			diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
		}
	}

	std::vector<predict_result> results;
	for (int i = 0; i < nboxes; i++)
	{
		int clsid = -1;
		float prob = 0;
		box bbox;
		int pts;
		for (int j = 0; j < l.classes; j++)
		{
			if (dets[i].prob[j] > thresh && dets[i].prob[j] > prob) {
				prob = dets[i].prob[j];
				clsid = j;
				bbox = dets[i].bbox;
				pts = dets[i].points;
			}
		}

		if (clsid >= 0)
		{
			predict_result rst;
			rst.class_id = clsid;
			rst.probability = prob;
			rst.w = bbox.w;
			rst.h = bbox.h;
			// Default is center.
			if (0 == pts || (pts & YOLO_CENTER) == YOLO_CENTER)
			{
				rst.x = bbox.x - bbox.w / 2;
				rst.y = bbox.y - bbox.h / 2;
			}
			else if ((pts & YOLO_LEFT_TOP) == YOLO_LEFT_TOP)
			{
				rst.x = bbox.x;
				rst.y = bbox.y;
			}
			else if ((pts & YOLO_RIGHT_BOTTOM) == YOLO_RIGHT_BOTTOM)
			{
				rst.x = bbox.x - bbox.w;
				rst.y = bbox.y - bbox.h;
			}
			else
			{
				// if no bit set, it's center point
				WL << "Unknown points location.";
			}

			results.push_back(rst);
		}
	}

	IL << "Parsing results elasped: " << sw.wrap() << " micro seconds.";

	return results;
}
