#pragma once

// let's be c-like

#ifndef _DPREDICTOR_H
#define _DPREDICTOR_H

#ifndef EXPORT
#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
#endif

struct predict_result
{
	int class_id;
	float x, y, w, h;		// topleft (x, y), width and height
	float probability;
};

#endif