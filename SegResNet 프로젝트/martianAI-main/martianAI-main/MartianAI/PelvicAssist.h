#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <vector>
#include <windows.h>
#include "ControlSharedMemory.h"
#include <c10/cuda/CUDACachingAllocator.h>

using namespace torch::indexing;
using namespace std;

// DEV for postprocessing 
#if !defined (NDEBUG) // DEBUG MODE
#define DEV_BYPASS_INFERENCE 0 // 0 do inference , 1 bypass
#define DEV_INFERENCE_STEP_BIN_OUT 1
#define DEV_DEBUG_MARK_VALUE 1 // mark point as this value
#define DEV_DEBUG_STANDARD_PLANE_BIN_OUT 1 // to store 2d Standard Plane Image
#define DEV_SKIP_PREPROCESS 0   // 0 do preprocess , 1 skip 
#define DEV_FORCE_CPU_CC3D  0   // 0 conditional cc3d , 1 cpu cc3d
#else
#define DEV_DEBUG_STANDARD_PLANE_BIN_OUT 0 // 
#endif
//
#define BG_KEY           0
#define PUBIS_KEY        1
#define LEVATORANI_KEY   2
#define ANUS_KEY         3
#define VAGINA_KEY       4
#define URETHRA_KEY      5
#define PELVICBRIM_KEY   6
#define BLADDER_KEY      7

#define PELVICBRIM_VOXEL_MIN 1000

#define PELVICASSIST_SEG_IN_VOLUME_SIZE                256
#define PELVICASSIST_SEG_OUT_VOLUME_SIZE               128
#define PELVIC2DSEG_IMAGE_SIZE						   256

#define SUPPORT_X_ROTATION 1

class PelvicAssist
{
public:
	PelvicAssist(int isCPUOnly, std::string device_string);
	~PelvicAssist();

	int PelvicAssist::segResnet(
		torch::jit::script::Module module,
		at::Tensor& input,
		at::Tensor& output,
		float* SegMeanOut,
		float* pcaMeanOut,
		int labelNum,
		float* labelThreshold,
		int* processOnLabel,
		int volumeDim,
		int gauissianFilterMode,
		int morphProcessMode,
		int* errorCode,
		int* returnValue,
		unsigned char* outputBuffer
	);
	int PelvicAssist::Measure2DSagmentation(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, int width, int height, double rotate);

private:
#define LABELNUM 6
	const int labelNum = LABELNUM;
    // 16345
	char labelName[LABELNUM][11] = { "BG", "PUBIS" , "PELVICBRIM" , "ANUS", "VARGINA", "URETHRA" };
	int labelOffsetTable[LABELNUM] = { BG_KEY, PUBIS_KEY, PELVICBRIM_KEY, ANUS_KEY, VAGINA_KEY, URETHRA_KEY };
	int labelMaxCountTable[LABELNUM] = { 0, 2, 1, 1, 1, 1 };
	float labelSizeTable[LABELNUM] = { 0, 50, 50, 50, 50, 70 };
	int labelMinSize = 30;
	int crop_roi[6] = { 16, 240, 32, 224, 64, 192 };
	int centroid_roi[3][2] = { {40,PELVICASSIST_SEG_OUT_VOLUME_SIZE - 40 - 1},
								{30,PELVICASSIST_SEG_OUT_VOLUME_SIZE - 30 - 1},
								{40,PELVICASSIST_SEG_OUT_VOLUME_SIZE - 40 - 1} 
	};
	int lengthMin = 10;

	int processOnLabel[LABELNUM] = { 0, 0, 0, 0, 0, 0 };
	int interpolationOrder[LABELNUM] = { 0, 0, 0, 0, 0, 0 };
	float labelThreshold[LABELNUM] = { 0.5,0.5,0.5,0.5,0.5,0.5 };
	float scale = 0.5;
	int m_isCPUOnly = 1;
	std::string m_device_string = "cpu";

	struct ComponentInfo { UINT32 volume; UINT32 ccLabel; float centroid[3]; };
	struct Point {
		float x; float y; float z;
		bool operator<(const Point& other) {
			if (x == other.x) {
				if (y == other.y) {
					return z < other.z;
				}
				return y < other.y;
			}
			return x < other.x;
		}
		bool operator!=(const Point& other) {
			return (x != other.x || y != other.y || z != other.z);
		}
		bool operator==(const Point& other) {
			return (x == other.x && y == other.y && z == other.z);
		}
	};
	typedef vector<Point> PointVector;
#if defined(DEV_INFERENCE_STEP_BIN_OUT) && (DEV_INFERENCE_STEP_BIN_OUT)
	PointVector debugPoints;
#endif
	struct less_than_size
	{
		inline bool operator() (const at::Tensor& a, const at::Tensor& b)
		{
			return a.size(0) > b.size(0);
		}
	};
	struct CurveItem { float curveDiff; PointVector curve; PointVector mucosaCurve; Point ios; Point eos; };
	typedef vector<CurveItem> CurveItemVector;
	typedef vector<at::Tensor> TensorVector;
	typedef map< int, TensorVector> GroupMap;

	
	float PelvicAssist::distance3d(float x1, float y1, float z1, float x2, float y2, float z2);
	float PelvicAssist::distance2d(float x1, float y1, float x2, float y2);
	int PelvicAssist::Preprocess(at::Tensor& input, at::Tensor& output, GroupMap& Groups, int* errorCode);
	int PelvicAssist::cc3d(TensorVector& outlist, at::Tensor& inTensor);
    int PelvicAssist::GetGroup(GroupMap& Groups,at::Tensor input, int labelNum, int* labelOffsetTable);
	int PelvicAssist::RemoveSmall(GroupMap& Groups, at::Tensor& output, int labelNum, int maxCount, int minSize);
	int PelvicAssist::Tensor2PointVector(at::Tensor& t, PointVector& v);
	int PelvicAssist::GetNormalVector(at::Tensor planePoints,float* pcaVectorOut);
	int PelvicAssist::getPlanePoints(at::Tensor& object, at::Tensor& planePoint,int* errorCode, int debugout = 0);
	int PelvicAssist::getPathPoint(at::Tensor points1, at::Tensor points2, float angleRangeMin, float angleRangeMax, at::Tensor& out, int mode);
	int PelvicAssist::printTensorInfo(at::Tensor t);
	at::Tensor PelvicAssist::toDevice(const at::Tensor& t);
	at::Tensor PelvicAssist::PointVector2Tensor(PointVector& pointVector);
	vector<double> PelvicAssist::Linspace(double start_in, double end_in, int num_in);
	at::Tensor PelvicAssist::LinspaceT(double start_in, double end_in, int num_in);
	torch::Tensor PelvicAssist::linalg_norm(torch::Tensor tensor, int dim , bool keepdim);
	at::Tensor PelvicAssist::rotate_y(float angle, at::Tensor& point);
	float PelvicAssist::getAngle(at::Tensor point);

	int PelvicAssist::segResnet_sub(
		torch::jit::script::Module module,
		at::Tensor& input,
		at::Tensor& output,
		int* crop_roi,
		float scale,
		int* resize,
		int labelNum,
		float* labelThreshold,
		int* labelOffsetTable,
		int* errorCode
		);

#if defined(DEV_BYPASS_INFERENCE) && (DEV_BYPASS_INFERENCE == 1)
	int PelvicAssist::LoadVolume(const std::string& filepath, at::Tensor& volume);
#endif
#if defined(DEV_INFERENCE_STEP_BIN_OUT) && (DEV_INFERENCE_STEP_BIN_OUT == 1 )	
	void PelvicAssist::SaveVolume(const std::string& filepath, at::Tensor volume , 
		int dimA = PELVICASSIST_SEG_OUT_VOLUME_SIZE, 
		int dimB = PELVICASSIST_SEG_OUT_VOLUME_SIZE, 
		int dimC = PELVICASSIST_SEG_OUT_VOLUME_SIZE);
#endif
#if defined (SUPPORT_X_ROTATION) && ( SUPPORT_X_ROTATION )
	int PelvicAssist::distance_point_to_line(at::Tensor P, at::Tensor& out, at::Tensor top, at::Tensor bottom);
	int PelvicAssist::getLeftRightPoint(at::Tensor object, int z, at::Tensor top, at::Tensor bottom, at::Tensor& leftrightOut, int maskOffset=1);
#endif
#if defined(DEV_DEBUG_STANDARD_PLANE_BIN_OUT) && (DEV_DEBUG_STANDARD_PLANE_BIN_OUT == 1)
	at::Tensor PelvicAssist::GetImageForStandardPlane(at::Tensor& input);
	void PelvicAssist::GetStandardPlane(at::Tensor image, at::Tensor centroid,float* pcaVectorOut);
	void PelvicAssist::Save2dVolume(const std::string& filepath, at::Tensor& volume);
#endif
};

