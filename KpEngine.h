/*
Derue François-Xavier
*/

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>

/*
-------------------------------------------------------
KpEngine allows to extract and describe different
types of features provided by openCV 3 with
a simple framework. Different options must be specified
in #define macro.
ex : 
KpEngine kpEngine;
// if same extractor and descriptor
kpEngine.extrAndDescrKp(im);

// if different extractor and descriptor
kpEngine.extractKp(im); 
kpEngine.describeKp(im);

// if need to download descriptors and keypoints from gpu if GPU used 
kpEngine1.getDescFromGpu();
kpEngine.getKpFromGpu();
-------------------------------------------------------
Features that can be used
- D : Descriptor
- E : Extractor
xfeatures2d::SIFT : D+E
xfeatures2d::SURF : D+E
xfeatures2d::DAISY : D
xfeatures2d::FREAK : D
xfeatures2d::LATCH : D
BRISK : D+E
ORB : D+E
KAZE : D+E
AKAZE : D+E
------ GPU Implementation -------
cuda::SURF_CUDA : D+E
cuda::ORB : D+E but D alone is not working !
cuda::FastFeatureDetector : E
*/

#define KP_EXTRACTOR xfeatures2d::SIFT
#define KP_DESCRIPTOR xfeatures2d::SIFT
#define KP_GPU 0 // to activate if use Gpu Feature
#define SURF_GPU_EXTRACTOR 0 // flag needed for SURF_GPU 
#define SURF_GPU_DESCRIPTOR 0 // flag needed for SURF_GPU 

using namespace std;
using namespace cv;

class KpEngine
{
public:
#if KP_GPU
	cuda::GpuMat im_gpu;
	cuda::GpuMat d_kp_gpu;
	cuda::GpuMat v_kp_gpu;
#endif
#if SURF_GPU_EXTRACTOR
	KP_EXTRACTOR kpExtractor;
#else
	Ptr<KP_EXTRACTOR> kpExtractor;
#endif
#if SURF_GPU_DESCRIPTOR
	KP_DESCRIPTOR kpDescriptor;
#else
	Ptr<KP_DESCRIPTOR> kpDescriptor;
#endif
	Mat d_kp;
	vector<KeyPoint> v_kp;

public:

	KpEngine(){
#if !SURF_GPU_EXTRACTOR
		kpExtractor = KP_EXTRACTOR::create();
#endif
#if !SURF_GPU_DESCRIPTOR
		kpDescriptor = KP_DESCRIPTOR::create();
#endif
	}
	~KpEngine(){}

	void extrAndDescrKp(Mat& im);
	void extractKp(Mat& im);
	void describeKp(Mat& im);
	void getKpFromGpu();
	void getDescFromGpu();
};


void KpEngine::extractKp(Mat& im)
{
#if KP_GPU
	Mat imGray;
	cvtColor(im, imGray, CV_BGR2GRAY);
	im_gpu.upload(imGray);
	try{
#if SURF_GPU_EXTRACTOR
		kpExtractor(im_gpu, cuda::GpuMat(), v_kp_gpu);
#else
		kpExtractor->detectAsync(im_gpu, v_kp_gpu);
#endif
	}
	catch (cv::Exception e){
		cerr << "this feature can not extract" << endl;
	}
#else
	try{
		kpExtractor->detect(im, v_kp);
	}
	catch (cv::Exception e){
		cerr << "this feature can not extract" << endl;
	}
#endif
}

void KpEngine::describeKp(Mat& im)
{
#if KP_GPU
	try{
#if SURF_GPU_DESCRIPTOR
		kpDescriptor(im_gpu, cuda::GpuMat(), v_kp_gpu, d_kp_gpu, true);
#else
		kpDescriptor->computeAsync(im_gpu, v_kp_gpu, d_kp_gpu);
#endif
	}
	catch (cv::Exception e){
		cerr << "!!!! this feature can not describe" << endl;
	}
#else
	try{
		kpDescriptor->compute(im, v_kp, d_kp);
	}
	catch (cv::Exception e){
		cerr << "this feature can not describe" << endl;
	}
#endif
}

void KpEngine::extrAndDescrKp(Mat& im)
{
#if KP_GPU
	try{
		CV_Assert(im.channels() == 3);
		Mat imGray;
		cvtColor(im, imGray, CV_BGR2GRAY);
		im_gpu.upload(imGray);
#if (SURF_GPU_DESCRIPTOR & SURF_GPU_EXTRACTOR)
		kpExtractor(im_gpu, cuda::GpuMat(), v_kp_gpu, d_kp_gpu);
#else
		kpExtractor->detectAndComputeAsync(im_gpu, cuda::GpuMat(), v_kp_gpu, d_kp_gpu);
#endif
	}
	catch (cv::Exception e){
		extractKp(im);
		describeKp(im);
	}
#else

	try{
		kpExtractor->detectAndCompute(im, Mat(), v_kp, d_kp);
	}
	catch (cv::Exception e){
		extractKp(im);
		describeKp(im);
	}
#endif
}
void KpEngine::getKpFromGpu()
{
#if SURF_GPU_EXTRACTOR
	kpExtractor.downloadKeypoints(v_kp_gpu, v_kp);
#elif KP_GPU
	kpExtractor->convert(v_kp_gpu, v_kp);
#endif
}
void KpEngine::getDescFromGpu()
{
#if KP_GPU
	d_kp_gpu.download(d_kp);
#endif
}