#pragma once

/* gpfBNgwθyΡθ` */
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
using namespace std;
using namespace cv;
string win_src = "src";				// όΝζEBhE
string win_dst = "dst";				// oΝζEBhE(mCYζ)
string win_dst2 = "dst_MRF";	// oΝζEBhE(mCYζMRF)
string win_dst3 = "dst_HMRF";	// oΝζEBhE(mCYζHMRF)
string win_dst4 = "dst_NLM";	// oΝζEBhE(mCYζHMRF)

#define _M_PI 3.1415926535897932384626433


/* όoΝζ */
Mat Image_src;			// όΝβCζ
Mat Image_dst;			// oΝβCζ(mCYζ)
Mat Image_dst_average;	// oΝβCζ(mCY½Οζ)
Mat Image_dst_MRF;		// oΝβCζ(mCYζMRF)
Mat Image_dst_HMRF;		// oΝβCζ(mCYζHMRF)
Mat Image_dst_NLM;		// oΝβCζ(mCYζNonLocalMeans)
Mat Image_dst_NLMdef;	// oΝβCζ(mCYζNonLocalMeans ¦ftHgέθ)

/* oΝqXgOζ */
Mat Image_hist_src;
Mat Image_hist_dst;
Mat Image_hist_dst_average;
Mat Image_hist_dst_MRF;
Mat Image_hist_dst_HMRF;
Mat Image_hist_dst_NLM;

/* θ */
int MAX_INTENSE = 255;	// ΕεFl
int WIDTH;				// όΝζΜ‘isNZj
int HEIGHT;				// όΝζΜcisNZj
int MAX_DATA;			// όΝζΜsNZ
int DO_NUMBER = 10;		// ΐ±ρ (K=0ΜΚΜ½Οπίι)
int IMAGE_NUMBER = 1;	// ζ K (0:Ο¦Δ‘ΐ±)

/* KEXmCYtΑΜp[^ */
static double NoiseMean = 0.0;	// mCYΜ½Ο
static double NoiseSigma = 30;	// mCYΜWΞ·i2ζΕͺUj

/* Φ */
void Input_Image();			// ζΜόΝ
void Output_Image();		// ζΜoΝ
void MSE_PSNR_SSIM(Mat& Original, Mat& Inpaint);			// MSE&PSNR&SSIMΙζιζ]Ώ
void SSIMcalc(double& ssim, Mat& image_1, Mat& image_2);	// SSIMZo

// double^3`lΜMatζπKi»
void doubleMatStandardization(Mat& doubleMatImage) {
	int x, y, c;
	int doubleMatIndex;
	double doubleTMP;
	Vec3f doublePut;
	Mat doubleMatImage_tmp = Mat(doubleMatImage.size(), CV_64FC3);
	if (doubleMatImage.type() != CV_64FC3) {
		cout << "ERROR! CalcAverage : Input type is wrong." << endl;
	}
	else {
#pragma omp parallel for private(x, c, doubleTMP)
		for (y = 0; y < doubleMatImage.rows; y++) {
			for (x = 0; x < doubleMatImage.cols; x++) {
				for (c = 0; c < 3; c++) {
					doubleMatIndex = (y * doubleMatImage.cols + x) * 3 + c;
					doubleTMP = (double)doubleMatImage.data[doubleMatIndex];
					doubleTMP = (double)doubleTMP / (double)MAX_INTENSE;	// 0~1 ³K»
					doubleMatImage_tmp.data[doubleMatIndex] = (double)doubleTMP;
					cout << (double)doubleMatImage.data[doubleMatIndex] << " -> " << doubleTMP << " = " << (double)doubleMatImage_tmp.data[doubleMatIndex] << endl;	// mFp
				}
			}
		}
		cout << " Standardization :  ex.) " << doubleMatImage.at<Vec3f>(10, 10);	// mFp1
		doubleMatImage_tmp.copyTo(doubleMatImage);
		cout << " => " << doubleMatImage_tmp.at<Vec3f>(10, 10) << endl;	// mFp2
	}
}
// Ki»π³ΜlΝΝΙΰΗ·
void doubleMatReturnStandardization(Mat& doubleMatImage) {
	int x, y, c;
	int doubleMatIndex;
	double doubleTMP[3];
	Vec3f doublePut;
	Mat doubleMatImage_tmp = Mat(doubleMatImage.size(), CV_64FC3);
#pragma omp parallel for private(x, c)
	for (y = 0; y < doubleMatImage.rows; y++) {
		for (x = 0; x < doubleMatImage.cols; x++) {
			doubleMatIndex = (y * doubleMatImage.cols + x) * 3;
			for (c = 0; c < 3; c++) {
				doubleTMP[c] = (double)doubleMatImage.data[doubleMatIndex + c];
				doubleTMP[c] = (double)doubleTMP[c] * (double)MAX_INTENSE;	// 0~1 ³K»ΰΗ΅
				doubleMatImage_tmp.data[doubleMatIndex + c] = (double)doubleTMP[c];
				cout << (double)doubleMatImage.data[doubleMatIndex] << " -> " << doubleTMP << " = " << (double)doubleMatImage_tmp.data[doubleMatIndex] << endl;	// mFp
			}
			/*doublePut = Vec3f(doubleTMP[0], doubleTMP[1], doubleTMP[2]);
			doubleMatImage_tmp.at<Vec3f>(y, x) = doublePut;*/
		}
	}
	cout << " ReturnStandardization :  ex.) " << doubleMatImage.at<Vec3f>(10, 10);	// mFp1
	doubleMatImage_tmp.copyTo(doubleMatImage);
	cout << " => " << doubleMatImage_tmp.at<Vec3f>(10, 10) << endl;	// mFp
}

// ½ΟlΜZo
double CalcAverage(Mat& doubleMat) {
	int x, y, c;
	int CalAveIndex;
	double average = 0.0;
	int ALL_PIX_NUM = doubleMat.rows * doubleMat.cols * 3;
	if (doubleMat.type() != CV_64FC3 && doubleMat.type() != CV_8UC3) {
		cout << "ERROR! CalcAverage : Input type is wrong." << endl;
	}
	else {
#pragma omp parallel for private(x, c)
		for (y = 0; y < doubleMat.rows; y++) {
			for (x = 0; x < doubleMat.cols; x++) {
				for (c = 0; c < 3; c++) {
					CalAveIndex = (y * doubleMat.cols + x) * 3.0 + c;
					average += (double)doubleMat.data[CalAveIndex] / (double)ALL_PIX_NUM;
				}
			}
		}
	}

	//cout << "  average = " << average << endl;	// mFp
	return average;
}
// double^3`lΜMatζπS»
void doubleMatCentralization(Mat& doubleMatImage) {
	int x, y, c;
	int doubleMatIndex;
	double doubleAverage;
	/* ½ΟlΜZo */
	doubleAverage = CalcAverage(doubleMatImage);
	Mat doubleMatImage_tmp = Mat(doubleMatImage.size(), CV_64FC3);

	if (doubleMatImage.type() != CV_64FC3) {
		cout << "ERROR! CalcAverage : Input type is wrong." << endl;
	}
	else {
		//cout << " Centralization :  ex.) " << doubleMatImage.at<Vec3f>(10, 10) << " => ";	// mFp
		cout << " Centralization :  ex.) " << (double)doubleMatImage.data[200] << " => ";	// mFp
#pragma omp parallel for private(x, c)
		for (y = 0; y < doubleMatImage.rows; y++) {
			for (x = 0; x < doubleMatImage.cols; x++) {
				for (c = 0; c < 3; c++) {
					doubleMatIndex = (y * doubleMatImage.cols + x) * 3 + c;
					//cout << (double)doubleMatImage.data[doubleMatIndex] << "->" << (double)((double)doubleMatImage.data[doubleMatIndex] - (double)doubleAverage) << "=";	// mFp
					//doubleMatImage.data[doubleMatIndex] -= (double)doubleAverage;
					doubleMatImage_tmp.data[doubleMatIndex] -= (double)doubleAverage;
					//cout << (double)doubleMatImage.data[doubleMatIndex] << endl;	// mFp
					//cout << (double)doubleMatImage_tmp.data[doubleMatIndex] << endl;	// mFp
				}
			}
		}
		doubleMatImage_tmp.copyTo(doubleMatImage);
		//cout << doubleMatImage.at<Vec3f>(10, 10) << endl;	// mFp
		cout << (double)doubleMatImage.data[200] << endl;	// mFp
	}
}
// S»π³ΜlΝΝΙΰΗ·
void doubleMatReturnCentralization(Mat& doubleMatImage, double& double_Average) {
	int x, y, c;
	int doubleMatIndex;
	double doubleAverage;

	if (doubleMatImage.type() != CV_64FC3) {
		cout << "ERROR! CalcAverage : Input type is wrong." << endl;
	}
	else {
		cout << " average = " << double_Average << endl;;	// mFp
		//cout << " ReturnCentralization :  ex.) " << doubleMatImage.at<Vec3f>(10, 10) << " => ";	// mFp
		cout << " Centralization :  ex.) " << (double)doubleMatImage.data[200] << " => ";	// mFp
#pragma omp parallel for private(x, c)
		for (y = 0; y < doubleMatImage.rows; y++) {
			for (x = 0; x < doubleMatImage.cols; x++) {
				for (c = 0; c < 3; c++) {
					doubleMatIndex = (y * doubleMatImage.cols + x) * 3 + c;
					doubleMatImage.data[doubleMatIndex] += (double)double_Average;
				}
			}
		}
		//cout << doubleMatImage.at<Vec3f>(10, 10) << endl;	// mFp
		cout << (double)doubleMatImage.data[200] << endl;	// mFp
	}
}


/* NX */
// GaussianMarkovModel(GMM)
class GMM {
private:
	int imgK;
	int GMMx, GMMy, GMMc;
	int GMM_index;
public:
	int imageK;				// mCYζΜ K
	int imageMAX_INTENSE;	// Εεζfl
	int GMM_XSIZE;			// ζΜ
	int GMM_YSIZE;			// ζΜ³
	int GMM_MAX_PIX;		// ζΜsNZ
	Mat POSTERIOR;			// γͺz
	vector<Mat> LIKELIHOOD;	// ήx
	Mat averageVector;			// ½ΟxNg
	vector<double> AverageVector;
	Mat averageImage;			// ½Οζ
	vector<double> AverageImage;
	Mat averageSquareImage;		// 2ζ½Οζ
	vector<double> AverageSquareImage;
	Mat eigenValue;				// OtvVAΜΕLl
	vector<double> EigenValue;

	// ½Οl
	vector<double> LIKELIHOOD_Average;
	double POSTERIOR_Average;
	double averageImage_Average;

	// p[^
	double GMM_mean;
	double GMM_lambda;
	double GMM_alpha;
	double GMM_gamma;
	double GMM_sigma;
	double GMM_sigma2;

	GMM();
	void putLIKELIHOOD(int, vector<Mat>);
	void CreateOutputImage(Mat&);
	void Standardization();			// Ki»
	void ReturnStandardization();
	void Centralization();			// S»
	void ReturnCentralization();
	void CreateDoubleAverageImageMat();			// ½ΟζidoubleljΜΆ¬
	void CreateDoubleAverageSquareImageMat();	// 2ζ½ΟζidoubleljΜΆ¬
	void eigenValueGrid2D();					// OtvVAΜΕLlvZ
};
GMM::GMM() {
	imageK = 1;
	imageMAX_INTENSE = MAX_INTENSE;
	GMM_XSIZE = WIDTH;
	GMM_YSIZE = HEIGHT;
	GMM_MAX_PIX = GMM_XSIZE * GMM_YSIZE;
	LIKELIHOOD.clear();

	POSTERIOR = Mat(Size(Image_dst.cols, Image_dst.rows), CV_64FC3);
	POSTERIOR_Average = CalcAverage(Image_dst);
#pragma omp parallel for private(x, c)
	for (GMMy = 0; GMMy < Image_dst.rows; GMMy++) {
		for (GMMx = 0; GMMx < Image_dst.cols; GMMx++) {
			for (GMMc = 0; GMMc < 3; GMMc++) {
				GMM_index = (GMMy * Image_dst.cols + GMMx) * 3 + GMMc;
				POSTERIOR.data[GMM_index] = (double)Image_dst.data[GMM_index];
			}
		}
	}

	averageVector = Mat(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	averageImage = Mat(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	AverageVector.clear();
	AverageImage.clear();
	averageSquareImage = Mat(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	AverageSquareImage.clear();
	eigenValue = Mat(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	EigenValue.clear();

	GMM_mean = 0.0;
	GMM_lambda = 1.0e-7;
	GMM_alpha = 1.0e-4;
	GMM_gamma = 1.0e-5;
	GMM_sigma = 30;
	GMM_sigma2 = GMM_sigma * GMM_sigma;
}
void GMM::putLIKELIHOOD(int K, vector<Mat> K_Image) {
	imageK = K;
	Mat doubleMatTmp;

	if (K_Image.size() != imageK || K_Image[0].type() != CV_8UC3) { cout << "ERROR! GMM::putLIKELIHOOD" << endl; }
	else {
		LIKELIHOOD.clear();
#pragma omp parallel for private(GMMx, GMMc, doubleMatTmp)
		for (imgK = 0; imgK < imageK; imgK++) {
			doubleMatTmp = Mat(Size(K_Image[0].cols, K_Image[0].rows), CV_64FC3);
			for (GMMy = 0; GMMy < K_Image[imgK].rows; GMMy++) {
				for (GMMx = 0; GMMx < K_Image[imgK].cols; GMMx++) {
					for (GMMc = 0; GMMc < 3; GMMc++) {
						GMM_index = (GMMy * K_Image[imgK].cols + GMMx) * 3 + GMMc;
						doubleMatTmp.data[GMM_index] = (double)K_Image[imgK].data[GMM_index];
					}
				}
			}
			LIKELIHOOD.push_back(doubleMatTmp);
		}
	}
}
void GMM::CreateOutputImage(Mat& Output_Image) {
	Output_Image = Mat(POSTERIOR.size(), CV_8UC3);
#pragma omp parallel for private(GMMx, GMMc)
	for (GMMy = 0; GMMy < GMM_YSIZE; GMMy++) {
		for (GMMx = 0; GMMx < GMM_XSIZE; GMMx++) {
			for (GMMc = 0; GMMc < 3; GMMc++) {
				GMM_index = (GMMy * GMM_XSIZE + GMMx) * 3 + GMMc;
				Output_Image.data[GMM_index] = (uchar)POSTERIOR.data[GMM_index];
				//cout << "  " << (int)Output_Image.data[Denoising_index] << endl;	// mFp
			}
		}
	}
}
void GMM::Standardization() {
	doubleMatStandardization(POSTERIOR);
	for (imgK = 0; imgK < imageK; imgK++) {
		doubleMatStandardization(LIKELIHOOD[imgK]);
	}
}
void GMM::ReturnStandardization() {
	doubleMatReturnStandardization(POSTERIOR);
	doubleMatReturnStandardization(averageVector);
	for (imgK = 0; imgK < imageK; imgK++) {
		doubleMatReturnStandardization(LIKELIHOOD[imgK]);
	}
}
void GMM::Centralization() {
	double center_num;
	POSTERIOR_Average = CalcAverage(POSTERIOR);
	doubleMatCentralization(POSTERIOR);
	for (imgK = 0; imgK < imageK; imgK++) {
		center_num = CalcAverage(LIKELIHOOD[imgK]);
		LIKELIHOOD_Average.push_back(center_num);
		doubleMatCentralization(LIKELIHOOD[imgK]);
	}
}
void GMM::ReturnCentralization() {
	doubleMatReturnCentralization(POSTERIOR, POSTERIOR_Average);
	doubleMatReturnCentralization(averageVector, POSTERIOR_Average);
	for (imgK = 0; imgK < imageK; imgK++) {
		doubleMatReturnCentralization(LIKELIHOOD[imgK], LIKELIHOOD_Average[imgK]);
	}
}
void GMM::CreateDoubleAverageImageMat() {
	double pix[3];
	Vec3b color;
#pragma omp parallel for private(GMMx, GMMc, imgK, pix[0], pix[1], pix[2])
	for (GMMy = 0; GMMy < GMM_YSIZE; GMMy++) {
		for (GMMx = 0; GMMx < GMM_XSIZE; GMMx++) {
			GMM_index = (GMMy * GMM_XSIZE + GMMx) * 3;
			pix[0] = 0.0;
			pix[1] = 0.0;
			pix[2] = 0.0;
			for (imgK = 0; imgK < imageK; imgK++) {
				pix[0] += (double)LIKELIHOOD[imgK].data[GMM_index + 0] / (double)imageK;
				pix[1] += (double)LIKELIHOOD[imgK].data[GMM_index + 1] / (double)imageK;
				pix[2] += (double)LIKELIHOOD[imgK].data[GMM_index + 2] / (double)imageK;
			}

			for (GMMc = 0; GMMc < 3; GMMc++) {
				averageImage.data[GMM_index + GMMc] = (double)pix[GMMc];
			}
		}
	}
	// S»
	averageImage_Average = CalcAverage(averageImage);
	//doubleMatCentralization(averageImage);

	averageImage.copyTo(averageVector);
	POSTERIOR_Average = averageImage_Average;

	AverageVector.clear();
	AverageImage.clear();
	double pix_num = CalcAverage(averageVector);;
	for (GMMy = 0; GMMy < GMM_YSIZE; GMMy++) {
		for (GMMx = 0; GMMx < GMM_XSIZE; GMMx++) {
			for (GMMc = 0; GMMc < 3; GMMc++) {
				GMM_index = (GMMy * GMM_XSIZE + GMMx) * 3 + GMMc;
				pix[0] = (double)averageImage.data[GMM_index];
				pix[1] = (double)(pix[0] - averageImage_Average);
				pix[2] = (double)(pix[0] - pix_num);

				//cout << " Image_original : " << (double)pix[0] << endl;	// mFp
				AverageImage.push_back(pix[1]);
				AverageVector.push_back(pix[2]);
				//cout << " Image:" << (double)AverageImage[GMM_index] << " , vector:" << (double)AverageVector[GMM_index] << endl;	// mFp
			}
		}
	}
}
void GMM::CreateDoubleAverageSquareImageMat() {
	double pix[3];
	Vec3b color;
#pragma omp parallel for private(GMMx, GMMc, imgK, pix[0], pix[1], pix[2])
	for (GMMy = 0; GMMy < GMM_YSIZE; GMMy++) {
		for (GMMx = 0; GMMx < GMM_XSIZE; GMMx++) {
			GMM_index = (GMMy * GMM_XSIZE + GMMx) * 3;
			pix[0] = 0.0;
			pix[1] = 0.0;
			pix[2] = 0.0;
			for (imgK = 0; imgK < imageK; imgK++) {
				pix[0] += (double)LIKELIHOOD[imgK].data[GMM_index + 0] / (double)imageK;
				pix[1] += (double)LIKELIHOOD[imgK].data[GMM_index + 1] / (double)imageK;
				pix[2] += (double)LIKELIHOOD[imgK].data[GMM_index + 2] / (double)imageK;
				pix[0] = (double)pow(pix[0], 2);
				pix[1] = (double)pow(pix[1], 2);
				pix[2] = (double)pow(pix[2], 2);
			}

			for (GMMc = 0; GMMc < 3; GMMc++) {
				averageSquareImage.data[GMM_index + GMMc] = (double)pix[GMMc];
			}
		}
	}

	AverageSquareImage.clear();
	double pix_num;
	for (GMMy = 0; GMMy < GMM_YSIZE; GMMy++) {
		for (GMMx = 0; GMMx < GMM_XSIZE; GMMx++) {
			for (GMMc = 0; GMMc < 3; GMMc++) {
				GMM_index = (GMMy * GMM_XSIZE + GMMx) * 3 + GMMc;
				pix_num = (double)averageImage.data[GMM_index] - (double)averageImage_Average;
				pix_num = (double)pow(pix_num, 2);
				AverageSquareImage.push_back(pix_num);
			}
		}
	}
}
void GMM::eigenValueGrid2D() {
	EigenValue.clear();
	double eigenValue_tmp;
#pragma omp parallel for private(GMMx, GMMc)
	for (GMMy = 0; GMMy < GMM_YSIZE; GMMy++) {
		for (GMMx = 0; GMMx < GMM_XSIZE; GMMx++) {
			for (GMMc = 0; GMMc < 3; GMMc++) {
				GMM_index = (GMMy * GMM_XSIZE + GMMx) * 3 + GMMc;
				eigenValue_tmp = (double)(4 * sin(0.5 * GMMx * _M_PI / GMM_XSIZE) * sin(0.5 * GMMx * _M_PI / GMM_XSIZE) + 4 * sin(0.5 * GMMy * _M_PI / GMM_YSIZE) * sin(0.5 * GMMy * _M_PI / GMM_YSIZE));
				EigenValue.push_back(eigenValue_tmp);
				eigenValue.data[GMM_index] = eigenValue_tmp;
				//cout << (double)(4 * sin(0.5 * GMMx * _M_PI / GMM_XSIZE) * sin(0.5 * GMMx * _M_PI / GMM_XSIZE) + 4 * sin(0.5 * GMMy * _M_PI / GMM_YSIZE) * sin(0.5 * GMMy * _M_PI / GMM_YSIZE)) << endl;	// mFp
				//cout << (double)eigenValue.data[GMM_index] << endl;	// mFp
				//cout << (double)EigenValue[GMM_index] << endl;	// mFp
			}
		}
	}
}


/* Φ(wb_[Εθ`) */
void GaussianNoiseImage(const int K, const Mat& ORIGINAL_IMG, vector<Mat>& NOISE_IMG);			// K Μς»ζΜΆ¬ (KEXmCYtΑ)
void CreateAverageImage(const int K, Mat& AVERAGE_IMG, const vector<Mat>& ALL_IMG, const double maxINTENSE);	// ½ΟζiRGBljΜΆ¬
void DenoisingMRF(Mat& DenoiseImage, const vector<Mat>& noiseImage, const int K, const int maxIntensity);
void DenoisingHMRF(Mat& DenoiseImage, const vector<Mat>& noiseImage, const int K, const int maxIntensity);