#ifndef __INCLUDED_H_EstimatedParameterHMRF__
#define __INCLUDED_H_EstimatedParameterHMRF__

#include "main.h"


/* MRFのパラメータ */
double CONVERGE_H = 1.0e-8;		// パラメータ推定の収束判定値
int MAXIteration_H = 500;		// パラメータ推定の最大反復回数
const double LearningRate_Alpha_H = 1.0e-5;		// 学習率
const double LearningRate_Lambda_H = 1.0e-10;
const double LearningRate_Gamma_H = 1.0e-7;
//const double LearningRate_Alpha_H = 1.0e-6;		// 学習率
//const double LearningRate_Lambda_H = 1.0e-12;
//const double LearningRate_Gamma_H = 1.0e-9;
double SIGMA_HMRF_H = 35;		// パラメータ
double ALPHA_HMRF_H = 1.0e-4;
double LAMBDA_HMRF_H = 1.0e-7;
double GAMMA_HMRF_H = 1.0e-3;

/* 関数(クラス内計算用) */
// プサイ関数（波動関数）
double Psi_H(double& eigenvalue, double& lambda, double& alpha, double& gamma) {
	double calc_result, calc_numer, calc_denom;
	calc_numer = (double)(lambda + alpha * eigenvalue);
	calc_denom = (double)(calc_numer + gamma);
	calc_numer = (double)pow(calc_numer, 2);
	calc_result = (double)calc_numer / (double)calc_denom;

	return calc_result;
}
// カイ関数
double Kai_H(int& K, double& sigma2, double& psi) {
	double calc_result;
	calc_result = (double)K / sigma2 + psi;

	return calc_result;
}
double Kai_original_H(int& K, double& sigma2, double& eigenvalue, double& lambda, double& alpha, double& gamma) {
	double calc_result;
	calc_result = (double)K / sigma2 + (double)Psi_H(eigenvalue, lambda, alpha, gamma);

	return calc_result;
}


// Mat Σ_{(i,j)inV} 1/(lambda+(K/sigma2)+alpha*fhi) の計算
void CalculationFunction1_H(double& sigma2, double& alpha, double& lambda, double& gamma, int K, Mat& GuraphRap, Mat& X) {
	int x, y, c;
	const int imageSizeX = GuraphRap.cols;
	const int imageSizeY = GuraphRap.rows;
	int Function1_index;
	double function1_tmp, number_tmp;

#pragma omp parallel for private(x, c)
	for (y = 0; y < imageSizeY; y++) {
		for (x = 0; x < imageSizeX; x++) {
			for (c = 0; c < 3; c++) {
				Function1_index = (y * imageSizeX + x) * 3 + c;
				number_tmp = (double)GuraphRap.data[Function1_index];
				function1_tmp = (double)Kai_original_H(K, sigma2, number_tmp, lambda, alpha, gamma);
				X.data[Function1_index] = 1.0 / (double)function1_tmp;
			}
		}
	}
}
double CalculationFunction1_H_E(double& sigma2, double& alpha, double& lambda, double& gamma, int K, vector<double>& GuraphRap, Mat& X, vector<double> X_result) {
	int x, y, c;
	const int imageSizeX = X.cols;
	const int imageSizeY = X.rows;
	int Function1_index;
	double function1_tmp, number_tmp;
	X_result.clear();
	double sum_tmp = 0.0;

#pragma omp parallel for private(x, c)
	for (y = 0; y < imageSizeY; y++) {
		for (x = 0; x < imageSizeX; x++) {
			for (c = 0; c < 3; c++) {
				Function1_index = (y * imageSizeX + x) * 3 + c;
				number_tmp = (double)GuraphRap[Function1_index];
				function1_tmp = (double)Kai_original_H(K, sigma2, number_tmp, lambda, alpha, gamma);
				function1_tmp = 1.0 / (double)function1_tmp;
				X.data[Function1_index] = (double)function1_tmp;
				X_result.push_back(function1_tmp);
				sum_tmp += function1_tmp;
				//cout << (double)X.data[Function1_index] << " = " << (double)function1_tmp << endl;	// 確認用
				//cout << (double)X_result[Function1_index] << " = " << (double)function1_tmp << endl;	// 確認用
			}
		}
	}
	return sum_tmp;
}

// パラメータ推定時の関数
double CalcFunction_lambda(double lambda, double alpha, double gamma, Mat& GuraphRap, Mat& Kai) {
	double result_lambda = 0.0;
	int x, y, c;
	int lambda_index;

	double lambda_num_tmp1, lambda_num_tmp2;
#pragma omp parallel for private(x, c)
	for (y = 0; y < Kai.rows; y++) {
		for (x = 0; x < Kai.cols; x++) {
			for (c = 0; c < 3; c++) {
				lambda_index = (y * Kai.cols + x) * 3 + c;
				lambda_num_tmp1 = (double)(lambda + alpha * GuraphRap.data[lambda_index]);
				lambda_num_tmp2 = (double)(gamma + lambda + alpha * GuraphRap.data[lambda_index]);
				lambda_num_tmp1 = (double)(2.0 / lambda_num_tmp1) - (double)(1.0 / lambda_num_tmp2);
				if ((double)Kai.data[lambda_index] != 0) {
					result_lambda += ((double)lambda_num_tmp1 / (double)Kai.data[lambda_index]);
				}
			}
		}
	}

	return result_lambda;
}
double CalcFunction_lambda_E(double lambda, double alpha, double gamma, vector<double>& GuraphRap, Mat& Kai, int& K, double& sigma2) {
	double result_lambda = 0.0;
	int x, y, c;
	int lambda_index;

	double lambda_num_tmp1, lambda_num_tmp2, lambda_num_tmp3;
	double kai_tmp;
#pragma omp parallel for private(x, c)
	for (y = 0; y < Kai.rows; y++) {
		for (x = 0; x < Kai.cols; x++) {
			for (c = 0; c < 3; c++) {
				lambda_index = (y * Kai.cols + x) * 3 + c;
				lambda_num_tmp1 = (double)(lambda + alpha * GuraphRap[lambda_index]);
				lambda_num_tmp2 = (double)(gamma + lambda + alpha * GuraphRap[lambda_index]);
				//cout << (double)lambda_num_tmp1 << " , " << (double)lambda_num_tmp2 << endl;	// 確認用
				lambda_num_tmp3 = ((double)2.0 / (double)lambda_num_tmp1) - ((double)1.0 / (double)lambda_num_tmp2);
				//cout << (double)lambda_num_tmp3 << " = " << (double)(2.0 / lambda_num_tmp1) << " - " << (double)(1.0 / lambda_num_tmp2) << endl;	// 確認用
				kai_tmp = Kai_original_H(K, sigma2, GuraphRap[lambda_index], lambda, alpha, gamma);
				if (kai_tmp != 0) {
					result_lambda += ((double)lambda_num_tmp3 / (double)kai_tmp);
				}
				//cout << (double)lambda_num_tmp3 << " / " << (double)Kai.data[lambda_index] << endl;	// 確認用
				//cout << (double)result_lambda << endl;	// 確認用
			}
		}
	}
	//cout << (double)result_lambda << endl;	// 確認用

	return result_lambda;
}

double CalcFunction_alpha(double lambda, double alpha, double gamma, Mat& GuraphRap, Mat& Kai) {
	double result_alpha = 0.0;
	int x, y, c;
	int alpha_index;

	double alpha_num_tmp1, alpha_num_tmp2;
#pragma omp parallel for private(x, c)
	for (y = 0; y < Kai.rows; y++) {
		for (x = 0; x < Kai.cols; x++) {
			for (c = 0; c < 3; c++) {
				alpha_index = (y * Kai.cols + x) * 3 + c;
				alpha_num_tmp1 = (double)(lambda + alpha * GuraphRap.data[alpha_index]);
				alpha_num_tmp2 = (double)(gamma + lambda + alpha * GuraphRap.data[alpha_index]);
				alpha_num_tmp1 = (double)(2.0 / alpha_num_tmp1) - (double)(1.0 / alpha_num_tmp2);
				if ((double)Kai.data[alpha_index] != 0) {
					result_alpha += ((double)GuraphRap.data[alpha_index] / (double)Kai.data[alpha_index]) * (double)alpha_num_tmp1;
				}
			}
		}
	}

	return result_alpha;
}
double CalcFunction_alpha_E(double lambda, double alpha, double gamma, vector<double>& GuraphRap, Mat& Kai, int& K, double& sigma2) {
	double result_alpha = 0.0;
	int x, y, c;
	int alpha_index;

	double alpha_num_tmp1, alpha_num_tmp2;
	double kai_tmp;
#pragma omp parallel for private(x, c)
	for (y = 0; y < Kai.rows; y++) {
		for (x = 0; x < Kai.cols; x++) {
			for (c = 0; c < 3; c++) {
				alpha_index = (y * Kai.cols + x) * 3 + c;
				alpha_num_tmp1 = (double)(lambda + alpha * GuraphRap[alpha_index]);
				alpha_num_tmp2 = (double)(gamma + lambda + alpha * GuraphRap[alpha_index]);
				alpha_num_tmp1 = (double)(2.0 / alpha_num_tmp1) - (double)(1.0 / alpha_num_tmp2);
				kai_tmp = Kai_original_H(K, sigma2, GuraphRap[alpha_index], lambda, alpha, gamma);
				if (kai_tmp != 0) {
					result_alpha += ((double)GuraphRap[alpha_index] / (double)kai_tmp) * (double)alpha_num_tmp1;
				}
			}
		}
	}

	return result_alpha;
}

double CalcFunction_alpha2(Mat& X) {
	double result_alpha2 = 0.0;
	int x, y, c;
	int alpha_index, alpha_index2;

	double alpha_num_tmp;
#pragma omp parallel for private(x, c)
	for (y = 0; y < X.rows; y++) {
		for (x = 0; x < X.cols; x++) {
			for (c = 0; c < 3; c++) {
				alpha_index = (y * X.cols + x) * 3 + c;
				if (x > 0) {
					alpha_index2 = (y * X.cols + (x - 1)) * 3 + c;
					alpha_num_tmp = ((double)X.data[alpha_index] - (double)X.data[alpha_index2]);
					result_alpha2 += (double)pow(alpha_num_tmp, 2);
				}
				if (x + 1 < X.cols) {
					alpha_index2 = (y * X.cols + (x + 1)) * 3 + c;
					alpha_num_tmp = ((double)X.data[alpha_index] - (double)X.data[alpha_index2]);
					result_alpha2 += (double)pow(alpha_num_tmp, 2);
				}
				if (y > 0) {
					alpha_index2 = ((y - 1) * X.cols + x) * 3 + c;
					alpha_num_tmp = ((double)X.data[alpha_index] - (double)X.data[alpha_index2]);
					result_alpha2 += (double)pow(alpha_num_tmp, 2);
				}
				if (y + 1 < X.rows) {
					alpha_index2 = ((y + 1) * X.cols + x) * 3 + c;
					alpha_num_tmp = ((double)X.data[alpha_index] - (double)X.data[alpha_index2]);
					result_alpha2 += (double)pow(alpha_num_tmp, 2);
				}
			}
		}
	}

	return result_alpha2;
}
double CalcFunction_alpha3(int Xsize, int Ysize, vector<double>& X) {
	double result_alpha2 = 0.0;
	int x, y, c;
	int alpha_index, alpha_index2;

	double alpha_num_tmp;
#pragma omp parallel for private(x, c)
	for (y = 0; y < Ysize; y++) {
		for (x = 0; x < Xsize; x++) {
			for (c = 0; c < 3; c++) {
				alpha_index = (y * Xsize + x) * 3 + c;
				if (x > 0) {
					alpha_index2 = (y * Xsize + (x - 1)) * 3 + c;
					alpha_num_tmp = ((double)X[alpha_index] - (double)X[alpha_index2]);
					result_alpha2 += (double)pow(alpha_num_tmp, 2);
				}
				if (x + 1 < Xsize) {
					alpha_index2 = (y * Xsize + (x + 1)) * 3 + c;
					alpha_num_tmp = ((double)X[alpha_index] - (double)X[alpha_index2]);
					result_alpha2 += (double)pow(alpha_num_tmp, 2);
				}
				if (y > 0) {
					alpha_index2 = ((y - 1) * Xsize + x) * 3 + c;
					alpha_num_tmp = ((double)X[alpha_index] - (double)X[alpha_index2]);
					result_alpha2 += (double)pow(alpha_num_tmp, 2);
				}
				if (y + 1 < Ysize) {
					alpha_index2 = ((y + 1) * Xsize + x) * 3 + c;
					alpha_num_tmp = ((double)X[alpha_index] - (double)X[alpha_index2]);
					result_alpha2 += (double)pow(alpha_num_tmp, 2);
				}
			}
		}
	}

	return result_alpha2;
}

double CalcFunction_gamma(double lambda, double alpha, double gamma, Mat& GuraphRap, Mat& Kai) {
	double result_gamma = 0.0;
	int x, y, c;
	int gamma_index;

	double gamma_num_tmp;
#pragma omp parallel for private(x, c)
	for (y = 0; y < Kai.rows; y++) {
		for (x = 0; x < Kai.cols; x++) {
			for (c = 0; c < 3; c++) {
				gamma_index = (y * Kai.cols + x) * 3 + c;
				gamma_num_tmp = 1.0 / (double)(gamma + lambda + alpha * GuraphRap.data[gamma_index]);
				if ((double)Kai.data[gamma_index] != 0) {
					result_gamma += ((double)gamma_num_tmp / (double)Kai.data[gamma_index]);
				}
			}
		}
	}

	return result_gamma;
}
double CalcFunction_gamma_E(double lambda, double alpha, double gamma, vector<double>& GuraphRap, Mat& Kai, int& K, double& sigma2) {
	double result_gamma = 0.0;
	int x, y, c;
	int gamma_index;

	double gamma_num_tmp;
	double kai_tmp;
#pragma omp parallel for private(x, c)
	for (y = 0; y < Kai.rows; y++) {
		for (x = 0; x < Kai.cols; x++) {
			for (c = 0; c < 3; c++) {
				gamma_index = (y * Kai.cols + x) * 3 + c;
				gamma_num_tmp = 1.0 / (double)(gamma + lambda + alpha * GuraphRap[gamma_index]);
				kai_tmp = Kai_original_H(K, sigma2, GuraphRap[gamma_index], lambda, alpha, gamma);
				if (kai_tmp != 0) {
					result_gamma += ((double)gamma_num_tmp / (double)kai_tmp);
				}
			}
		}
	}

	return result_gamma;
}


/* クラス */
// GMM HierarchicalMarkovRandomField
class GMM_HMRF : public GMM {
private:
	int imgK;
	int HMRFx, HMRFy, HMRFc;
	int HMRF_index;
public:
	double CONVERGE_HMRF_H = CONVERGE_H;		// 収束判定値
	int MaxIteration_HMRF_H = MAXIteration_H;		// 最大反復回数
	Mat averageVector2;
	vector<double> averageVec2;
	Mat MapW;
	vector<double> Map_W;

	GMM_HMRF(int, int, Mat&, vector<Mat>, double, double, double, double);
	void CreateOutput();				// 修復画像の生成
	void MaximumPosteriorEstimation();	// 事後分布のガウスザイデル法によるMAP推定
	void MaximumMapWEstimation();	// 周辺尤度のガウスザイデル法によるMAP推定
	void EstimatedParameter(double, int);	// EMアルゴリズムによるパラメータ推定
};
GMM_HMRF::GMM_HMRF(int K, int max_intense, Mat& ImageDst, vector<Mat> likelihood, double gamma, double lambda, double alpha, double sigma) {
	imageK = K;
	imageMAX_INTENSE = max_intense;
	GMM_XSIZE = ImageDst.cols;
	GMM_YSIZE = ImageDst.rows;
	GMM_MAX_PIX = GMM_XSIZE * GMM_YSIZE;
	LIKELIHOOD.clear();
	LIKELIHOOD_Average.clear();
	Mat lilelihood_tmp;
#pragma omp parallel for private(MRFy, MRFx, MRFc)
	for (imgK = 0; imgK < imageK; imgK++) {
		lilelihood_tmp = Mat(Size(ImageDst.cols, ImageDst.rows), CV_64FC3);
		for (HMRFy = 0; HMRFy < ImageDst.rows; HMRFy++) {
			for (HMRFx = 0; HMRFx < ImageDst.cols; HMRFx++) {
				for (HMRFc = 0; HMRFc < 3; HMRFc++) {
					HMRF_index = (HMRFy * ImageDst.cols + HMRFx) * 3 + HMRFc;
					lilelihood_tmp.data[HMRF_index] = (double)likelihood[imgK].data[HMRF_index];
				}
			}
		}
		LIKELIHOOD.push_back(lilelihood_tmp);
		
		double Average = CalcAverage(lilelihood_tmp);
		LIKELIHOOD_Average.push_back(Average);
	}

	POSTERIOR = Mat(Size(ImageDst.cols, ImageDst.rows), CV_64FC3);
#pragma omp parallel for private(MRFx, MRFc)
	for (HMRFy = 0; HMRFy < ImageDst.rows; HMRFy++) {
		for (HMRFx = 0; HMRFx < ImageDst.cols; HMRFx++) {
			for (HMRFc = 0; HMRFc < 3; HMRFc++) {
				HMRF_index = (HMRFy * ImageDst.cols + HMRFx) * 3 + HMRFc;
				POSTERIOR.data[HMRF_index] = (double)ImageDst.data[HMRF_index];
			}
		}
	}

	averageVector = Mat::zeros(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	averageVector2 = Mat::zeros(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	averageVec2.clear();
	averageImage = Mat::zeros(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	averageSquareImage = Mat(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	eigenValue = Mat(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	EigenValue.clear();
	MapW = Mat(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	Map_W.clear();

	GMM_gamma = gamma;
	GMM_lambda = lambda;
	GMM_alpha = alpha;
	GMM_sigma = sigma;
	GMM_sigma2 = GMM_sigma * GMM_sigma;
}
void GMM_HMRF::CreateOutput() {
	//POSTERIOR.copyTo(averageVector);
	//LIKELIHOOD[1].copyTo(averageVector);
	//averageImage.copyTo(averageVector);

	double color_checker;
#pragma omp parallel for private(x, c, color_checker)
	for (HMRFy = 0; HMRFy < GMM_YSIZE; HMRFy++) {
		for (HMRFx = 0; HMRFx < GMM_XSIZE; HMRFx++) {
			for (HMRFc = 0; HMRFc < 3; HMRFc++) {
				HMRF_index = (HMRFy * GMM_XSIZE + HMRFx) * 3 + HMRFc;
				color_checker = (double)(averageVector.data[HMRF_index]);
				if (color_checker < 0) {
					POSTERIOR.data[HMRF_index] = (double)0;
				}
				else if (color_checker > imageMAX_INTENSE) {
					POSTERIOR.data[HMRF_index] = (double)imageMAX_INTENSE;
				}
				else {
					POSTERIOR.data[HMRF_index] = (double)color_checker;
				}
			}
		}
	}
}
void GMM_HMRF::MaximumPosteriorEstimation() {
	int HMRF_POST_flg = 1; // 収束判定フラグ
	int x, y, c;

	double errorConvergence = 0.0;
	double numer[3], denom, ave[3], Yi[3];
	double numer2[3], denom2, ave2[3];
	int adjacent_pix_num;
	int col_index;

	averageVec2.clear();
	// 確率分布の初期化
	Mat RandomMap_B = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));	// 確率変数m
	Mat RandomMap_G = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));
	Mat RandomMap_R = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));
	Mat RandomMap_B2 = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));	// 確率変数myu
	Mat RandomMap_G2 = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));
	Mat RandomMap_R2 = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));
	//averageImage.copyTo(averageVector);	 // 事後分布の平均ベクトル初期化

	for (int count = 0; count < MaxIteration_HMRF_H; count++) {
		errorConvergence = 0.0;
#pragma omp parallel for private(x, numer, denom, ave, numer2, denom2, ave2, c) reduction(+ : errorConvergence)
		for (y = 0; y < GMM_YSIZE; y++) {
			for (x = 0; x < GMM_XSIZE; x++) {
				col_index = (y * GMM_XSIZE + x) * 3;
				for (int c = 0; c < 3; c++) {
					Yi[c] = (double)averageImage.data[col_index + c];
				}
				//Yi[2] = ((double)(Yi[2] * 2.0) / (double)imageMAX_INTENSE) - 1.0;	// (-1)~1 正規化
				//Yi[1] = ((double)(Yi[1] * 2.0) / (double)imageMAX_INTENSE) - 1.0;
				//Yi[0] = ((double)(Yi[0] * 2.0) / (double)imageMAX_INTENSE) - 1.0;
				for (int c = 0; c < 3; c++) {
					numer[c] = (double)(Yi[c] * (double)imageK / GMM_sigma2);
					numer2[c] = 0.0;
				}
				denom = GMM_lambda + ((double)imageK / GMM_sigma2);
				denom2 = GMM_lambda + GMM_gamma;
				adjacent_pix_num = 0;

				if (x > 0) {
					numer[2] += GMM_alpha * (double)RandomMap_R.at<double>(y, x - 1);
					numer[1] += GMM_alpha * (double)RandomMap_G.at<double>(y, x - 1);
					numer[0] += GMM_alpha * (double)RandomMap_B.at<double>(y, x - 1);
					numer2[2] += GMM_alpha * ((double)RandomMap_R2.at<double>(y, x - 1) - (double)RandomMap_R.at<double>(y, x - 1));
					numer2[1] += GMM_alpha * ((double)RandomMap_G2.at<double>(y, x - 1) - (double)RandomMap_G.at<double>(y, x - 1));
					numer2[0] += GMM_alpha * ((double)RandomMap_B2.at<double>(y, x - 1) - (double)RandomMap_B.at<double>(y, x - 1));
					denom += GMM_alpha;
					denom2 += GMM_alpha;
					adjacent_pix_num++;
				}
				if (x + 1 < GMM_XSIZE)
				{
					numer[2] += GMM_alpha * (double)RandomMap_R.at<double>(y, x + 1);
					numer[1] += GMM_alpha * (double)RandomMap_G.at<double>(y, x + 1);
					numer[0] += GMM_alpha * (double)RandomMap_B.at<double>(y, x + 1);
					numer2[2] += GMM_alpha * ((double)RandomMap_R2.at<double>(y, x + 1) - (double)RandomMap_R.at<double>(y, x + 1));
					numer2[1] += GMM_alpha * ((double)RandomMap_G2.at<double>(y, x + 1) - (double)RandomMap_G.at<double>(y, x + 1));
					numer2[0] += GMM_alpha * ((double)RandomMap_B2.at<double>(y, x + 1) - (double)RandomMap_B.at<double>(y, x + 1));
					denom += GMM_alpha;
					denom2 += GMM_alpha;
					adjacent_pix_num++;
				}
				if (y > 0)
				{
					numer[2] += GMM_alpha * (double)RandomMap_R.at<double>(y - 1, x);
					numer[1] += GMM_alpha * (double)RandomMap_G.at<double>(y - 1, x);
					numer[0] += GMM_alpha * (double)RandomMap_B.at<double>(y - 1, x);
					numer2[2] += GMM_alpha * ((double)RandomMap_R2.at<double>(y - 1, x) - (double)RandomMap_R.at<double>(y - 1, x));
					numer2[1] += GMM_alpha * ((double)RandomMap_G2.at<double>(y - 1, x) - (double)RandomMap_G.at<double>(y - 1, x));
					numer2[0] += GMM_alpha * ((double)RandomMap_B2.at<double>(y - 1, x) - (double)RandomMap_B.at<double>(y - 1, x));
					denom += GMM_alpha;
					denom2 += GMM_alpha;
					adjacent_pix_num++;
				}
				if (y + 1 < GMM_YSIZE)
				{
					numer[2] += GMM_alpha * (double)RandomMap_R.at<double>(y + 1, x);
					numer[1] += GMM_alpha * (double)RandomMap_G.at<double>(y + 1, x);
					numer[0] += GMM_alpha * (double)RandomMap_B.at<double>(y + 1, x);
					numer2[2] += GMM_alpha * ((double)RandomMap_R2.at<double>(y + 1, x) - (double)RandomMap_R.at<double>(y + 1, x));
					numer2[1] += GMM_alpha * ((double)RandomMap_G2.at<double>(y + 1, x) - (double)RandomMap_G.at<double>(y + 1, x));
					numer2[0] += GMM_alpha * ((double)RandomMap_B2.at<double>(y + 1, x) - (double)RandomMap_B.at<double>(y + 1, x));
					denom += GMM_alpha;
					denom2 += GMM_alpha;
					adjacent_pix_num++;
				}
				numer2[2] += ((double)GMM_lambda + GMM_alpha * (double)adjacent_pix_num) * (double)RandomMap_R.at<double>(y, x);
				numer2[1] += ((double)GMM_lambda + GMM_alpha * (double)adjacent_pix_num) * (double)RandomMap_G.at<double>(y, x);
				numer2[0] += ((double)GMM_lambda + GMM_alpha * (double)adjacent_pix_num) * (double)RandomMap_B.at<double>(y, x);

				for (c = 0; c < 3; c++) {
					ave2[c] = numer2[c] / denom2;
					numer[c] += ave2[c] * GMM_gamma;
					ave[c] = numer[c] / denom;
					switch (c) {
					case 0:
						errorConvergence += fabs(RandomMap_B.at<double>(y, x) - (double)ave[c]);
						RandomMap_B.at<double>(y, x) = ave[c];
						RandomMap_B2.at<double>(y, x) = ave2[c];
						break;
					case 1:
						errorConvergence += fabs(RandomMap_G.at<double>(y, x) - (double)ave[c]);
						RandomMap_G.at<double>(y, x) = ave[c];
						RandomMap_G2.at<double>(y, x) = ave2[c];
						break;
					case 2:
						errorConvergence += fabs(RandomMap_R.at<double>(y, x) - (double)ave[c]);
						RandomMap_R.at<double>(y, x) = ave[c];
						RandomMap_R2.at<double>(y, x) = ave2[c];
						break;
					default:
						break;
					}
				}
			}
		}
		errorConvergence = (double)(errorConvergence / ((double)GMM_MAX_PIX * 3.0));

		if (errorConvergence < CONVERGE_HMRF_H) {
			HMRF_POST_flg = 0; // 収束成功
			break;
		}
	}

	// 出力画像
	double double_ave[3], double_ave2[3];
#pragma omp parallel for private(x, c)
	for (y = 0; y < GMM_YSIZE; y++) {
		for (x = 0; x < GMM_XSIZE; x++) {
			col_index = (y * GMM_XSIZE + x) * 3;
			for (int c = 0; c < 3; c++) {
				switch (c) {
				case 0:
					double_ave[c] = (double)RandomMap_B.at<double>(y, x);
					double_ave2[c] = (double)RandomMap_B2.at<double>(y, x);
					//double_ave[c] = ((double)((double)RandomMap_B.at<double>(y, x) + 1.0) / (double)2.0) * (double)imageMAX_INTENSE;
					//double_ave2[c] = ((double)((double)RandomMap_B2.at<double>(y, x) + 1.0) / (double)2.0) * (double)imageMAX_INTENSE;
					break;
				case 1:
					double_ave[c] = (double)RandomMap_G.at<double>(y, x);
					double_ave2[c] = (double)RandomMap_G2.at<double>(y, x);
					//double_ave[c] = ((double)((double)RandomMap_G.at<double>(y, x) + 1.0) / (double)2.0) * (double)imageMAX_INTENSE;
					//double_ave2[c] = ((double)((double)RandomMap_G2.at<double>(y, x) + 1.0) / (double)2.0) * (double)imageMAX_INTENSE;
					break;
				case 2:
					double_ave[c] = (double)RandomMap_R.at<double>(y, x);
					double_ave2[c] = (double)RandomMap_R2.at<double>(y, x);
					//double_ave[c] = ((double)((double)RandomMap_R.at<double>(y, x) + 1.0) / (double)2.0) * (double)imageMAX_INTENSE;
					//double_ave2[c] = ((double)((double)RandomMap_R2.at<double>(y, x) + 1.0) / (double)2.0) * (double)imageMAX_INTENSE;
					break;
				default:
					double_ave[c] = 0.0;
					double_ave2[c] = 0.0;
					cout << " ERROR! GaussSeidelMethod_HierarchicalMRF_POSTERIOR()" << endl;
					break;
				}

				if (double_ave[c] < 0.0) { double_ave[c] = 0.0; }
				else if (double_ave[c] > imageMAX_INTENSE) { double_ave[c] = imageMAX_INTENSE; }

				averageVector.data[col_index + c] = (double)double_ave[c];
				averageVector2.data[col_index + c] = (double)double_ave2[c];
				averageVec2.push_back(double_ave2[c]);
				//cout << " double_ave = " << (double)double_ave[2] << " , double_ave2 = " << (double)double_ave2[2] << endl;	// 確認用
				//cout << " averageVector = " << (double)averageVector.data[col_index + c] << " , averageVector2 = " << (double)averageVector2.data[col_index + c] << endl;	// 確認用
				//cout << " averageVec2 = " << (double)averageVec2[col_index + c] << "<-" << (double)double_ave2[c] << endl;
			}
		}
	}

	if (HMRF_POST_flg != 0) { cout << " GaussSeidelアルゴリズム 収束失敗 : errorConvergence = " << errorConvergence << endl; }
	//else { cout << " GaussSeidelアルゴリズム 収束成功" << endl; }
}
void GMM_HMRF::MaximumMapWEstimation() {
	int GS_flg = 1;	// 収束フラグ
	int x, y, c;

	double errorConvergence;
	double numer[3], denom, ave[3], Yi[3];
	int col_index;

	Map_W.clear();
	// 確率分布の初期化
	Mat RandomMap_B = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));	// 確率変数w
	Mat RandomMap_G = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));
	Mat RandomMap_R = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));

	for (int count = 0; count < MaxIteration_HMRF_H; count++) {
		errorConvergence = 0;
#pragma omp parallel for private(x, c, numer, denom, ave, Yi, col_index) reduction(+ : errorConvergence)
		for (y = 0; y < GMM_YSIZE; y++) {
			for (x = 0; x < GMM_XSIZE; x++) {
				col_index = (y * GMM_XSIZE + x) * 3;
				/*Yi[0] = (double)averageVector2.data[col_index + 0];
				Yi[1] = (double)averageVector2.data[col_index + 1];
				Yi[2] = (double)averageVector2.data[col_index + 2];*/
				Yi[0] = (double)averageVec2[col_index + 0];
				Yi[1] = (double)averageVec2[(int)(col_index + 1)];
				Yi[2] = (double)averageVec2[(int)(col_index + 2)];
				//cout << (double)Yi[2] << endl;	// 確認用
				for (int c = 0; c < 3; c++) {
					numer[c] = (double)Yi[c];
				}
				denom = GMM_lambda;

				if (x > 0) {
					numer[2] += GMM_alpha * (double)RandomMap_R.at<double>(y, x - 1);
					numer[1] += GMM_alpha * (double)RandomMap_G.at<double>(y, x - 1);
					numer[0] += GMM_alpha * (double)RandomMap_B.at<double>(y, x - 1);
					denom += GMM_alpha;
				}
				if (x + 1 < GMM_XSIZE)
				{
					numer[2] += GMM_alpha * (double)RandomMap_R.at<double>(y, x + 1);
					numer[1] += GMM_alpha * (double)RandomMap_G.at<double>(y, x + 1);
					numer[0] += GMM_alpha * (double)RandomMap_B.at<double>(y, x + 1);
					denom += GMM_alpha;
				}
				if (y > 0)
				{
					numer[2] += GMM_alpha * (double)RandomMap_R.at<double>(y - 1, x);
					numer[1] += GMM_alpha * (double)RandomMap_G.at<double>(y - 1, x);
					numer[0] += GMM_alpha * (double)RandomMap_B.at<double>(y - 1, x);
					denom += GMM_alpha;
				}
				if (y + 1 < GMM_YSIZE)
				{
					numer[2] += GMM_alpha * (double)RandomMap_R.at<double>(y + 1, x);
					numer[1] += GMM_alpha * (double)RandomMap_G.at<double>(y + 1, x);
					numer[0] += GMM_alpha * (double)RandomMap_B.at<double>(y + 1, x);
					denom += GMM_alpha;
				}

				for (c = 0; c < 3; c++) {
					ave[c] = numer[c] / denom;
					//cout << (double)ave[c] << endl;	// 確認用
					switch (c) {
					case 0:
						errorConvergence += fabs(RandomMap_B.at<double>(y, x) - (double)ave[c]);
						RandomMap_B.at<double>(y, x) = ave[c];
						break;
					case 1:
						errorConvergence += fabs(RandomMap_G.at<double>(y, x) - (double)ave[c]);
						RandomMap_G.at<double>(y, x) = ave[c];
						break;
					case 2:
						errorConvergence += fabs(RandomMap_R.at<double>(y, x) - (double)ave[c]);
						RandomMap_R.at<double>(y, x) = ave[c];
						break;
					default:
						break;
					}
				}
			}
		}
		errorConvergence = (double)(errorConvergence / ((double)GMM_MAX_PIX * 3.0));

		if (errorConvergence < CONVERGE_HMRF_H) {
			GS_flg = 0; // 収束成功
			break;
		}
	}

	// 出力画像
	double double_ave[3];
#pragma omp parallel for private(x, c)
	for (y = 0; y < GMM_YSIZE; y++) {
		for (x = 0; x < GMM_XSIZE; x++) {
			col_index = (y * GMM_XSIZE + x) * 3;
			for (int c = 0; c < 3; c++) {
				switch (c) {
				case 0:
					double_ave[c] = (double)RandomMap_B.at<double>(y, x);
					break;
				case 1:
					double_ave[c] = (double)RandomMap_G.at<double>(y, x);
					break;
				case 2:
					double_ave[c] = (double)RandomMap_R.at<double>(y, x);
					break;
				default:
					double_ave[c] = 0.0;
					cout << " ERROR! GaussSeidelMethod_MapW" << endl;
					break;
				}

				Map_W.push_back(double_ave[c]);
				MapW.data[col_index + c] = (double)double_ave[c];
				//cout << (double)double_ave[c] << endl;	// 確認用
			}
		}
	}

	//if (GS_flg != 0) { cout << " GSアルゴリズム 収束失敗 : errorConvergence = " << errorConvergence << endl; }
	//else { cout << " GSアルゴリズム 収束成功!" << endl; }
}
void GMM_HMRF::EstimatedParameter(double converge, int Max_Iteration) {
	int BP_flg = 1;	// 収束フラグ
	int x, y, c;

	int pix_index;
	double doubleIntensity;
	int c_E, c_M;
	const int Iteration_BPstep = 1/*Max_Iteration*/;	// 最大反復回数
	const int Iteration_Estimate = 10/*Max_Iteration*/;
	const double eps_BPstep = 0.1/*converge*/;			// 収束判定値
	const double eps_Estimate = 1.0e-9/*converge*/;

	double gamma_old, lambda_old, sigma2_old, alpha_old;
	double grad_lambda, grad_alpha, grad_gamma;
	double grad_lambda2, grad_alpha2, grad_gamma2;
	double errorE = 0.0, errorM = 0.0;
	double tmp1 =0.0, tmp2 = 0.0, tmp3 = 0.0;
	Mat calc_function1 = Mat(averageImage.size(), CV_64FC3);	// Kai_{i}
	vector<double> calc_func1;
#pragma omp parallel for private(x, c)
	for (y = 0; y < GMM_YSIZE; y++) {
		for (x = 0; x < GMM_XSIZE; x++) {
			for (c = 0; c < 3; c++) {
				calc_func1.push_back(tmp1);
			}
		}
	}

	gamma_old = GMM_gamma; lambda_old = GMM_lambda; sigma2_old = GMM_sigma2; alpha_old = GMM_alpha;  // パラメータ_old初期化
	averageImage.copyTo(averageVector);	 // 事後分布の平均ベクトル初期化
#pragma omp parallel for private(x, c)
	for (y = 0; y < GMM_YSIZE; y++) {
		for (x = 0; x < GMM_XSIZE; x++) {
			for (c = 0; c < 3; c++) {
				pix_index = (y * GMM_XSIZE + x) * 3 + c;
				calc_function1.data[pix_index] = 0.0;	// 関数1 初期化
			}
		}
	}

	// ループ
	for (c_E = 0; c_E < Iteration_BPstep; c_E++) {
		errorE = 0.0;

		// 事後分布のガウスザイデル法による推定
		MaximumPosteriorEstimation();

		// sigma2 の推定
		cout << " sigma2=" << GMM_sigma2 << "  =>  ";	// 確認用
		//CalculationFunction1_H(sigma2_old, alpha_old, lambda_old, gamma_old, imageK, eigenValue, calc_function1);
		tmp3 = CalculationFunction1_H_E(sigma2_old, alpha_old, lambda_old, gamma_old, imageK, EigenValue, calc_function1, calc_func1);
		tmp1 = 0.0, tmp2 = 0.0;
#pragma omp parallel for private(x, c) reduction(+ : tmp1, temp2)
		for (y = 0; y < GMM_YSIZE; y++) {
			for (x = 0; x < GMM_XSIZE; x++) {
				for (c = 0; c < 3; c++) {
					pix_index = (y * GMM_XSIZE + x) * 3 + c;
					//tmp1 += (double)calc_function1.data[pix_index] / ((double)GMM_MAX_PIX * 3.0);
					/*tmp3 = (double)calc_func1[pix_index];
					tmp1 += (double)tmp3 / ((double)GMM_MAX_PIX * 3.0);*/
					//tmp2 += pow((double)(averageVector.data[pix_index] - averageImage.data[pix_index]), 2) / ((double)GMM_MAX_PIX * 3.0);
					for (imgK = 0; imgK < imageK; imgK++) {
						//tmp2 += pow((double)(LIKELIHOOD[imgK].data[pix_index] - averageImage.data[pix_index]), 2) / ((double)GMM_MAX_PIX * 3.0 * (double)imageK);
						double TMP = (double)(LIKELIHOOD[imgK].data[pix_index] - LIKELIHOOD_Average[imgK]) - (double)(averageImage.data[pix_index] - averageImage_Average);
						//cout << (double)TMP << "=" << (double)(LIKELIHOOD[imgK].data[pix_index] - LIKELIHOOD_Average[imgK]) << "-" << (double)(averageImage.data[pix_index] - averageImage_Average) << endl;	// 確認用
						tmp2 += pow(TMP, 2) / ((double)GMM_MAX_PIX * 3.0 * (double)imageK);
					}
				}
			}
		}
		//cout << " tmp3 = " << (double)tmp3 << endl;	// 確認用
		tmp1 = (double)tmp3 / ((double)GMM_MAX_PIX * 3.0);
		GMM_sigma2 = tmp1 + tmp2;
		GMM_sigma = sqrt(GMM_sigma2);
		cout << "sigma2=" << GMM_sigma2 << " , sigma=" << GMM_sigma << endl;	// 確認用

		// alpha, lambda, gamma の推定
		for (c_M = 0; c_M < Iteration_Estimate; c_M++) {
			errorM = 0.0;

			// 事後分布のガウスザイデル法による推定
			MaximumPosteriorEstimation();
			// 周辺尤度wの更新
			MaximumMapWEstimation();
			//CalculationFunction1_H(sigma2_old, alpha_old, lambda_old, gamma_old, imageK, eigenValue, calc_function1);
			tmp3 = CalculationFunction1_H_E(sigma2_old, alpha_old, lambda_old, gamma_old, imageK, EigenValue, calc_function1, calc_func1);
			
			// 尤度関数の微分の計算
			//grad_lambda = (double)CalcFunction_lambda(GMM_lambda, GMM_alpha, GMM_gamma, eigenValue, calc_function1) / GMM_sigma2;
			//grad_alpha = (double)CalcFunction_alpha(GMM_lambda, GMM_alpha, GMM_gamma, eigenValue, calc_function1) / GMM_sigma2;
			//grad_gamma = (double)CalcFunction_gamma(GMM_lambda, GMM_alpha, GMM_gamma, eigenValue, calc_function1) / GMM_sigma2;
			grad_lambda = (double)CalcFunction_lambda_E(GMM_lambda, GMM_alpha, GMM_gamma, EigenValue, calc_function1, imageK, GMM_sigma2) / GMM_sigma2;
			grad_alpha = (double)CalcFunction_alpha_E(GMM_lambda, GMM_alpha, GMM_gamma, EigenValue, calc_function1, imageK, GMM_sigma2) / GMM_sigma2;
			grad_gamma = (double)CalcFunction_gamma_E(GMM_lambda, GMM_alpha, GMM_gamma, EigenValue, calc_function1, imageK, GMM_sigma2) / GMM_sigma2;
			//cout << "  " << (double)CalcFunction_lambda_E(GMM_lambda, GMM_alpha, GMM_gamma, EigenValue, calc_function1) << " , " << (double)CalcFunction_alpha_E(GMM_lambda, GMM_alpha, GMM_gamma, EigenValue, calc_function1) << " , " << (double)CalcFunction_gamma_E(GMM_lambda, GMM_alpha, GMM_gamma, EigenValue, calc_function1) << endl;	// 確認用
			//cout << "  grad_gamma=" << grad_gamma << " , grad_lambda=" << grad_lambda << " , grad_alpha=" << grad_alpha << endl;	// 確認用
			
			//grad_alpha2 = ((double)CalcFunction_alpha2(averageVector) / (double)imageK) + (((double)CalcFunction_alpha2(MapW) * (double)pow(GMM_gamma, 2)) / (double)imageK);
			grad_alpha2 = ((double)CalcFunction_alpha2(averageVector) / (double)imageK) + (((double)CalcFunction_alpha3(GMM_XSIZE, GMM_YSIZE, Map_W) * (double)pow(GMM_gamma, 2)) / (double)imageK);
			grad_lambda2 = 0.0, grad_gamma2 = 0.0;
			for (y = 0; y < GMM_YSIZE; y++) {
				for (x = 0; x < GMM_XSIZE; x++) {
					for (c = 0; c < 3; c++) {
						pix_index = (y * GMM_XSIZE + x) * 3 + c;
						//grad_lambda2 += (double)(pow((double)averageVector.data[pix_index], 2) / (double)imageK) + ((double)(pow((double)MapW.data[pix_index], 2) * GMM_gamma) / (double)imageK);
						grad_lambda2 += (double)(pow((double)averageVector.data[pix_index], 2) / (double)imageK) + ((double)(pow((double)Map_W[pix_index], 2) * GMM_gamma) / (double)imageK);
						grad_gamma2 += (double)(pow((double)averageVector2.data[pix_index], 2) / (double)imageK);
					}
				}
			}
			//cout << "  grad_gamma2=" << grad_gamma2 << " , grad_lambda2=" << grad_lambda2 << " , grad_alpha2=" << grad_alpha2 << endl;	// 確認用

			grad_lambda += grad_lambda2;
			grad_alpha += grad_alpha2;
			grad_gamma += grad_gamma2;

			grad_lambda /= (double)3.0 * (double)GMM_MAX_PIX * (double)2.0;
			grad_alpha /= (double)3.0 * (double)GMM_MAX_PIX * (double)2.0;
			grad_gamma /= (double)3.0 * (double)GMM_MAX_PIX * (double)2.0;

			doubleIntensity = (double)imageMAX_INTENSE;
			grad_gamma /= doubleIntensity;
			grad_lambda /= (doubleIntensity * doubleIntensity);
			grad_alpha /= doubleIntensity;
			//cout << " " << c_M << " : grad_gamma=" << grad_gamma << " , grad_lambda=" << grad_lambda << " , grad_alpha=" << grad_alpha << endl;	// 確認用

			// 感受率伝搬
			GMM_lambda += LearningRate_Lambda_H * grad_lambda;
			GMM_alpha += LearningRate_Alpha_H * grad_alpha;
			GMM_gamma += LearningRate_Gamma_H * grad_gamma;
			//cout << " += : gamma=" << (double)(LearningRate_Gamma * grad_gamma) << " , lambda=" << (double)(LearningRate_Lambda * grad_lambda) << " , alpha=" << (double)(LearningRate_Alpha * grad_alpha) << endl;	// 確認用
			if (GMM_lambda < 0) {
				GMM_lambda = 0;
				cout << " WARNING! Estimate_AlgorithmHGMRF(): Parameter(lambda) became under 0." << endl;
				break;
			}
			if (GMM_alpha < 0) {
				GMM_alpha = 0;
				cout << " WARNING! Estimate_AlgorithmHGMRF(): Parameter(alpha) became under 0." << endl;
				break;
			}
			if (GMM_gamma < 0) {
				GMM_gamma = 0;
				cout << " WARNING! Estimate_AlgorithmHGMRF(): Parameter(gamma) became under 0." << endl;
				break;
			}

			if (errorM < fabs(alpha_old - GMM_alpha)) {
				errorM = fabs(alpha_old - GMM_alpha);
			}
			if (errorM < fabs(lambda_old - GMM_lambda)) {
				errorM = fabs(lambda_old - GMM_lambda);
			}
			if (errorM < fabs(gamma_old - GMM_gamma)) {
				errorM = fabs(gamma_old - GMM_gamma);
			}
			cout << " errorM : gamma=" << fabs(gamma_old - GMM_gamma) << " , lambda=" << fabs(lambda_old - GMM_lambda) << " , alpha=" << fabs(alpha_old - GMM_alpha) << endl;	// 確認用
			//cout << " errorM = " << errorM << " : gamma=" << GMM_gamma << " , lambda=" << GMM_lambda << " , alpha=" << GMM_alpha << endl;	// 確認用

			gamma_old = GMM_gamma;
			lambda_old = GMM_lambda;
			alpha_old = GMM_alpha;

			if (errorM < eps_Estimate) {
				BP_flg = 0;
				break;
			}
		}

		if (GMM_sigma2 < 0) {
			GMM_sigma2 = 0;
			GMM_sigma = 0;
			cout << " WARNING! Estimate_AlgorithmHGMRF(): Parameter(sigma) became under 0." << endl;
			break;
		}

		errorE = fabs(sigma2_old - GMM_sigma2);
		sigma2_old = GMM_sigma2;

		if (errorE < eps_BPstep) {
			break;
		}
	}
	cout << " errorM = " << errorM << " , c_M = " << c_M << endl;	// 確認用
	calc_func1.clear();

	// パラメータ表示
	cout << "--- パラメータ (GMM_HMRF) ------------------------------------" << endl;
	cout << " 収束判定値: " << converge << endl;
	cout << " 学習率    :gamma  = " << LearningRate_Gamma_H << endl;
	cout << "            alpha  = " << LearningRate_Alpha_H << endl;
	cout << "            lambda = " << LearningRate_Lambda_H << endl;
	//cout << "            sigma  = " << LearningRate_Sigma_H << endl;
	cout << "-------------------------------------------------------------" << endl;

	if (BP_flg == 0) { cout << " BPアルゴリズム 収束成功" << endl; }
	else { cout << " BPアルゴリズム 収束失敗!" << endl; }
}


/* 関数 */
// パラメータを推定して画像修復を生成
void DenoisingHMRF(Mat& DenoiseImage, const vector<Mat>& noiseImage, const int K, const int maxIntensity) {
	double gamma, alpha, sigma, lambda;
	// パラメータの初期値設定
	alpha = ALPHA_HMRF_H;
	sigma = SIGMA_HMRF_H;
	lambda = LAMBDA_HMRF_H;
	gamma = GAMMA_HMRF_H;

	GMM_HMRF gmmHMRF = GMM_HMRF(K, maxIntensity, DenoiseImage, noiseImage, gamma, lambda, alpha, sigma);
	// グラフラプラシアンの固有地
	gmmHMRF.eigenValueGrid2D();
	// 平均画像生成と修復画像初期化
	gmmHMRF.CreateDoubleAverageImageMat();
	gmmHMRF.CreateDoubleAverageSquareImageMat();

	// 規格化規格化(0~1) ※修正必要
	//gmmHMRF.Standardization();
	// 中心化 ※修正必要
	//gmmHMRF.Centralization();

	// EMアルゴリズムによるパラメータ推定
	gmmHMRF.EstimatedParameter(CONVERGE_H, MAXIteration_H);

	// 事後分布のガウスザイデル法による推定
	gmmHMRF.MaximumPosteriorEstimation();

	// 規格化(0~1)もどし
	//gmmHMRF.ReturnStandardization();
	// 中心化もどし
	//gmmHMRF.ReturnCentralization();

	// 修復画像の生成
	gmmHMRF.CreateOutput();
	gmmHMRF.CreateOutputImage(DenoiseImage);

	// パラメータ表示
	cout << "--- パラメータ (DenoisingHMRF) ------------------------------" << endl;
	cout << " alpha  = " << gmmHMRF.GMM_alpha << "  <-  " << ALPHA_HMRF_H << endl;
	cout << " sigma  = " << gmmHMRF.GMM_sigma << "  <-  " << SIGMA_HMRF_H << endl;
	cout << " lambda = " << gmmHMRF.GMM_lambda << "  <-  " << LAMBDA_HMRF_H << endl;
	cout << " gamma  = " << gmmHMRF.GMM_gamma << "  <-  " << GAMMA_HMRF_H << endl;
	cout << "-------------------------------------------------------------" << endl;
}

#endif