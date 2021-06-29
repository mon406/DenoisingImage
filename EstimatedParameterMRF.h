#ifndef __INCLUDED_H_EstimatedParameterMRF__
#define __INCLUDED_H_EstimatedParameterMRF__

#include "main.h"


/* MRF�̃p�����[�^ */
double Converge_MRF = 1.0e-8;	// �p�����[�^����̎�������l
int MaxIteration_MRF = 100;		// �p�����[�^����̍ő唽����
const double LearningRate_mean = 1.0e-9;			// �w�K��
//const double LearningRate_alpha = 1.0e-6;
const double LearningRate_alpha = 1.0e-4;
const double LearningRate_lambda = 1.0e-13;
double h_MRF = 0.0;					// �p�����[�^
double SIGMA_MRF = 40;
double ALPHA_MRF = 1.0e-4;
double LAMBDA_MRF = 1.0e-7;


/* �֐�(�N���X���v�Z�p) */
// Mat ��_{(i,j)inV} 1/(lambda+(K/sigma2)+alpha*fhi) �̌v�Z
void CalculationFunction1(double& sigma2, double& alpha, double& lambda, int K, Mat& GuraphRap, Mat& X) {
	int x, y, c;
	const int imageSizeX = GuraphRap.cols;
	const int imageSizeY = GuraphRap.rows;
	int Function1_index;
	double function1_tmp;

#pragma omp parallel for private(x, c)
	for (y = 0; y < imageSizeY; y++) {
		for (x = 0; x < imageSizeX; x++) {
			for (c = 0; c < 3; c++) {
				Function1_index = (y * imageSizeX + x) * 3 + c;
				function1_tmp = lambda + ((double)K / sigma2) + alpha * (double)GuraphRap.data[Function1_index];
				X.data[Function1_index] = 1.0 / function1_tmp;
			}
		}
	}
}

// ��_{(i,j)inV} 1/(lambda+(K/sigma2)+alpha*fhi)+��_{(i,j)inV} 1/(lambda+alpha*fhi) �̌v�Z
double CalculationSumFunction1(double& sigma2, double& alpha, double& lambda, int K, Mat& GuraphRap) {
	int x, y, c;
	const int imageSizeX = GuraphRap.cols;
	const int imageSizeY = GuraphRap.rows;
	int Function1_index;
	double function1_tmp1, function1_tmp2;
	double function1_answer = 0.0;

#pragma omp parallel for private(x, c)
	for (y = 0; y < imageSizeY; y++) {
		for (x = 0; x < imageSizeX; x++) {
			for (c = 0; c < 3; c++) {
				Function1_index = (y * imageSizeX + x) * 3 + c;
				function1_tmp1 = lambda + alpha * (double)GuraphRap.data[Function1_index];
				function1_tmp2 = function1_tmp1 + ((double)K / sigma2);
				function1_answer += (1.0 / function1_tmp1) + (1.0 / function1_tmp2);
			}
		}
	}

	return function1_answer;
}

// �摜�̕��ϒl�̌v�Z
double CalculationFunction2(Mat& X) {
	int x, y, c;
	const int imageSizeX = X.cols;
	const int imageSizeY = X.rows;
	const int SUM_Pix = imageSizeX * imageSizeY * 3;
	int Function2_index;
	double function2_answer = 0.0;
	//double center_X = CalcAverage(X);

#pragma omp parallel for private(x, c) reduction(+ : function2_answer)
	for (y = 0; y < imageSizeY; y++) {
		for (x = 0; x < imageSizeX; x++) {
			for (c = 0; c < 3; c++) {
				Function2_index = (y * imageSizeX + x) * 3 + c;
				function2_answer += (double)X.data[Function2_index] / (double)SUM_Pix;
				//function2_answer += ((double)X.data[Function2_index] - (double)center_X) / (double)SUM_Pix;
			}
		}
	}

	return function2_answer;
}

// �摜�̑��a�̌v�Z
double CalculationSum(Mat& X) {
	int x, y, c;
	const int imageSizeX = X.cols;
	const int imageSizeY = X.rows;
	const int SUM_Pix = 3;
	int FuncSum_index;
	double funcSum_answer = 0.0;

#pragma omp parallel for private(x, c) reduction(+ : funcSum_answer)
	for (y = 0; y < imageSizeY; y++) {
		for (x = 0; x < imageSizeX; x++) {
			for (c = 0; c < 3; c++) {
				FuncSum_index = (y * imageSizeX + x) * 3 + c;
				funcSum_answer += (double)X.data[FuncSum_index] / (double)SUM_Pix;
			}
		}
	}

	return funcSum_answer;
}


/* �N���X */
// GMM MarkovRandomField
class GMM_MRF : public GMM {
private:
	int imgK;
	int MRFx, MRFy, MRFc;
	int MRF_index;
public:
	double CONVERGE_MRF = Converge_MRF;			// ��������l
	int MAXIteration_MRF = MaxIteration_MRF;	// �ő唽����

	GMM_MRF(int, int, Mat&, vector<Mat>, double, double, double, double);
	void CreateOutput();				// �C���摜�̐���
	void MaximumPosteriorEstimation();	// ���㕪�z�̃K�E�X�U�C�f���@�ɂ��MAP����
	void EstimatedParameter(int, int);	// EM�A���S���Y���ɂ��p�����[�^����
};
GMM_MRF::GMM_MRF(int K, int max_intense, Mat& ImageDst, vector<Mat> likelihood, double h, double lambda, double alpha, double sigma) {
	imageK = K;
	imageMAX_INTENSE = max_intense;
	GMM_XSIZE = ImageDst.cols;
	GMM_YSIZE = ImageDst.rows;
	GMM_MAX_PIX = GMM_XSIZE * GMM_YSIZE;
	LIKELIHOOD.clear();
	Mat lilelihood_tmp;
#pragma omp parallel for private(MRFy, MRFx, MRFc)
	for (imgK = 0; imgK < imageK; imgK++) {
		lilelihood_tmp = Mat(Size(ImageDst.cols, ImageDst.rows), CV_64FC3);
		for (MRFy = 0; MRFy < ImageDst.rows; MRFy++) {
			for (MRFx = 0; MRFx < ImageDst.cols; MRFx++) {
				for (MRFc = 0; MRFc < 3; MRFc++) {
					MRF_index = (MRFy * ImageDst.cols + MRFx) * 3 + MRFc;
					lilelihood_tmp.data[MRF_index] = (double)likelihood[imgK].data[MRF_index];
				}
			}
		}
		LIKELIHOOD.push_back(lilelihood_tmp);
	}

	POSTERIOR = Mat(Size(ImageDst.cols, ImageDst.rows), CV_64FC3);
#pragma omp parallel for private(MRFx, MRFc)
	for (MRFy = 0; MRFy < ImageDst.rows; MRFy++) {
		for (MRFx = 0; MRFx < ImageDst.cols; MRFx++) {
			for (MRFc = 0; MRFc < 3; MRFc++) {
				MRF_index = (MRFy * ImageDst.cols + MRFx) * 3 + MRFc;
				POSTERIOR.data[MRF_index] = (double)ImageDst.data[MRF_index];
				//cout << "  " << (double)POSTERIOR.data[MRF_index] << " <- " << (double)ImageDst.data[MRF_index] << endl;	// �m�F�p
			}
		}
	}

	averageVector = Mat::zeros(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	averageImage = Mat::zeros(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	averageSquareImage = Mat(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	eigenValue = Mat(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);

	GMM_mean = h;
	GMM_lambda = lambda;
	GMM_alpha = alpha;
	GMM_sigma = sigma;
	GMM_sigma2 = GMM_sigma * GMM_sigma;
}
void GMM_MRF::CreateOutput() {
	//POSTERIOR.copyTo(averageVector);
	//LIKELIHOOD[1].copyTo(averageVector);
	//averageImage.copyTo(averageVector);

	double color_checker;
#pragma omp parallel for private(MRFx, MRFc, color_checker)
	for (MRFy = 0; MRFy < GMM_YSIZE; MRFy++) {
		for (MRFx = 0; MRFx < GMM_XSIZE; MRFx++) {
			for (MRFc = 0; MRFc < 3; MRFc++) {
				MRF_index = (MRFy * GMM_XSIZE + MRFx) * 3 + MRFc;
				color_checker = (double)(averageVector.data[MRF_index]);
				if (color_checker < 0) {
					POSTERIOR.data[MRF_index] = (double)0;
					cout << " WARNING! GMM_MRF::CreateOutput() : POSTERIOR < 0" << endl;	// �m�F�p
				}
				else if (color_checker > imageMAX_INTENSE) {
					POSTERIOR.data[MRF_index] = (double)imageMAX_INTENSE;
					cout << " WARNING! GMM_MRF::CreateOutput() : POSTERIOR > MAX_INTENSE" << endl;	// �m�F�p
				}
				else {
					POSTERIOR.data[MRF_index] = (double)color_checker;
					//cout << "  " << (double)POSTERIOR.data[MRF_index] << endl;	// �m�F�p
				}
			}
		}
	}
}
void GMM_MRF::MaximumPosteriorEstimation() {
	int MRF_POST_flg = 1; // ��������t���O
	int x, y, c;

	double errorConvergence = 0.0;
	double numer[3], denom, ave[3], Yi[3];
	int col_index;
	Mat RandomMap_B = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));	// �m���ϐ�
	Mat RandomMap_G = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));
	Mat RandomMap_R = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));
	//averageImage.copyTo(averageVector);	 // ���㕪�z�̕��σx�N�g��������
#pragma omp parallel for private(x, c)
	for (y = 0; y < GMM_YSIZE; y++) {
		for (x = 0; x < GMM_XSIZE; x++) {
			col_index = (y * GMM_XSIZE + x) * 3;
			Yi[2] = (double)averageImage.data[col_index + 2];
			Yi[1] = (double)averageImage.data[col_index + 1];
			Yi[0] = (double)averageImage.data[col_index];
			Yi[2] = ((double)(Yi[2] * 2.0) / (double)imageMAX_INTENSE) - 1.0;	// (-1)~1 ���K��
			Yi[1] = ((double)(Yi[1] * 2.0) / (double)imageMAX_INTENSE) - 1.0;
			Yi[0] = ((double)(Yi[0] * 2.0) / (double)imageMAX_INTENSE) - 1.0;
			RandomMap_R.at<double>(y, x) = (double)Yi[2];
			RandomMap_G.at<double>(y, x) = (double)Yi[1];
			RandomMap_B.at<double>(y, x) = (double)Yi[0];
			//cout << " Yi[2] = " << (double)Yi[2] << " , RandomMap_R = " << (double)RandomMap_R.at<double>(y, x) << endl;	// �m�F�p
		}
	}

	for (int count = 0; count < MAXIteration_MRF; count++) {
		errorConvergence = 0;
#pragma omp parallel for private(x, numer, denom, ave, c) reduction(+ : errorConvergence)
		for (y = 0; y < GMM_YSIZE; y++) {
			for (x = 0; x < GMM_XSIZE; x++) {
				col_index = (y * GMM_XSIZE + x) * 3;
				Yi[2] = (double)averageImage.data[col_index + 2];
				Yi[1] = (double)averageImage.data[col_index + 1];
				Yi[0] = (double)averageImage.data[col_index];
				Yi[2] = ((double)(Yi[2] * 2.0) / (double)imageMAX_INTENSE) - 1.0;	// (-1)~1 ���K��
				Yi[1] = ((double)(Yi[1] * 2.0) / (double)imageMAX_INTENSE) - 1.0;
				Yi[0] = ((double)(Yi[0] * 2.0) / (double)imageMAX_INTENSE) - 1.0;
				for (c = 0; c < 3; c++) {
					numer[c] = GMM_mean + (double)(Yi[c] / GMM_sigma2) * (double)imageK;
				}
				denom = GMM_lambda + ((double)imageK / GMM_sigma2);

				if (x > 0) {
					//col_index = (y * GMM_XSIZE + x - 1) * 3;
					//numer[2] += GMM_alpha * (double)averageVector.data[col_index + 2];
					//numer[1] += GMM_alpha * (double)averageVector.data[col_index + 1];
					//numer[0] += GMM_alpha * (double)averageVector.data[col_index];
					/*numer[2] += GMM_alpha * (double)RandomMap.data[col_index + 2];
					numer[1] += GMM_alpha * (double)RandomMap.data[col_index + 1];
					numer[0] += GMM_alpha * (double)RandomMap.data[col_index];*/
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
					col_index = (y * GMM_XSIZE + x) * 3;
					//errorConvergence += fabs(averageVector.data[col_index + c] - (double)ave[c]);
					//averageVector.data[col_index + c] = ave[c];
					/*errorConvergence += fabs(RandomMap.data[col_index + c] - (double)ave[c]);
					RandomMap.data[col_index + c] = ave[c];*/
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

		if (errorConvergence < CONVERGE_MRF) {
			MRF_POST_flg = 0; // ��������
			break;
		}
	}

	// �o�͉摜
	double double_ave[3];
#pragma omp parallel for private(x, c, double_ave)
	for (y = 0; y < GMM_YSIZE; y++) {
		for (x = 0; x < GMM_XSIZE; x++) {
			col_index = (y * GMM_XSIZE + x) * 3;
			for (int c = 0; c < 3; c++) {
				//double_ave[c] = ((double)((double)averageVector.data[col_index + c] + 1.0) / (double)2.0) * (double)imageMAX_INTENSE;
				//double_ave[c] = ((double)((double)RandomMap.data[col_index + c] + 1.0) / (double)2.0) * (double)imageMAX_INTENSE;
				switch (c) {
				case 0:
					double_ave[c] = (double)RandomMap_B.at<double>(y, x);
					double_ave[c] = ((double)((double)RandomMap_B.at<double>(y, x) + 1.0) / (double)2.0) * (double)imageMAX_INTENSE;
					break;
				case 1:
					double_ave[c] = (double)RandomMap_G.at<double>(y, x);
					double_ave[c] = ((double)((double)RandomMap_G.at<double>(y, x) + 1.0) / (double)2.0) * (double)imageMAX_INTENSE;
					break;
				case 2:
					double_ave[c] = (double)RandomMap_R.at<double>(y, x);
					double_ave[c] = ((double)((double)RandomMap_R.at<double>(y, x) + 1.0) / (double)2.0) * (double)imageMAX_INTENSE;
					break;
				default:
					double_ave[c] = 0.0;
					cout << " ERROR! GaussSeidelMethod_HMRF_POSTERIOR()" << endl;
					break;
				}

				/*if (double_ave[c] < 0.0) { double_ave[c] = 0.0; }
				else if (double_ave[c] > imageMAX_INTENSE) { double_ave[c] = imageMAX_INTENSE; }*/

				averageVector.data[col_index + c] = (double)double_ave[c];
				//cout << " double_ave[c] = " << (double)double_ave[c] << " , averageVector = " << (double)averageVector.data[col_index + c] << endl;	// �m�F�p
			}
		}
	}

	if (MRF_POST_flg != 0) { cout << " GaussSeidel�A���S���Y�� �������s! : errorConvergence = " << (double)errorConvergence << endl; }
	//else { cout << " GaussSeidel�A���S���Y�� ��������" << endl; }
	//cout << "GaussSeidel : mean=" << (double)GMM_mean << ",alpha=" << (double)GMM_alpha << ",lambda=" << (double)GMM_lambda << ",sigma2=" << (double)GMM_sigma2 << endl;	// �m�F�p
}
void GMM_MRF::EstimatedParameter(int converge, int Max_Iteration) {
	int EM_flg = 1;	// �����t���O
	int x, y, c;

	int pix_index;
	double doubleIntensity;
	int c_EM, c_M;
	const int Iteration_EMstep = 1/*Max_Iteration*/;	// �ő唽����
	const int Iteration_Mstep = 10/*MAXIteration_MRF*/;
	const double eps_EMstep = 0.1/*converge*/;			// ��������l
	const double eps_Mstep = 1.0e-5/*converge*/;

	double h_old, lambda_old, sigma2_old, alpha_old;
	double grad_h, grad_lambda, grad_alpha;
	double grad_h_post, grad_lambda_post, grad_alpha_post;
	double errorEM = 0.0, errorM = 0.0;
	double tmp1, tmp2, tmp3;
	Mat calc_function1 = Mat(averageImage.size(), CV_64FC3);

	h_old = GMM_mean; lambda_old = GMM_lambda; sigma2_old = GMM_sigma2; alpha_old = GMM_alpha;  // �p�����[�^_old������
	averageImage.copyTo(averageVector);	 // ���㕪�z�̕��σx�N�g��������
#pragma omp parallel for private(x, c)
	for (y = 0; y < GMM_YSIZE; y++) {
		for (x = 0; x < GMM_XSIZE; x++) {
			for (c = 0; c < 3; c++) {
				pix_index = (y * GMM_XSIZE + x) * 3 + c;
				calc_function1.data[pix_index] = 0.0;	// �֐�1 ������
			}
		}
	}

	// EM ���[�v
	for (c_EM = 0; c_EM < Iteration_EMstep; c_EM++) {
		errorEM = 0;

		// ���㕪�z�̃K�E�X�U�C�f���@�ɂ�鐄��
		MaximumPosteriorEstimation();

		// sigma2 �̐���
		/*double center_aveVec = CalcAverage(averageVector);
		double center_aveImg = CalcAverage(averageImage);*/
		cout << " sigma2=" << GMM_sigma2 << "  =>  ";	// �m�F�p
		CalculationFunction1(sigma2_old, alpha_old, lambda_old, imageK, eigenValue, calc_function1);
		tmp1 = 0.0, tmp2 = 0.0;
#pragma omp parallel for private(x, c) reduction(+ : tmp1, temp2)
		for (y = 0; y < GMM_YSIZE; y++) {
			for (x = 0; x < GMM_XSIZE; x++) {
				for (c = 0; c < 3; c++) {
					pix_index = (y * GMM_XSIZE + x) * 3 + c;
					tmp1 += (double)calc_function1.data[pix_index] / ((double)GMM_MAX_PIX * 3.0);
					tmp2 += pow((double)(averageVector.data[pix_index] - averageImage.data[pix_index]), 2) / ((double)GMM_MAX_PIX * 3.0 * (double)imageK);
					/*tmp2 = ((double)averageVector.data[pix_index] - center_aveVec) - ((double)averageImage.data[pix_index] - center_aveImg);
					tmp2 += pow((double)tmp2, 2) / ((double)GMM_MAX_PIX * 3.0 * (double)imageK);*/
				}
			}
		}
		GMM_sigma2 = tmp1 + tmp2;
		GMM_sigma = sqrt(GMM_sigma2);
		cout << "sigma2=" << GMM_sigma2 << " , sigma=" << GMM_sigma << endl;	// �m�F�p
		//cout << " += : h=" << (double)GMM_mean << " , lambda=" << (double)GMM_lambda << " , alpha=" << (double)GMM_alpha << endl;	// �m�F�p

		// h, alpha, lambda �̐���
		for (c_M = 0; c_M < Iteration_Mstep; c_M++) {
			errorM = 0;

			// ���㕪�z�̃K�E�X�U�C�f���@�ɂ�鐄��
			MaximumPosteriorEstimation();

			grad_h_post = -((double)GMM_MAX_PIX * GMM_mean) / GMM_lambda;
			tmp1 = CalculationFunction2(averageImage);
			grad_h = ((double)GMM_MAX_PIX * ((h_old + ((double)imageK / sigma2_old) * tmp1)) / (lambda_old + ((double)imageK / sigma2_old)));
			grad_h += grad_h_post;

			grad_alpha_post = -(((double)GMM_MAX_PIX - 1.0) / (double)GMM_MAX_PIX) * (1.0 / GMM_alpha);
			grad_alpha = ((double)GMM_MAX_PIX - 1.0) / (2.0 * GMM_alpha);
			grad_alpha += grad_alpha_post;

			tmp3 = GMM_mean;
			if (tmp3 > 0) { tmp3 = pow(GMM_mean, 2); }
			else { tmp3 = 1.0; }
			grad_lambda_post = ((double)GMM_MAX_PIX * tmp3) / pow(GMM_lambda, 2);
			grad_lambda = CalculationSum(averageSquareImage) + CalculationSumFunction1(sigma2_old, alpha_old, lambda_old, imageK, eigenValue);
			grad_lambda += grad_lambda_post;

			grad_h /= (double)3.0 * (double)GMM_MAX_PIX;
			grad_lambda /= (double)3.0 * (double)GMM_MAX_PIX;
			grad_alpha /= (double)3.0 * (double)GMM_MAX_PIX;
			/*grad_h /= (double)3.0 * (double)GMM_MAX_PIX;
			grad_lambda /= (double)3.0 * (double)GMM_MAX_PIX * (double)2.0;
			grad_alpha /= (double)3.0 * (double)GMM_MAX_PIX * (double)2.0;*/

			doubleIntensity = (double)imageMAX_INTENSE;
			grad_h /= doubleIntensity;
			grad_lambda /= (doubleIntensity * doubleIntensity);
			grad_alpha /= doubleIntensity;
			//cout << " " << c_M << " : grad_h=" << grad_h << " , grad_lambda=" << grad_lambda << " , grad_alpha=" << grad_alpha << endl;	// �m�F�p

			// ���z�㏸�@
			GMM_mean += LearningRate_mean * grad_h;
			GMM_lambda += LearningRate_lambda * grad_lambda;
			GMM_alpha += LearningRate_alpha * grad_alpha;
			//cout << " += : h=" << (double)(LearningRate_h * grad_h) << " , lambda=" << (double)(LearningRate_lambda * grad_lambda) << " , alpha=" << (double)(LearningRate_alpha * grad_alpha) << endl;	// �m�F�p
			if (GMM_mean < 0) {
				cout << " WARNING! EM_AlgorithmMRF(): Parameter(h) became under 0. (" << GMM_mean << ")" << endl;
				GMM_mean = 0.0;
				break;
			}
			if (GMM_lambda < 0) {
				cout << " WARNING! EM_AlgorithmMRF(): Parameter(lambda) became under 0. (" << GMM_lambda << ")" << endl;
				GMM_lambda = 0.0;
				break;
			}
			if (GMM_alpha < 0) {
				cout << " WARNING! EM_AlgorithmMRF(): Parameter(alpha) became under 0. (" << GMM_alpha << ")" << endl;
				GMM_alpha = 0.0;
				break;
			}

			if (errorM < fabs(h_old - GMM_mean)) {
				errorM = fabs(h_old - GMM_mean);
			}
			if (errorM < fabs(alpha_old - GMM_alpha)) {
				errorM = fabs(alpha_old - GMM_alpha);
			}
			if (errorM < fabs(lambda_old - GMM_lambda)) {
				errorM = fabs(lambda_old - GMM_lambda);
			}
			cout << " errorM: mean=" << fabs(h_old - GMM_mean) << " , lambda=" << fabs(lambda_old - GMM_lambda) << " , alpha=" << fabs(alpha_old - GMM_alpha) << endl;	// �m�F�p
			//cout << " errorM = " << errorM << " : mean=" << GMM_mean << " , lambda=" << GMM_lambda << " , alpha=" << GMM_alpha << endl;	// �m�F�p

			h_old = GMM_mean;
			lambda_old = GMM_lambda;
			alpha_old = GMM_alpha;

			if (errorM < eps_Mstep) {
				EM_flg = 0;
				break;
			}
		}

		errorEM = fabs(sigma2_old - GMM_sigma2);
		sigma2_old = GMM_sigma2;

		if (errorEM < eps_EMstep) {
			break;
		}
	}
	cout << " errorM = " << errorM << " , c_M = " << c_M << endl;	// �m�F�p

	// �p�����[�^�\��
	cout << "--- �p�����[�^ (GMM_MRF) ------------------------------------" << endl;
	cout << " ��������l: " << converge << endl;
	cout << " �w�K��    :mean   = " << LearningRate_mean << endl;
	cout << "            alpha  = " << LearningRate_alpha << endl;
	cout << "            lambda = " << LearningRate_lambda << endl;
	cout << "-------------------------------------------------------------" << endl;

	if (EM_flg == 0) { cout << " EM�A���S���Y�� ��������" << endl; }
	else { cout << " EM�A���S���Y�� �������s!" << endl; }
}


/* �֐� */
// �p�����[�^�𐄒肵�ĉ摜�C���𐶐�
void DenoisingMRF(Mat& DenoiseImage, const vector<Mat>& noiseImage, const int K, const int maxIntensity) {
	double h, alpha, sigma, lambda;
	// �p�����[�^�̏����l�ݒ�
	h = h_MRF;
	alpha = ALPHA_MRF;
	sigma = SIGMA_MRF;
	lambda = LAMBDA_MRF;

	GMM_MRF gmmMRF = GMM_MRF(K, maxIntensity, DenoiseImage, noiseImage, h, lambda, alpha, sigma);
	// �K�i��(0~1) ���C���K�v
	//gmmMRF.Standardization();
	// ���S�� ���C���K�v
	//gmmMRF.Centralization();

	// �O���t���v���V�A���̌ŗL�n
	gmmMRF.eigenValueGrid2D();
	// ���ω摜�����ƏC���摜������
	gmmMRF.CreateDoubleAverageImageMat();
	gmmMRF.CreateDoubleAverageSquareImageMat();

	// EM�A���S���Y���ɂ��p�����[�^����
	gmmMRF.EstimatedParameter(Converge_MRF, MaxIteration_MRF);

	// ���㕪�z�̃K�E�X�U�C�f���@�ɂ�鐄��
	gmmMRF.MaximumPosteriorEstimation();

	// �K�i��(0~1)���ǂ�
	//gmmMRF.ReturnStandardization();
	// ���S�����ǂ�
	//gmmMRF.ReturnCentralization();

	// �C���摜�̐���
	gmmMRF.CreateOutput();
	gmmMRF.CreateOutputImage(DenoiseImage);

	// �p�����[�^�\��
	cout << "--- �p�����[�^ (DenoisingMRF) -------------------------------" << endl;
	cout << " h      = " << gmmMRF.GMM_mean << "  <-  " << h_MRF << endl;
	cout << " alpha  = " << gmmMRF.GMM_alpha << "  <-  " << ALPHA_MRF << endl;
	cout << " sigma  = " << gmmMRF.GMM_sigma << "  <-  " << SIGMA_MRF << endl;
	cout << " lambda = " << gmmMRF.GMM_lambda << "  <-  " << LAMBDA_MRF << endl;
	cout << "-------------------------------------------------------------" << endl;
}

#endif