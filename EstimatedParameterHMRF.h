#ifndef __INCLUDED_H_EstimatedParameterHMRF__
#define __INCLUDED_H_EstimatedParameterHMRF__

#include "main.h"


/* MRF�̃p�����[�^ */
double CONVERGE_H = 1.0e-8;	// �p�����[�^����̎�������l
int MAXIteration_H = 100;		// �p�����[�^����̍ő唽����
//const double LearningRate_Sigma_H = 1.0e-6;		// �w�K��
const double LearningRate_Alpha_H = 1.0e-6;
const double LearningRate_Lambda_H = 1.0e-13;
const double LearningRate_Gamma_H = 1.0e-9;
double SIGMA_HMRF_H = 40;		// �p�����[�^
double ALPHA_HMRF_H = 1.0e-4;
//double ALPHA_HMRF_H = 1.0;				// (�K�i����)
double LAMBDA_HMRF_H = 1.0e-7;
double GAMMA_HMRF_H = 1.0e-4;

/* �֐�(�N���X���v�Z�p) */
// �v�T�C�֐��i�g���֐��j
double Psi_H(double& eigenvalue, double& lambda, double& alpha, double& gamma) {
	double calc_result, calc_numer, calc_denom;
	calc_numer = lambda + alpha * eigenvalue;
	calc_denom = calc_numer + gamma;
	calc_numer = pow(calc_numer, 2);
	calc_result = calc_numer / calc_denom;

	return calc_result;
}
// �J�C�֐�
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


// Mat ��_{(i,j)inV} 1/(lambda+(K/sigma2)+alpha*fhi) �̌v�Z
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
				function1_tmp = Kai_original_H(K, sigma2, number_tmp, lambda, alpha, gamma);
				X.data[Function1_index] = 1.0 / function1_tmp;
			}
		}
	}
}

// �p�����[�^���莞�̊֐�


// ��_{E}Xi �̌v�Z
//double CalcDoubleSumAdjective_H(int pixX, int pixY, int pixC, Mat& X) {
//	double function_answer = 0.0;
//	int function_index;
//	if (pixX > 0) {
//		function_index = (pixY * X.cols + (pixX - 1)) * 3 + pixC;
//		function_answer += X.data[function_index];
//	}
//	if (pixX + 1 < X.cols) {
//		function_index = (pixY * X.cols + (pixX + 1)) * 3 + pixC;
//		function_answer += X.data[function_index];
//	}
//	if (pixY > 0) {
//		function_index = ((pixY - 1) * X.cols + pixX) * 3 + pixC;
//		function_answer += X.data[function_index];
//	}
//	if (pixY + 1 < X.rows) {
//		function_index = ((pixY + 1) * X.cols + pixX) * 3 + pixC;
//		function_answer += X.data[function_index];
//	}
//
//	return function_answer;
//}
//// |a(i)| �̌v�Z
//int CalcIntNumberAdjective_H(int pixX, int pixY, Mat& X) {
//	int adjective_counter = 0;
//	if (pixX > 0) { adjective_counter++; }
//	if (pixX + 1 < X.cols) { adjective_counter++; }
//	if (pixY > 0) { adjective_counter++; }
//	if (pixY + 1 < X.rows) { adjective_counter++; }
//
//	return adjective_counter;
//}


/* �N���X */
// GMM HierarchicalMarkovRandomField
class GMM_HierarchicalMRF : public GMM {
private:
	int x, y, c;
	int imgK;
public:
	double CONVERGE_HMRF_H = 1.0e-8;		// ��������l
	int MaxIteration_HMRF_H = 1000;		// �ő唽����
	Mat averageVector2;

	GMM_HierarchicalMRF(int, int, Mat&, vector<Mat>, double, double, double, double);
	void Standardization();				// �K�i��
	void returnStandardization();		// �K�i�����ǂ�
	void CreateOutput();				// �C���摜�̐���
	void MaximumPosteriorEstimation();	// ���㕪�z�̃K�E�X�U�C�f���@�ɂ��MAP����
	void EstimatedParameter(double, int);	// EM�A���S���Y���ɂ��p�����[�^����
};
GMM_HierarchicalMRF::GMM_HierarchicalMRF(int K, int max_intense, Mat& ImageDst, vector<Mat> likelihood, double gamma, double lambda, double alpha, double sigma) {
	imageK = K;
	imageMAX_INTENSE = max_intense;
	GMM_XSIZE = ImageDst.cols;
	GMM_YSIZE = ImageDst.rows;
	GMM_MAX_PIX = GMM_XSIZE * GMM_YSIZE;
	LIKELIHOOD.clear();
#pragma omp parallel
	for (imgK = 0; imgK < imageK; imgK++) {
		LIKELIHOOD.push_back(likelihood[imgK]);
	}

	ImageDst.copyTo(POSTERIOR);
	averageVector = Mat::zeros(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	averageImage = Mat::zeros(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	averageSquareImage = Mat(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	eigenValue = Mat(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);
	averageVector2 = Mat::zeros(Size(GMM_XSIZE, GMM_YSIZE), CV_64FC3);

	GMM_gamma = gamma;
	GMM_lambda = lambda;
	GMM_alpha = alpha;
	GMM_sigma = sigma;
	GMM_sigma2 = GMM_sigma * GMM_sigma;
}
void GMM_HierarchicalMRF::Standardization() {
	int indexK;

	doubleMatStandardization(averageVector);
	doubleMatStandardization(averageVector2);
	doubleMatStandardization(averageImage);
	doubleMatStandardization(averageSquareImage);
	for (indexK = 0; indexK < imageK; indexK++) {
		doubleMatStandardization(LIKELIHOOD[indexK]);
	}
}
void GMM_HierarchicalMRF::returnStandardization() {
	int indexK;

	doubleMatReturnStandardization(POSTERIOR);
	doubleMatReturnStandardization(averageVector);
	doubleMatReturnStandardization(averageVector2);
	doubleMatReturnStandardization(averageImage);
	doubleMatReturnStandardization(averageSquareImage);
	for (indexK = 0; indexK < imageK; indexK++) {
		doubleMatReturnStandardization(LIKELIHOOD[indexK]);
	}
}
void GMM_HierarchicalMRF::CreateOutput() {
	int DenoisingHMRF_index;
	double color_checker;
#pragma omp parallel for private(x, c, color_checker)
	for (y = 0; y < GMM_YSIZE; y++) {
		for (x = 0; x < GMM_XSIZE; x++) {
			for (c = 0; c < 3; c++) {
				DenoisingHMRF_index = (y * GMM_XSIZE + x) * 3;
				color_checker = (double)(averageVector.data[DenoisingHMRF_index + c]);
				if (color_checker < 0) {
					POSTERIOR.data[DenoisingHMRF_index + c] = (uchar)0;
				}
				else if (color_checker > imageMAX_INTENSE) {
					POSTERIOR.data[DenoisingHMRF_index + c] = (uchar)imageMAX_INTENSE;
				}
				else {
					POSTERIOR.data[DenoisingHMRF_index + c] = (uchar)color_checker;
				}
			}
		}
	}
}
void GMM_HierarchicalMRF::MaximumPosteriorEstimation() {
	int HMRF_POST_flg = 1; // ��������t���O
	double errorConvergence;
	double numer[3], denom, ave[3], Yi[3];
	double numer2[3], denom2, ave2[3];
	int adjacent_pix_num;
	int col_index;

	// �m�����z�̏�����
	Mat RandomMap_B = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));	// �m���ϐ�m
	Mat RandomMap_G = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));
	Mat RandomMap_R = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));
	Mat RandomMap_B2 = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));	// �m���ϐ�myu
	Mat RandomMap_G2 = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));
	Mat RandomMap_R2 = Mat(averageImage.size(), CV_64F, Scalar::all(0.0));
	//averageImage.copyTo(averageVector);	 // ���㕪�z�̕��σx�N�g��������

	for (int count = 0; count < MaxIteration_HMRF_H; count++) {
		errorConvergence = 0;
#pragma omp parallel for private(x, numer, denom, ave, numer2, denom2, ave2, c) reduction(+ : errorConvergence)
		for (y = 0; y < GMM_YSIZE; y++) {
			for (x = 0; x < GMM_XSIZE; x++) {
				col_index = (y * GMM_XSIZE + x) * 3;
				for (int c = 0; c < 3; c++) {
					Yi[c] = (double)averageImage.data[col_index + c];
				}
				//Yi[2] = ((double)(Yi[2] * 2.0) / (double)imageMAX_INTENSE) - 1.0;	// (-1)~1 ���K��
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
					numer2[2] += (double)RandomMap_R2.at<double>(y, x - 1) - (double)RandomMap_R.at<double>(y, x - 1);
					numer2[1] += (double)RandomMap_G2.at<double>(y, x - 1) - (double)RandomMap_G.at<double>(y, x - 1);
					numer2[0] += (double)RandomMap_B2.at<double>(y, x - 1) - (double)RandomMap_B.at<double>(y, x - 1);
					denom += GMM_alpha;
					denom2 += GMM_alpha;
					adjacent_pix_num++;
				}
				if (x + 1 < GMM_XSIZE)
				{
					numer[2] += GMM_alpha * (double)RandomMap_R.at<double>(y, x + 1);
					numer[1] += GMM_alpha * (double)RandomMap_G.at<double>(y, x + 1);
					numer[0] += GMM_alpha * (double)RandomMap_B.at<double>(y, x + 1);
					numer2[2] += (double)RandomMap_R2.at<double>(y, x + 1) - (double)RandomMap_R.at<double>(y, x + 1);
					numer2[1] += (double)RandomMap_G2.at<double>(y, x + 1) - (double)RandomMap_G.at<double>(y, x + 1);
					numer2[0] += (double)RandomMap_B2.at<double>(y, x + 1) - (double)RandomMap_B.at<double>(y, x + 1);
					denom += GMM_alpha;
					denom2 += GMM_alpha;
					adjacent_pix_num++;
				}
				if (y > 0)
				{
					numer[2] += GMM_alpha * (double)RandomMap_R.at<double>(y - 1, x);
					numer[1] += GMM_alpha * (double)RandomMap_G.at<double>(y - 1, x);
					numer[0] += GMM_alpha * (double)RandomMap_B.at<double>(y - 1, x);
					numer2[2] += (double)RandomMap_R2.at<double>(y - 1, x) - (double)RandomMap_R.at<double>(y - 1, x);
					numer2[1] += (double)RandomMap_G2.at<double>(y - 1, x) - (double)RandomMap_G.at<double>(y - 1, x);
					numer2[0] += (double)RandomMap_B2.at<double>(y - 1, x) - (double)RandomMap_B.at<double>(y - 1, x);
					denom += GMM_alpha;
					denom2 += GMM_alpha;
					adjacent_pix_num++;
				}
				if (y + 1 < GMM_YSIZE)
				{
					numer[2] += GMM_alpha * (double)RandomMap_R.at<double>(y + 1, x);
					numer[1] += GMM_alpha * (double)RandomMap_G.at<double>(y + 1, x);
					numer[0] += GMM_alpha * (double)RandomMap_B.at<double>(y + 1, x);
					numer2[2] += (double)RandomMap_R2.at<double>(y + 1, x) - (double)RandomMap_R.at<double>(y + 1, x);
					numer2[1] += (double)RandomMap_G2.at<double>(y + 1, x) - (double)RandomMap_G.at<double>(y + 1, x);
					numer2[0] += (double)RandomMap_B2.at<double>(y + 1, x) - (double)RandomMap_B.at<double>(y + 1, x);
					denom += GMM_alpha;
					denom2 += GMM_alpha;
					adjacent_pix_num++;
				}
				numer2[2] += (double)adjacent_pix_num * (double)RandomMap_R.at<double>(y, x);
				numer2[1] += (double)adjacent_pix_num * (double)RandomMap_G.at<double>(y, x);
				numer2[0] += (double)adjacent_pix_num * (double)RandomMap_B.at<double>(y, x);
				/*numer2[2] += ((double)GMM_lambda + (double)GMM_alpha * (double)adjacent_pix_num) * (double)RandomMap_R.at<double>(y, x);
				numer2[1] += ((double)GMM_lambda + (double)GMM_alpha * (double)adjacent_pix_num) * (double)RandomMap_G.at<double>(y, x);
				numer2[0] += ((double)GMM_lambda + (double)GMM_alpha * (double)adjacent_pix_num) * (double)RandomMap_B.at<double>(y, x);*/
				denom2 /= GMM_alpha;

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
			HMRF_POST_flg = 0; // ��������
			break;
		}
	}

	// �o�͉摜
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
				//cout << " double_ave = " << (double)double_ave[2] << " , double_ave2 = " << (double)double_ave2[2] << endl;	// �m�F�p
			}
		}
	}

	if (HMRF_POST_flg != 0) { cout << " GaussSeidel�A���S���Y�� �������s : errorConvergence = " << errorConvergence << endl; }
	//else { cout << " GaussSeidel�A���S���Y�� ��������" << endl; }
}
void GMM_HierarchicalMRF::EstimatedParameter(double converge, int Max_Iteration) {
	int BP_flg = 1;	// �����t���O

	int pix_index;
	double doubleIntensity;
	int c_E, c_M;
	const int Iteration_BPstep = 2/*Max_Iteration*/;	// �ő唽����
	const int Iteration_Estimate = Max_Iteration;
	const double eps_BPstep = 0.1/*converge*/;			// ��������l
	const double eps_Estimate = converge;

	double gamma_old, lambda_old, sigma2_old, alpha_old;
	double grad_sigma, grad_lambda, grad_alpha, grad_gamma;
	double grad_lambda2, grad_alpha2, grad_gamma2;
	double errorE = 0.0, errorM = 0.0;
	double tmp1, tmp2, tmp3;
	Mat calc_function1 = Mat(averageImage.size(), CV_64FC3);

	gamma_old = GMM_gamma; lambda_old = GMM_lambda; sigma2_old = GMM_sigma2; alpha_old = GMM_alpha;  // �p�����[�^_old������
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

	// ���[�v
	for (c_E = 0; c_E < Iteration_BPstep; c_E++) {
		errorE = 0.0;

		// ���㕪�z�̃K�E�X�U�C�f���@�ɂ�鐄��
		MaximumPosteriorEstimation();

		// sigma2 �̐���
		cout << " sigma2=" << GMM_sigma2 << "  =>  ";	// �m�F�p
		CalculationFunction1_H(sigma2_old, alpha_old, lambda_old, gamma_old, imageK, eigenValue, calc_function1);
		tmp1 = 0.0, tmp2 = 0.0;
#pragma omp parallel for private(x, c) reduction(+ : tmp1, temp2)
		for (y = 0; y < GMM_YSIZE; y++) {
			for (x = 0; x < GMM_XSIZE; x++) {
				for (c = 0; c < 3; c++) {
					pix_index = (y * GMM_XSIZE + x) * 3 + c;
					tmp1 += (double)calc_function1.data[pix_index] / ((double)GMM_MAX_PIX * 3.0);
					tmp2 += pow((double)(averageVector.data[pix_index] - averageImage.data[pix_index]), 2) / ((double)GMM_MAX_PIX * 3.0 * (double)imageK);
				}
			}
		}
		GMM_sigma2 = tmp1 + tmp2;
		GMM_sigma = sqrt(GMM_sigma2);
		cout << "sigma2=" << GMM_sigma2 << " , sigma=" << GMM_sigma << endl;	// �m�F�p

		// alpha, lambda, gamma �̐���
		for (c_M = 0; c_M < Iteration_Estimate; c_M++) {
			errorM = 0.0;

			// �ޓx�֐��̔����̌v�Z
			grad_lambda = 0.0;
			grad_alpha = 0.0;
			grad_gamma = 0.0;

			grad_lambda /= (double)3.0 * (double)GMM_MAX_PIX * (double)2.0;
			grad_alpha /= (double)3.0 * (double)GMM_MAX_PIX * (double)2.0;
			grad_gamma /= (double)3.0 * (double)GMM_MAX_PIX * (double)2.0;

			doubleIntensity = (double)imageMAX_INTENSE;
			grad_gamma /= doubleIntensity;
			grad_lambda /= (doubleIntensity * doubleIntensity);
			grad_alpha /= doubleIntensity;
			//cout << " " << c_M << " : grad_gamma=" << grad_gamma << " , grad_lambda=" << grad_lambda << " , grad_alpha=" << grad_alpha << endl;	// �m�F�p

			// ���󗦓`��
			GMM_lambda += LearningRate_Lambda_H * grad_lambda;
			GMM_alpha += LearningRate_Alpha_H * grad_alpha;
			GMM_gamma += LearningRate_Gamma_H * grad_gamma;
			//cout << " += : gamma=" << (double)(LearningRate_Gamma * grad_gamma) << " , lambda=" << (double)(LearningRate_Lambda * grad_lambda) << " , alpha=" << (double)(LearningRate_Alpha * grad_alpha) << endl;	// �m�F�p
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
			//cout << " errorM : gamma=" << fabs(gamma_old - GMM_gamma) << " , lambda=" << fabs(lambda_old - GMM_lambda) << " , alpha=" << fabs(alpha_old - GMM_alpha) << endl;	// �m�F�p
			//cout << " errorM = " << errorM << " : gamma=" << GMM_gamma << " , lambda=" << GMM_lambda << " , alpha=" << GMM_alpha << endl;	// �m�F�p

			gamma_old = GMM_gamma;
			lambda_old = GMM_lambda;
			alpha_old = GMM_alpha;

			if (errorM < eps_BPstep) {
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
	cout << " errorM = " << errorM << " , c_M = " << c_M << endl;	// �m�F�p

	// �p�����[�^�\��
	cout << "--- �p�����[�^ (GMM_HMRF) ------------------------------------" << endl;
	cout << " ��������l: " << converge << endl;
	cout << " �w�K��    :gamma  = " << LearningRate_Gamma_H << endl;
	cout << "            alpha  = " << LearningRate_Alpha_H << endl;
	cout << "            lambda = " << LearningRate_Lambda_H << endl;
	//cout << "            sigma  = " << LearningRate_Sigma_H << endl;
	cout << "-------------------------------------------------------------" << endl;

	if (BP_flg == 0) { cout << " BP�A���S���Y�� ��������" << endl; }
	else { cout << " BP�A���S���Y�� �������s!" << endl; }
}


/* �֐� */
// �p�����[�^�𐄒肵�ĉ摜�C���𐶐�
void DenoisingHMRF(Mat& DenoiseImage, const vector<Mat>& noiseImage, const int K, const int maxIntensity) {
	double gamma, alpha, sigma, lambda;
	// �p�����[�^�̏����l�ݒ�
	alpha = ALPHA_HMRF_H;
	sigma = SIGMA_HMRF_H;
	lambda = LAMBDA_HMRF_H;
	gamma = GAMMA_HMRF_H;

	GMM_HierarchicalMRF gmmHierarchicalMRF = GMM_HierarchicalMRF(K, maxIntensity, DenoiseImage, noiseImage, gamma, lambda, alpha, sigma);
	// �O���t���v���V�A���̌ŗL�n
	gmmHierarchicalMRF.eigenValueGrid2D();
	// ���ω摜�����ƏC���摜������
	gmmHierarchicalMRF.CreateDoubleAverageImageMat();
	gmmHierarchicalMRF.CreateDoubleAverageSquareImageMat();

	// �K�i��(-1~1)
	//gmmHierarchicalMRF.Standardization();

	// EM�A���S���Y���ɂ��p�����[�^����
	gmmHierarchicalMRF.EstimatedParameter(CONVERGE_H, MAXIteration_H);

	// ���㕪�z�̃K�E�X�U�C�f���@�ɂ�鐄��
	gmmHierarchicalMRF.MaximumPosteriorEstimation();

	// �K�i��(-1~1)���ǂ�
	//gmmHierarchicalMRF.returnStandardization();

	// �C���摜�̐���
	gmmHierarchicalMRF.CreateOutput();
	gmmHierarchicalMRF.CreateOutputImage(DenoiseImage);

	// �p�����[�^�\��
	cout << "--- �p�����[�^ (DenoisingHMRF) ------------------------------" << endl;
	cout << " alpha  = " << gmmHierarchicalMRF.GMM_alpha << "  <-  " << ALPHA_HMRF_H << endl;
	cout << " sigma  = " << gmmHierarchicalMRF.GMM_sigma << "  <-  " << SIGMA_HMRF_H << endl;
	cout << " lambda = " << gmmHierarchicalMRF.GMM_lambda << "  <-  " << LAMBDA_HMRF_H << endl;
	cout << " gamma  = " << gmmHierarchicalMRF.GMM_gamma << "  <-  " << GAMMA_HMRF_H << endl;
	cout << "-------------------------------------------------------------" << endl;
}

#endif