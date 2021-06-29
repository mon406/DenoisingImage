#ifndef __INCLUDED_H_OutputDataToExcel__
#define __INCLUDED_H_OutputDataToExcel__

#include "main.h"
#define _USE_MATH_DEFINES	// ���l���Z�萔����`���ꂽ�w�b�_�t�@�C���̓ǂݍ���

FILE* fp; // FILE�|�C���^�̐錾
int IMAGE_NUM[] = { 1,2,3,4,6,8,10,15,20,25,30 };	// �摜���� K
Mat Image_dst_sub;			// �w��o�͕�C�摜(�m�C�Y�摜)
Mat Image_dst_average_sub;	// �w��o�͕�C�摜(�m�C�Y���ω摜)
Mat Image_dst_MRF_sub;		// �w��o�͕�C�摜(�m�C�Y�����摜MRF)
Mat Image_dst_HMRF_sub;		// �w��o�͕�C�摜(�m�C�Y�����摜HMRF)

void MSE_PSNR_SSIM_Output(Mat& Original, Mat& Inpaint, double& MSE_result, double& PSNR_result, double& SSIM_result);
void SSIMcalculation(double& ssim, Mat& image_1, Mat& image_2);


// �摜������ς��ďC�����x�̕]�����s���A�G�N�Z���ɕۑ�����
void Inpainting_and_OutputToExcel() {
	/* Step1. CSV�t�@�C�����J�� */
	double x, y;
	x = 2.5;
	y = sqrt(x);

	fopen_s(&fp, "result_data.csv", "w");
	/* CSV�t�@�C���ɍ��ڕۑ� */
	fprintf(fp, "K,noise_MSE,noise_PSNR,noise_SSIM,MRF_MSE,MRF_PSNR,MRF_SSIM,HMRF_MSE,HMRF_PSNR,HMRF_SSIM,average_MSE,average_PSNR,average_SSIM,timeMRF,timeHMRF\n");

	//--- �摜���� ------------------------------------------------------------------------
	clock_t Start, Start2, End, End2;	// �������ԕ\���p
	double Time_difference = 0.0, Time_difference2 = 0.0;

	vector<Mat> K_Image_dst;		// k���̗򉻉摜
	int nowK = 1;
	double resultMSE = 0.0, resultPSNR = 0.0, resultSSIM = 0.0;
	cout << "#########################################################################" << endl;
	for (int K_number = 0; K_number < 11; K_number++) {
		/* �K�E�X�m�C�Y�t�� */
		K_Image_dst.clear();
		nowK = IMAGE_NUM[K_number];
		cout << nowK << "���̗򉻉摜�����c" << endl;		// ���s�m�F�p
		GaussianNoiseImage(nowK, Image_src, K_Image_dst);
		K_Image_dst[0].copyTo(Image_dst);
		CreateAverageImage(nowK, Image_dst_average, K_Image_dst, MAX_INTENSE);	// ���ω摜����
		Image_dst.copyTo(Image_dst_MRF);
		Image_dst.copyTo(Image_dst_HMRF);

		/* MRF(�}���R�t�m����)�ɂ��m�C�Y���� */
		cout << "�p�����[�^���肠��(MRF)�̃m�C�Y�����c" << endl;	// ���s�m�F�p
		Start = clock();
		DenoisingMRF(Image_dst_MRF, K_Image_dst, nowK, MAX_INTENSE);
		End = clock();
		Time_difference = (double)End - (double)Start;
		double time = static_cast<double>(Time_difference) / CLOCKS_PER_SEC * 1000.0;
		cout << "time : " << time << "[ms]" << endl;
		cout << endl;

		/* HMRF(�K�w�^�}���R�t�m����)�ɂ��m�C�Y���� */
		cout << "�p�����[�^���肠��(HMRF)�̃m�C�Y�����c" << endl;	// ���s�m�F�p
		Start2 = clock();
		//DenoisingHMRF(Image_dst_HMRF, K_Image_dst, nowK, MAX_INTENSE);
		End2 = clock();
		Time_difference2 = (double)End2 - (double)Start2;
		double time2 = static_cast<double>(Time_difference2) / CLOCKS_PER_SEC * 1000.0;
		cout << "time : " << time2 << "[ms]" << endl;
		cout << endl;


		/* Step2. �f�[�^�̏������� */
		fprintf(fp, "%d,", nowK);	//CSV�t�@�C���ɕۑ�

		cout << "�m�C�Y�摜 �� ���摜" << endl;			// ���s�m�F�p
		MSE_PSNR_SSIM_Output(Image_src, Image_dst, resultMSE, resultPSNR, resultSSIM);
		fprintf(fp, "%g,%g,%g,", resultMSE, resultPSNR, resultSSIM);		//CSV�t�@�C���ɏ㏑���ۑ�

		cout << "MRF�C���摜 �� ���摜" << endl;		// ���s�m�F�p
		MSE_PSNR_SSIM_Output(Image_src, Image_dst_MRF, resultMSE, resultPSNR, resultSSIM);
		fprintf(fp, "%g,%g,%g,", resultMSE, resultPSNR, resultSSIM);		//CSV�t�@�C���ɏ㏑���ۑ�

		cout << "HMRF�C���摜 �� ���摜" << endl;		// ���s�m�F�p
		MSE_PSNR_SSIM_Output(Image_src, Image_dst_HMRF, resultMSE, resultPSNR, resultSSIM);
		fprintf(fp, "%g,%g,%g,", resultMSE, resultPSNR, resultSSIM);		//CSV�t�@�C���ɏ㏑���ۑ�

		cout << "�m�C�Y���ω摜 �� ���摜" << endl;		// ���s�m�F�p
		MSE_PSNR_SSIM_Output(Image_src, Image_dst_average, resultMSE, resultPSNR, resultSSIM);
		fprintf(fp, "%g,%g,%g,", resultMSE, resultPSNR, resultSSIM);		//CSV�t�@�C���ɏ㏑���ۑ�

		fprintf(fp, "%g,%g\n", time, time2);	//CSV�t�@�C���ɕۑ�

		/* �ŏI�o�͉摜�̎w�� */
		if (nowK == 1) {
			Image_dst.copyTo(Image_dst_sub);
			Image_dst_average.copyTo(Image_dst_average_sub);
			Image_dst_MRF.copyTo(Image_dst_MRF_sub);
			Image_dst_HMRF.copyTo(Image_dst_HMRF_sub);
		}
		cout << "#########################################################################" << endl;
	}
	//-----------------------------------------------------------------------------------------

	/* Step3. CSV�t�@�C������� */
	fclose(fp);

	/* �ŏI�o�͉摜 */
	Image_dst_sub.copyTo(Image_dst);
	Image_dst_average_sub.copyTo(Image_dst_average);
	Image_dst_MRF_sub.copyTo(Image_dst_MRF);
	Image_dst_HMRF_sub.copyTo(Image_dst_HMRF);
}

// MSE&PSNR&SSIM�ɂ��摜�]��
void MSE_PSNR_SSIM_Output(Mat& Original, Mat& Inpaint, double& MSE_result, double& PSNR_result, double& SSIM_result) {
	double MSE, PSNR, SSIM;
	Mat beforeIMG, afterIMG;
	Original.copyTo(beforeIMG);
	Inpaint.copyTo(afterIMG);

	double MSE_sum = 0.0;	// MSE�l
	double image_cost;		// ��f�l�̍���
	int compare_size, color_ind;
	int occ_pix_count = 0;

	/* MSE�v�Z(RGB) */
	for (int i = 0; i < Original.rows; i++) {
		for (int j = 0; j < Original.cols; j++) {
			image_cost = 0.0;
			color_ind = i * Original.cols * 3 + j * 3;
			for (int k = 0; k < 3; k++) {
				image_cost += pow((int)Inpaint.data[color_ind] - (int)Original.data[color_ind], 2.0);
				color_ind++;
			}
			MSE_sum += (double)image_cost;
			occ_pix_count++;
		}
	}
	compare_size = occ_pix_count * 3;
	MSE = (double)MSE_sum / (double)compare_size;

	/* PSNR�v�Z */
	PSNR = 20 * (double)log10(MAX_INTENSE) - 10 * (double)log10(MSE);

	/* SSIM�v�Z */
	SSIMcalculation(SSIM, beforeIMG, afterIMG);

	/* �]�����ʕ\�� */
	cout << "--- �]�� ------------------------------------------" << endl;
	cout << " MSE  : " << MSE << endl;
	cout << " PSNR : " << PSNR << endl;
	cout << " SSIM : " << SSIM << endl;
	cout << "---------------------------------------------------" << endl;
	cout << endl;
	/* �]�����ʂ��o�� */
	MSE_result = MSE;
	PSNR_result = PSNR;
	SSIM_result = SSIM;
}
// SSIM�Z�o
void SSIMcalculation(double& ssim, Mat& image_1, Mat& image_2) {
	const double C1 = pow(0.01 * 255, 2), C2 = pow(0.03 * 255, 2);

	Mat I1, I2;
	image_1.convertTo(I1, CV_32F);	// cannot calculate on one byte large values
	image_2.convertTo(I2, CV_32F);
	Mat I2_2 = I2.mul(I2);			// I2^2
	Mat I1_2 = I1.mul(I1);			// I1^2
	Mat I1_I2 = I1.mul(I2);			// I1 * I2

	Mat mu1, mu2;   // PRELIMINARY COMPUTING
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);

	Mat sigma1_2, sigma2_2, sigma12;
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	Mat ssim_map;
	divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
	Scalar mssim = mean(ssim_map); // mssim = average of ssim map

	/* SSIM����(RGB) */
	double SSIM;
	SSIM = (double)mssim[0] + (double)mssim[1] + (double)mssim[2];
	ssim = (double)SSIM / 3.0;
}

#endif