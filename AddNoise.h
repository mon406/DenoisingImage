#ifndef __INCLUDED_H_AddNoise__
#define __INCLUDED_H_AddNoise__

#include <time.h>
#include "main.h"

/* �֐��錾 */
void Add_Salt_and_pepper_Noise(Mat SrcImg, Mat DstImg);		// �S�}���i�C���p���X�j�m�C�Y
void Add_Gauss_Noise(Mat SrcImg, Mat DstImg);				// �K�E�X�m�C�Y


/* �֐� */
// �S�}���i�C���p���X�j�m�C�Y�t��
void Add_Salt_and_pepper_Noise(Mat SrcImg, Mat DstImg) {
	double amount = 0.004;  //0.4%�Ƀm�C�Y��������
	cv::Mat dst = SrcImg.clone();
	int width = dst.size().width;
	int height = dst.size().height;
	for (int y = 0; y < height; y++) {
		cv::Vec3b* src = dst.ptr<cv::Vec3b>(y);
		for (int x = 0; x < width; x++) {
			if ((double)rand() / RAND_MAX <= amount) {
				src[x][0] = 255;
				src[x][1] = 255;
				src[x][2] = 255;
			}
			if ((double)rand() / RAND_MAX <= amount) {
				src[x][0] = 0;
				src[x][1] = 0;
				src[x][2] = 0;
			}
		}
	}
	dst.copyTo(DstImg);
}

// �K�E�X�m�C�Y�t��
void Add_Gauss_Noise(Mat SrcImg, Mat DstImg) {
	/* �K�E�X�m�C�Y�t�����̃p�����[�^ */
	static double Noise_Mean = 0.0;	// �m�C�Y�̕���
	static double Noise_Sigma = 30;	// �m�C�Y�̕W���΍��i2��ŕ��U�j

	Mat noiseImg = Mat(SrcImg.size(), SrcImg.type());	// k�Ԗڂ̗򉻉摜
	Mat noise = Mat(SrcImg.size(), SrcImg.type());
	unsigned int now_time = (unsigned int)time(NULL);		// �����p�^�[���̏����l�͌��ݎ���
	srand(now_time);										// �����̎��ݒ�(srand�֐�)
	/* ����قȂ闐�����擾�ł��� */
	double ORIGINAL_average = CalcAverage(SrcImg);
	randn(noise, Noise_Mean + ORIGINAL_average, Noise_Sigma);	// ����randn(����Mat, ����, �W���΍�)

	int x, y, c;
	int pix_index, number_tmp;
#pragma omp parallel for private(x, c)
	for (int y = 0; y < SrcImg.rows; y++) {
		for (int x = 0; x < SrcImg.cols; x++) {
			for (int c = 0; c < 3; c++) {
				pix_index = (y * SrcImg.cols + x) * 3 + c;
				number_tmp = (int)((double)SrcImg.data[pix_index] + (double)noise.data[pix_index] - (double)ORIGINAL_average);

				if (number_tmp < 0) { number_tmp = 0; }
				else if (number_tmp > MAX_INTENSE) { number_tmp = MAX_INTENSE; }
				noiseImg.data[pix_index] = (uchar)number_tmp;
			}
		}
	}

	noiseImg.copyTo(DstImg);
}


#endif