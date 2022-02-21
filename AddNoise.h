#ifndef __INCLUDED_H_AddNoise__
#define __INCLUDED_H_AddNoise__

#include <time.h>
#include "main.h"

/* 関数宣言 */
void Add_Salt_and_pepper_Noise(Mat SrcImg, Mat DstImg);		// ゴマ塩（インパルス）ノイズ
void Add_Gauss_Noise(Mat SrcImg, Mat DstImg);				// ガウスノイズ


/* 関数 */
// ゴマ塩（インパルス）ノイズ付加
void Add_Salt_and_pepper_Noise(Mat SrcImg, Mat DstImg) {
	double amount = 0.004;  //0.4%にノイズを加える
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

// ガウスノイズ付加
void Add_Gauss_Noise(Mat SrcImg, Mat DstImg) {
	/* ガウスノイズ付加時のパラメータ */
	static double Noise_Mean = 0.0;	// ノイズの平均
	static double Noise_Sigma = 30;	// ノイズの標準偏差（2乗で分散）

	Mat noiseImg = Mat(SrcImg.size(), SrcImg.type());	// k番目の劣化画像
	Mat noise = Mat(SrcImg.size(), SrcImg.type());
	unsigned int now_time = (unsigned int)time(NULL);		// 乱数パターンの初期値は現在時刻
	srand(now_time);										// 乱数の種を設定(srand関数)
	/* 毎回異なる乱数が取得できる */
	double ORIGINAL_average = CalcAverage(SrcImg);
	randn(noise, Noise_Mean + ORIGINAL_average, Noise_Sigma);	// 乱数randn(生成Mat, 平均, 標準偏差)

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