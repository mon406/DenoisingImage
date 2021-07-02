#ifndef __INCLUDED_H_MakeNoiseImage__
#define __INCLUDED_H_MakeNoiseImage__

#include "main.h"


// K 枚の劣化画像の生成 (ガウスノイズ付加)
void GaussianNoiseImage(const int K, const Mat& ORIGINAL_IMG, vector<Mat>& NOISE_IMG) {
	NOISE_IMG.clear();
	Mat ORIGINAL_tmp;
	ORIGINAL_IMG.copyTo(ORIGINAL_tmp);

	for (int k = 0; k < K; k++) {
		Mat noise_image_K = Mat(ORIGINAL_IMG.size(), ORIGINAL_IMG.type());	// k番目の劣化画像
		Mat noise = Mat(ORIGINAL_IMG.size(), ORIGINAL_IMG.type());
		unsigned int now_time = (unsigned int)time(NULL);			// 乱数パターンの初期値は現在時刻
		srand(now_time);											// 乱数の種を設定(srand関数)
		/* 毎回異なる乱数が取得できる */
		double ORIGINAL_average = CalcAverage(ORIGINAL_tmp);
		//cout << (double)ORIGINAL_average << endl;	// 確認用
		randn(noise, NoiseMean + ORIGINAL_average, NoiseSigma);	// 乱数randn(生成Mat, 平均, 標準偏差)
		
		int x, y, c;
		int pix_index, number_tmp;
#pragma omp parallel for private(x, c)
		for (int y = 0; y < ORIGINAL_IMG.rows; y++) {
			for (int x = 0; x < ORIGINAL_IMG.cols; x++) {
				for (int c = 0; c < 3; c++) {
					pix_index = (y * ORIGINAL_IMG.cols + x) * 3 + c;
					number_tmp = (int)((double)ORIGINAL_IMG.data[pix_index] + (double)noise.data[pix_index] - (double)ORIGINAL_average);
					if (number_tmp < 0) { number_tmp = 0; }
					else if (number_tmp > MAX_INTENSE) { number_tmp = MAX_INTENSE; }
					//cout << (int)ORIGINAL_IMG.data[pix_index] << " -> ";	// 確認用
					noise_image_K.data[pix_index] = (uchar)number_tmp;
					//cout << (int)noise_image_K.data[pix_index] << endl;	// 確認用
				}
			}
		}
		//noise_image_K = ORIGINAL_IMG + noise;
		NOISE_IMG.push_back(noise_image_K);
	}
	cout << " ガウスノイズ : NoiseMean = " << NoiseMean << " , NoiseSigma = " << NoiseSigma << endl;	// 確認用
	cout << endl;
}

// 平均画像（RGB値）の生成
void CreateAverageImage(const int K, Mat& AVERAGE_IMG, const vector<Mat>& ALL_IMG, const double maxINTENSE)
{
	if (ALL_IMG[0].type() != CV_8UC3) {
		cout << "ERROR! CreateAverageImage : Input type is wrong." << endl;
	}
	else {
		Mat average_image_K = Mat(ALL_IMG[0].size(), CV_8UC3);	// k枚の画像の平均画像

		Vec3b color;
		uchar r, g, b;
		double rr = 0, gg = 0, bb = 0;
		int rrr = 0, ggg = 0, bbb = 0;
#pragma omp parallel for private(X_index, k, Vec, color[0], color[1], color[2], r, g, b, rr, gg, bb, rrr, ggg, bbb)
		for (int Y_index = 0; Y_index < average_image_K.rows; Y_index++) {
			for (int X_index = 0; X_index < average_image_K.cols; X_index++) {
				// k枚の画像の合計値(double値)
				rr = 0, gg = 0, bb = 0;
				for (int k = 0; k < K; k++) {
					color = ALL_IMG[k].at<Vec3b>(Y_index, X_index);	// ピクセル値（カラー）を取得
					r = color[2];	// R,G,B値に分解
					g = color[1];
					b = color[0];

					bb += (double)((double)b / maxINTENSE);
					gg += (double)((double)g / maxINTENSE);
					rr += (double)((double)r / maxINTENSE);
					//cout << " (r,g,b) : " << r << " , " << g << " , " << b << endl;	// 確認用
				}

				// k枚の画像の平均値
				bb = (double)bb / (double)K;
				gg = (double)gg / (double)K;
				rr = (double)rr / (double)K;
				bbb = (int)((double)bb * maxINTENSE);
				ggg = (int)((double)gg * maxINTENSE);
				rrr = (int)((double)rr * maxINTENSE);
				color[0] = (uchar)bbb;
				color[1] = (uchar)ggg;
				color[2] = (uchar)rrr;
				for (int Color_index = 0; Color_index < 3; Color_index++) {
					int check_color = (int)color[Color_index];
					if (check_color < 0) {
						cout << "WARNNING! CreateAverageImage : int_ave is under 0. (" << check_color << ")" << endl;
						color[Color_index] = 0;
					}
					else if (check_color > MAX_INTENSE) {
						cout << "WARNNING! CreateAverageImage : int_ave is over MAX_INTENSE. (" << check_color << ")" << endl;
						color[Color_index] = MAX_INTENSE;
					}
				}
				average_image_K.at<Vec3b>(Y_index, X_index) = color;
			}
		}

		average_image_K.copyTo(AVERAGE_IMG);
	}
}

#endif