#ifndef __INCLUDED_H_MakeNoiseImage__
#define __INCLUDED_H_MakeNoiseImage__

#include "main.h"

/* �K�E�X�m�C�Y�t�����̃p�����[�^ */
static double NoiseMean = 0.0;	// �m�C�Y�̕���
static double NoiseSigma = 30;	// �m�C�Y�̕W���΍��i2��ŕ��U�j


// K ���̗򉻉摜�̐��� (�K�E�X�m�C�Y�t��)
void GaussianNoiseImage(const int K, const Mat& ORIGINAL_IMG, vector<Mat>& NOISE_IMG) {
	NOISE_IMG.clear();

	for (int k = 0; k < K; k++) {
		Mat noise_image_K;	// k�Ԗڂ̗򉻉摜
		Mat noise = Mat(ORIGINAL_IMG.size(), ORIGINAL_IMG.type());
		unsigned int now_time = (unsigned int)time(NULL);			// �����p�^�[���̏����l�͌��ݎ���
		srand(now_time);											// �����̎��ݒ�(srand�֐�)
		/* ����قȂ闐�����擾�ł��� */
		randn(noise, NoiseMean, NoiseSigma);	// ����randn(����Mat, ����, �W���΍�)
		noise_image_K = ORIGINAL_IMG + noise;
		NOISE_IMG.push_back(noise_image_K);
	}
	cout << " �K�E�X�m�C�Y : NoiseMean = " << NoiseMean << " , NoiseSigma = " << NoiseSigma << endl;	// �m�F�p
	cout << endl;
}

// ���ω摜�iRGB�l�j�̐���
void CreateAverageImage(const int K, Mat& AVERAGE_IMG, const vector<Mat>& ALL_IMG, const double maxINTENSE)
{
	if (ALL_IMG[0].type() != CV_8UC3) {
		cout << "ERROR! CreateAverageImage : Input type is wrong." << endl;
	}
	else {
		Mat average_image_K = Mat(ALL_IMG[0].size(), CV_8UC3);	// k���̉摜�̕��ω摜

		Vec3b color;
		uchar r, g, b;
		double rr = 0, gg = 0, bb = 0;
		int rrr = 0, ggg = 0, bbb = 0;
#pragma omp parallel for private(X_index, k, Vec, color[0], color[1], color[2], r, g, b, rr, gg, bb, rrr, ggg, bbb)
		for (int Y_index = 0; Y_index < average_image_K.rows; Y_index++) {
			for (int X_index = 0; X_index < average_image_K.cols; X_index++) {
				// k���̉摜�̍��v�l(double�l)
				rr = 0, gg = 0, bb = 0;
				for (int k = 0; k < K; k++) {
					color = ALL_IMG[k].at<Vec3b>(Y_index, X_index);	// �s�N�Z���l�i�J���[�j���擾
					r = color[2];	// R,G,B�l�ɕ���
					g = color[1];
					b = color[0];

					bb += (double)((double)b / maxINTENSE);
					gg += (double)((double)g / maxINTENSE);
					rr += (double)((double)r / maxINTENSE);
					//cout << " (r,g,b) : " << r << " , " << g << " , " << b << endl;	// �m�F�p
				}

				// k���̉摜�̕��ϒl
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