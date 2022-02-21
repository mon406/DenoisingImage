#ifndef __INCLUDED_H_OutputDataToExcel__
#define __INCLUDED_H_OutputDataToExcel__

#include "main.h"
#define _USE_MATH_DEFINES	// 数値演算定数が定義されたヘッダファイルの読み込み

FILE* fp; // FILEポインタの宣言
int IMAGE_NUM[] = { 1,2,3,4,6,8,10,15,20,25,30 };	// 画像枚数 K
Mat Image_dst_sub;			// 指定出力補修画像(ノイズ画像)
Mat Image_dst_average_sub;	// 指定出力補修画像(ノイズ平均画像)
Mat Image_dst_MRF_sub;		// 指定出力補修画像(ノイズ除去画像MRF)
Mat Image_dst_HMRF_sub;		// 指定出力補修画像(ノイズ除去画像HMRF)
Mat Image_dst_NLM_sub;		// 指定出力補修画像(ノイズ除去画像MRF)

void MSE_PSNR_SSIM_Output(Mat& Original, Mat& Inpaint, double& MSE_result, double& PSNR_result, double& SSIM_result);
void SSIMcalculation(double& ssim, Mat& image_1, Mat& image_2);


// 画像枚数を変えて修復精度の評価を行い、エクセルに保存する
void Inpainting_and_OutputToExcel() {
	/* Step1. CSVファイルを開く */
	double x, y;
	x = 2.5;
	y = sqrt(x);

	fopen_s(&fp, "result_data.csv", "w");
	/* CSVファイルに項目保存 */
	fprintf(fp, "K,noise_MSE,noise_PSNR,noise_SSIM,NLM_MSE,NLM_PSNR,NLM_SSIM,MRF_MSE,MRF_PSNR,MRF_SSIM,HMRF_MSE,HMRF_PSNR,HMRF_SSIM,average_MSE,average_PSNR,average_SSIM,timeMRF,timeHMRF\n");

	//--- 画像処理 ------------------------------------------------------------------------
	clock_t Start, Start2, Start3, End, End2, End3;	// 処理時間表示用
	double Time_difference = 0.0, Time_difference2 = 0.0, Time_difference3 = 0.0;

	vector<Mat> K_Image_dst;		// k枚の劣化画像
	int nowK = 1;
	double resultMSE = 0.0, resultPSNR = 0.0, resultSSIM = 0.0;
	double aveMSE1 = 0.0, avePSNR1 = 0.0, aveSSIM1 = 0.0;
	double aveMSE2 = 0.0, avePSNR2 = 0.0, aveSSIM2 = 0.0;
	double aveMSE3 = 0.0, avePSNR3 = 0.0, aveSSIM3 = 0.0;
	double aveMSE4 = 0.0, avePSNR4 = 0.0, aveSSIM4 = 0.0;
	double aveMSE5 = 0.0, avePSNR5 = 0.0, aveSSIM5 = 0.0;
	double aveTIME1 = 0.0, aveTIME2 = 0.0;
	cout << "#########################################################################" << endl;
	for (int K_number = 0; K_number < 11; K_number++) {
		/* ノイズ画像枚数指定 */
		K_Image_dst.clear();
		nowK = IMAGE_NUM[K_number];

		aveMSE1 = 0.0, avePSNR1 = 0.0, aveSSIM1 = 0.0;
		aveMSE2 = 0.0, avePSNR2 = 0.0, aveSSIM2 = 0.0;
		aveMSE3 = 0.0, avePSNR3 = 0.0, aveSSIM3 = 0.0;
		aveMSE4 = 0.0, avePSNR4 = 0.0, aveSSIM4 = 0.0;
		aveMSE5 = 0.0, avePSNR5 = 0.0, aveSSIM5 = 0.0;
		aveTIME1 = 0.0, aveTIME2 = 0.0;

		for (int Do_number = 0; Do_number < DO_NUMBER; Do_number++) {
			cout << "# Do_number=" << (int)Do_number << endl;		// 実行確認用

			/* ガウスノイズ付加 */
			cout << nowK << "枚の劣化画像生成…" << endl;		// 実行確認用
			GaussianNoiseImage(nowK, Image_src, K_Image_dst);
			K_Image_dst[0].copyTo(Image_dst);
			CreateAverageImage(nowK, Image_dst_average, K_Image_dst, MAX_INTENSE);	// 平均画像生成
			Image_dst.copyTo(Image_dst_MRF);
			Image_dst.copyTo(Image_dst_HMRF);
			Image_dst.copyTo(Image_dst_NLM);

			/* MRF(マルコフ確率場)によるノイズ除去 */
			cout << "パラメータ推定あり(MRF)のノイズ除去…" << endl;	// 実行確認用
			Start = clock();
			DenoisingMRF(Image_dst_MRF, K_Image_dst, nowK, MAX_INTENSE);
			End = clock();
			Time_difference = (double)End - (double)Start;
			double time = static_cast<double>(Time_difference) / CLOCKS_PER_SEC * 1000.0;
			aveTIME1 += (double)time;
			cout << "time : " << time << "[ms]" << endl;
			cout << endl;

			/* HMRF(階層型マルコフ確率場)によるノイズ除去 */
			cout << "パラメータ推定あり(HMRF)のノイズ除去…" << endl;	// 実行確認用
			Start2 = clock();
			DenoisingHMRF(Image_dst_HMRF, K_Image_dst, nowK, MAX_INTENSE);
			End2 = clock();
			Time_difference2 = (double)End2 - (double)Start2;
			double time2 = static_cast<double>(Time_difference2) / CLOCKS_PER_SEC * 1000.0;
			aveTIME2 += (double)time2;
			cout << "time : " << time2 << "[ms]" << endl;
			cout << endl;

			/* NonLocalMeansによるノイズ除去 */
			cout << "Non-Local Meansのノイズ除去…" << endl;	// 実行確認用
			Start3 = clock();
			//void cv::fastNlMeansDenoisingColored(src, dst, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
			fastNlMeansDenoisingColored(Image_dst_average, Image_dst_NLMdef);
			double best_h = (double)NoiseSigma / (double)sqrt(nowK);
			cout << " best_h = " << (double)best_h << endl;
			fastNlMeansDenoisingColored(Image_dst_average, Image_dst_NLM, best_h, best_h, 7, 21);
			End3 = clock();
			Time_difference3 = (double)End3 - (double)Start3;
			double time3 = static_cast<double>(Time_difference3) / CLOCKS_PER_SEC * 1000.0;
			cout << "time : " << time3 << "[ms]" << endl;
			cout << endl;

			/* 修復精度の評価 */
			cout << "ノイズ画像 と 元画像" << endl;			// 実行確認用
			MSE_PSNR_SSIM_Output(Image_src, Image_dst, resultMSE, resultPSNR, resultSSIM);
			aveMSE1 += resultMSE, avePSNR1 += resultPSNR, aveSSIM1 += resultSSIM;

			cout << "NLM修復画像 と 元画像" << endl;		// 実行確認用
			MSE_PSNR_SSIM_Output(Image_src, Image_dst_NLM, resultMSE, resultPSNR, resultSSIM);
			aveMSE2 += resultMSE, avePSNR2 += resultPSNR, aveSSIM2 += resultSSIM;

			cout << "MRF修復画像 と 元画像" << endl;		// 実行確認用
			MSE_PSNR_SSIM_Output(Image_src, Image_dst_MRF, resultMSE, resultPSNR, resultSSIM);
			aveMSE3 += resultMSE, avePSNR3 += resultPSNR, aveSSIM3 += resultSSIM;

			cout << "HMRF修復画像 と 元画像" << endl;		// 実行確認用
			MSE_PSNR_SSIM_Output(Image_src, Image_dst_HMRF, resultMSE, resultPSNR, resultSSIM);
			aveMSE4 += resultMSE, avePSNR4 += resultPSNR, aveSSIM4 += resultSSIM;

			cout << "ノイズ平均画像 と 元画像" << endl;		// 実行確認用
			MSE_PSNR_SSIM_Output(Image_src, Image_dst_average, resultMSE, resultPSNR, resultSSIM);
			aveMSE5 += resultMSE, avePSNR5 += resultPSNR, aveSSIM5 += resultSSIM;

			cout << "NLM修復画像(デフォルト) と 元画像" << endl;	// 実行確認用
			MSE_PSNR_SSIM(Image_src, Image_dst_NLMdef);
		}

		/* Step2. データの書き込み */
		fprintf(fp, "%d,", nowK);	//CSVファイルに保存

		cout << "　　　　　　　　　　平均 : MSE  ,  PSNR  ,  SSIM" << endl;	// 実行確認用
		cout << "ノイズ画像 と 元画像　　 : ";
		aveMSE1 /= (double)DO_NUMBER, avePSNR1 /= (double)DO_NUMBER, aveSSIM1 /= (double)DO_NUMBER;
		fprintf(fp, "%g,%g,%g,", aveMSE1, avePSNR1, aveSSIM1);		//CSVファイルに上書き保存
		cout << aveMSE1 << " , " << avePSNR1 << " , " << aveSSIM1 << endl;

		cout << "NLM修復画像 と 元画像　　: ";
		aveMSE2 /= (double)DO_NUMBER, avePSNR2 /= (double)DO_NUMBER, aveSSIM2 /= (double)DO_NUMBER;
		fprintf(fp, "%g,%g,%g,", aveMSE2, avePSNR2, aveSSIM2);		//CSVファイルに上書き保存
		cout << aveMSE2 << " , " << avePSNR2 << " , " << aveSSIM2 << endl;

		cout << "MRF修復画像 と 元画像　　: ";
		aveMSE3 /= (double)DO_NUMBER, avePSNR3 /= (double)DO_NUMBER, aveSSIM3 /= (double)DO_NUMBER;
		fprintf(fp, "%g,%g,%g,", aveMSE3, avePSNR3, aveSSIM3);		//CSVファイルに上書き保存
		cout << aveMSE3 << " , " << avePSNR3 << " , " << aveSSIM3 << endl;

		cout << "HMRF修復画像 と 元画像　 : ";
		aveMSE4 /= (double)DO_NUMBER, avePSNR4 /= (double)DO_NUMBER, aveSSIM4 /= (double)DO_NUMBER;
		fprintf(fp, "%g,%g,%g,", aveMSE4, avePSNR4, aveSSIM4);		//CSVファイルに上書き保存
		cout << aveMSE4 << " , " << avePSNR4 << " , " << aveSSIM4 << endl;

		cout << "ノイズ平均画像 と 元画像 : ";
		aveMSE5 /= (double)DO_NUMBER, avePSNR5 /= (double)DO_NUMBER, aveSSIM5 /= (double)DO_NUMBER;
		fprintf(fp, "%g,%g,%g,", aveMSE5, avePSNR5, aveSSIM5);		//CSVファイルに上書き保存
		cout << aveMSE5 << " , " << avePSNR5 << " , " << aveSSIM5 << endl;

		aveTIME1 /= (double)DO_NUMBER, aveTIME2 /= (double)DO_NUMBER;
		fprintf(fp, "%g,%g\n", aveTIME1, aveTIME2);	//CSVファイルに保存

		/* 最終出力画像の指定 */
		if (nowK == 1) {
			Image_dst.copyTo(Image_dst_sub);
			Image_dst_average.copyTo(Image_dst_average_sub);
			Image_dst_MRF.copyTo(Image_dst_MRF_sub);
			Image_dst_HMRF.copyTo(Image_dst_HMRF_sub);
			Image_dst_NLM.copyTo(Image_dst_NLM_sub);
		}
		cout << "#########################################################################" << endl;
	}
	//-----------------------------------------------------------------------------------------

	/* Step3. CSVファイルを閉じる */
	fclose(fp);

	/* 最終出力画像 */
	Image_dst_sub.copyTo(Image_dst);
	Image_dst_average_sub.copyTo(Image_dst_average);
	Image_dst_MRF_sub.copyTo(Image_dst_MRF);
	Image_dst_HMRF_sub.copyTo(Image_dst_HMRF);
}

// MSE&PSNR&SSIMによる画像評価
void MSE_PSNR_SSIM_Output(Mat& Original, Mat& Inpaint, double& MSE_result, double& PSNR_result, double& SSIM_result) {
	double MSE, PSNR, SSIM;
	Mat beforeIMG, afterIMG;
	Original.copyTo(beforeIMG);
	Inpaint.copyTo(afterIMG);

	double MSE_sum = 0.0;	// MSE値
	double image_cost;		// 画素値の差分
	int compare_size, color_ind;
	int occ_pix_count = 0;

	/* MSE計算(RGB) */
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

	/* PSNR計算 */
	PSNR = 20 * (double)log10(MAX_INTENSE) - 10 * (double)log10(MSE);

	/* SSIM計算 */
	SSIMcalculation(SSIM, beforeIMG, afterIMG);

	/* 評価結果表示 */
	cout << "--- 評価 ------------------------------------------" << endl;
	cout << " MSE  : " << MSE << endl;
	cout << " PSNR : " << PSNR << endl;
	cout << " SSIM : " << SSIM << endl;
	cout << "---------------------------------------------------" << endl;
	cout << endl;
	/* 評価結果を出力 */
	MSE_result = MSE;
	PSNR_result = PSNR;
	SSIM_result = SSIM;
}
// SSIM算出
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

	/* SSIM平均(RGB) */
	double SSIM;
	SSIM = (double)mssim[0] + (double)mssim[1] + (double)mssim[2];
	ssim = (double)SSIM / 3.0;
}

#endif