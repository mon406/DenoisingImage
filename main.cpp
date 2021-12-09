#include "main.h"
#include "MakeNoiseImage.h"
#include "EstimatedParameterMRF.h"
#include "EstimatedParameterHMRF.h"
#include "OutputDataToExcel.h"
#include "Image_Histgram.h"


int main() {
	/* 画像の入力 */
	Input_Image();

	clock_t start, start2, end, end2;	// 処理時間表示用
	double time_difference = 0.0, time_difference2 = 0.0;
	//--- 画像処理 ---------------------------------------------------------------------------
	if (IMAGE_NUMBER == 0) {
		Inpainting_and_OutputToExcel();
	}
	else {
		/* ガウスノイズ付加 */
		vector<Mat> K_Image_dst;		// k枚の劣化画像
		cout << IMAGE_NUMBER << "枚の劣化画像生成…" << endl;		// 実行確認用
		GaussianNoiseImage(IMAGE_NUMBER, Image_src, K_Image_dst);
		K_Image_dst[0].copyTo(Image_dst);
		CreateAverageImage(IMAGE_NUMBER, Image_dst_average, K_Image_dst, MAX_INTENSE);	// 平均画像生成
		Image_dst.copyTo(Image_dst_MRF);
		Image_dst.copyTo(Image_dst_HMRF);
		Image_dst.copyTo(Image_dst_NLM);
		Image_dst.copyTo(Image_dst_NLMdef);

		/* MRF(マルコフ確率場)によるノイズ除去 */
		cout << "パラメータ推定あり(MRF)のノイズ除去…" << endl;	// 実行確認用
		start = clock();
		DenoisingMRF(Image_dst_MRF, K_Image_dst, IMAGE_NUMBER, MAX_INTENSE);
		end = clock();
		time_difference = (double)end - (double)start;
		const double time = static_cast<double>(time_difference) / CLOCKS_PER_SEC * 1000.0;
		cout << "time : " << time << "[ms]" << endl;
		cout << endl;

		/* HMRF(階層型マルコフ確率場)によるノイズ除去 */
		cout << "パラメータ推定あり(HMRF)のノイズ除去…" << endl;	// 実行確認用
		start2 = clock();
		DenoisingHMRF(Image_dst_HMRF, K_Image_dst, IMAGE_NUMBER, MAX_INTENSE);
		end2 = clock();
		time_difference2 = (double)end2 - (double)start2;
		const double time2 = static_cast<double>(time_difference2) / CLOCKS_PER_SEC * 1000.0;
		cout << "time : " << time2 << "[ms]" << endl;
		cout << endl;

		/* NonLocalMeansによるノイズ除去 */
		cout << "Non-Local Meansのノイズ除去…" << endl;	// 実行確認用
		//void cv::fastNlMeansDenoisingColored(src, dst, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
		fastNlMeansDenoisingColored(Image_dst_average, Image_dst_NLMdef);
		double best_h = (double)NoiseSigma / (double)sqrt(IMAGE_NUMBER);
		cout << " best_h = " << (double)best_h << endl;
		//fastNlMeansDenoisingColored(Image_dst_average, Image_dst_NLM, best_h, 3, 7, 21);
		fastNlMeansDenoisingColored(Image_dst_average, Image_dst_NLM, best_h, best_h, 7, 21);
		cout << endl;

		/* 画像の評価 */
		cout << "ノイズ画像 と 元画像" << endl;			// 実行確認用
		MSE_PSNR_SSIM(Image_src, Image_dst);
		cout << "NLM修復画像(デフォルト) と 元画像" << endl;		// 実行確認用
		MSE_PSNR_SSIM(Image_src, Image_dst_NLMdef);
		cout << "NLM修復画像(最適化) と 元画像" << endl;		// 実行確認用
		MSE_PSNR_SSIM(Image_src, Image_dst_NLM);
		cout << "MRF修復画像 と 元画像" << endl;		// 実行確認用
		MSE_PSNR_SSIM(Image_src, Image_dst_MRF);
		cout << "HMRF修復画像 と 元画像" << endl;		// 実行確認用
		MSE_PSNR_SSIM(Image_src, Image_dst_HMRF);
		cout << "ノイズ平均画像 と 元画像" << endl;		// 実行確認用
		MSE_PSNR_SSIM(Image_src, Image_dst_average);
	}

	/* 出力ヒストグラム作成 */
	/*DrawHist(Image_src, Image_hist_src);
	DrawHist(Image_dst, Image_hist_dst);
	DrawHist(Image_dst_MRF, Image_hist_dst_MRF);
	DrawHist(Image_dst_HMRF, Image_hist_dst_HMRF);
	DrawHist(Image_dst_average, Image_hist_dst_average);
	DrawHist(Image_dst_NLM, Image_hist_dst_NLM);*/
	int MAX_COUNTER_CONST = 7000;
	DrawHist(Image_src, Image_hist_src, MAX_COUNTER_CONST);
	DrawHist(Image_dst, Image_hist_dst, MAX_COUNTER_CONST);
	DrawHist(Image_dst_MRF, Image_hist_dst_MRF, MAX_COUNTER_CONST);
	DrawHist(Image_dst_HMRF, Image_hist_dst_HMRF, MAX_COUNTER_CONST);
	DrawHist(Image_dst_average, Image_hist_dst_average, MAX_COUNTER_CONST);
	DrawHist(Image_dst_NLM, Image_hist_dst_NLM, MAX_COUNTER_CONST);
	//--------------------------------------------------------------------------------------------

	/* 画像の出力 */
	Output_Image();
	return 0;
}


// 画像の入力
void Input_Image() {
	//string file_src = "C:\\Users\\mon25\\Desktop\\DenoisingImage\\src.jpg";		// 入力画像のファイル名
	string file_src = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\src.jpg";		// 入力画像のファイル名
	Image_src = imread(file_src, 1);						// 入力画像（カラー）の読み込み

	/* パラメータ定義 */
	WIDTH = Image_src.cols;
	HEIGHT = Image_src.rows;
	MAX_DATA = WIDTH * HEIGHT;
	Image_dst = Mat(Size(WIDTH, HEIGHT), CV_8UC3);			// 出力画像（カラー）の初期化設定
	Image_dst_average = Mat(Size(WIDTH, HEIGHT), CV_8UC3);
	Image_dst_MRF = Mat(Size(WIDTH, HEIGHT), CV_8UC3);
	Image_dst_HMRF = Mat(Size(WIDTH, HEIGHT), CV_8UC3);
}
// 画像の出力
void Output_Image() {
	//string file_dst = "C:\\Users\\mon25\\Desktop\\DenoisingImage\\dst.jpg";		// 出力画像のファイル名
	//string file_dst2 = "C:\\Users\\mon25\\Desktop\\DenoisingImage\\dst_MRF.jpg";
	//string file_dst3 = "C:\\Users\\mon25\\Desktop\\DenoisingImage\\dst_HMRF.jpg";
	//string file_dst4 = "C:\\Users\\mon25\\Desktop\\DenoisingImage\\dst_average.jpg";
	//string file_dst5 = "C:\\Users\\mon25\\Desktop\\DenoisingImage\\dst_NLM.jpg";
	string file_dst = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst.jpg";		// 出力画像のファイル名
	string file_dst2 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_MRF.jpg";
	string file_dst3 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_HMRF.jpg";
	string file_dst4 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_average.jpg";
	string file_dst5 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_NLM.jpg";
	string file_dst6 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_hist_src.jpg";	// （出力ヒストグラム）
	string file_dst7 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_hist_dst.jpg";
	string file_dst8 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_hist_dst_MRF.jpg";
	string file_dst9 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_hist_dst_HMRF.jpg";
	string file_dst10 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_hist_dst_average.jpg";
	string file_dst11 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_hist_dst_NLM.jpg";

	/* ウィンドウ生成 */
	namedWindow(win_src, WINDOW_AUTOSIZE);
	namedWindow(win_dst, WINDOW_AUTOSIZE);
	namedWindow(win_dst2, WINDOW_AUTOSIZE);
	namedWindow(win_dst3, WINDOW_AUTOSIZE);
	namedWindow(win_dst4, WINDOW_AUTOSIZE);

	/* 画像の表示 & 保存 */
	imshow(win_src, Image_src);				// 入力画像を表示
	imshow(win_dst, Image_dst);				// 出力画像を表示(ノイズ画像)
	imwrite(file_dst, Image_dst);			// 処理結果の保存
	imshow(win_dst2, Image_dst_MRF);		// 出力画像を表示(ノイズ除去画像MRF)
	imwrite(file_dst2, Image_dst_MRF);		// 処理結果の保存
	imshow(win_dst3, Image_dst_HMRF);		// 出力画像を表示(ノイズ除去画像HMRF)
	imwrite(file_dst3, Image_dst_HMRF);		// 処理結果の保存
	imwrite(file_dst4, Image_dst_average);	// 処理結果の保存(ノイズ平均画像)
	imshow(win_dst4, Image_dst_NLM);		// 出力画像を表示(ノイズ除去画像MRF)
	imwrite(file_dst5, Image_dst_NLM);		// 処理結果の保存

	imwrite(file_dst6, Image_hist_src);		// ヒストグラムの保存
	imwrite(file_dst7, Image_hist_dst);
	imwrite(file_dst8, Image_hist_dst_MRF);
	imwrite(file_dst9, Image_hist_dst_HMRF);
	imwrite(file_dst10, Image_hist_dst_average);
	imwrite(file_dst11, Image_hist_dst_NLM);

	waitKey(0); // キー入力待ち
}


// MSE&PSNR&SSIMによる画像評価
void MSE_PSNR_SSIM(Mat& Original, Mat& Inpaint) {
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
	SSIMcalc(SSIM, beforeIMG, afterIMG);

	/* 評価結果表示 */
	cout << "--- 評価 ------------------------------------------" << endl;
	cout << " MSE  : " << MSE << endl;
	cout << " PSNR : " << PSNR << endl;
	cout << " SSIM : " << SSIM << endl;
	cout << "---------------------------------------------------" << endl;
	cout << endl;
}
// SSIM算出
void SSIMcalc(double& ssim, Mat& image_1, Mat& image_2) {
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