#ifndef __INCLUDED_H_Image_Histgram__
#define __INCLUDED_H_Image_Histgram__

#include "main.h"


/* 入出力画像 */
string win_dst_hist = "dst_hist";	// 出力画像ウィンドウ
Mat Image_hist;		// ヒストグラム画像

/* 関数宣言 */
void DrawHist(Mat& targetImg);					// ヒストグラム計算&描画
void DrawHist(Mat& targetImg, int MaxCounter);	// ヒストグラム計算&描画(高さ固定)
void DrawHist(Mat& targetImg, Mat& dstHistImg);					// ヒストグラム計算&出力
void DrawHist(Mat& targetImg, Mat& dstHistImg, int MaxCounter);	// ヒストグラム計算&出力(高さ固定)
void DrawHist_line(Mat& targetImg, Mat& dstHistImg);					// ヒストグラム計算&線画出力
void DrawHist_line(Mat& targetImg, Mat& dstHistImg, int MaxCounter);	// ヒストグラム計算&線画出力(高さ固定)


/* 関数 */
// ヒストグラム計算&描画
void DrawHist(Mat& targetImg) {
	if (targetImg.channels() == 3) {
		Mat channels[3];
		int cha_index, ha_index;
		uchar Gray;
		for (int channel = 0; channel < 3; channel++) {
			channels[channel] = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
			for (int i = 0; i < targetImg.rows; i++) {
				for (int j = 0; j < targetImg.cols; j++) {
					cha_index = i * targetImg.cols * 3 + j * 3 + channel;
					ha_index = i * targetImg.cols + j;
					Gray = (uchar)targetImg.data[cha_index];
					channels[channel].data[ha_index] = Gray;
				}
			}
		}

		/* 変数宣言 */
		Mat R, G, B;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* 画素数を数える */
		calcHist(&channels[0], 1, 0, Mat(), B, 1, &hist_size, &hist_range);
		calcHist(&channels[1], 1, 0, Mat(), G, 1, &hist_size, &hist_range);
		calcHist(&channels[2], 1, 0, Mat(), R, 1, &hist_size, &hist_range);

		/* 確認（ヒストグラム高さ固定のため）*/
		int MAX_COUNT = 0;
		double Min_count[3], Max_count[3];
		for (int ch = 0; ch < 3; ch++) {
			if (ch == 0) { minMaxLoc(B, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 1) { minMaxLoc(G, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 2) { minMaxLoc(R, &Min_count[ch], &Max_count[ch]); }
			if (Max_count[ch] > MAX_COUNT) {
				MAX_COUNT = (int)Max_count[ch];
			}
		}
		//MAX_COUNT = 80000;

		/* ヒストグラム生成用の画像を作成 */
		Image_hist = Mat(Size(276, 320), CV_8UC3, Scalar(255, 255, 255));

		/* 背景を描画（見やすくするためにヒストグラム部分の背景をグレーにする） */
		for (int i = 0; i < 3; i++) {
			rectangle(Image_hist, Point(10, 10 + 100 * i), Point(265, 100 + 100 * i), Scalar(230, 230, 230), -1);
		}

		for (int i = 0; i < 256; i++) {
			line(Image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)((float)(R.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 0, 255), 1, 8, 0);
			line(Image_hist, Point(10 + i, 200), Point(10 + i, 200 - (int)((float)(G.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 255, 0), 1, 8, 0);
			line(Image_hist, Point(10 + i, 300), Point(10 + i, 300 - (int)((float)(B.at<float>(i) / MAX_COUNT) * 80)), Scalar(255, 0, 0), 1, 8, 0);

			if (i % 10 == 0) {		// 横軸10ずつラインを引く
				line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
					Scalar(170, 170, 170), 1, 8, 0);
				line(Image_hist, Point(10 + i, 200), Point(10 + i, 110),
					Scalar(170, 170, 170), 1, 8, 0);
				line(Image_hist, Point(10 + i, 300), Point(10 + i, 210),
					Scalar(170, 170, 170), 1, 8, 0);

				if (i % 50 == 0) {	// 横軸50ずつ濃いラインを引く
					line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
						Scalar(50, 50, 50), 1, 8, 0);
					line(Image_hist, Point(10 + i, 200), Point(10 + i, 110),
						Scalar(50, 50, 50), 1, 8, 0);
					line(Image_hist, Point(10 + i, 300), Point(10 + i, 210),
						Scalar(50, 50, 50), 1, 8, 0);
				}
			}
		}

		/* ヒストグラム情報表示 */
		cout << "--- ヒストグラム情報 -------------------------" << endl;
		cout << " RGB画像" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "----------------------------------------------" << endl;

		/* ヒストグラム画像の表示 & 保存 */
		string file_dst_hist = "dst_hist.jpg";		// 出力画像のファイル名
		namedWindow(win_dst_hist, WINDOW_AUTOSIZE);	// ウィンドウ生成
		imshow(win_dst_hist, Image_hist);				// 出力画像を表示
		imwrite(file_dst_hist, Image_hist);			// 出力画像の保存

		waitKey(0);			// キー入力待ち
		destroyWindow(win_dst_hist);
	}
	else if (targetImg.channels() == 1) {
		Mat channels;
		int ha_index;
		uchar Gray;
		channels = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
		for (int i = 0; i < targetImg.rows; i++) {
			for (int j = 0; j < targetImg.cols; j++) {
				ha_index = i * targetImg.cols + j;
				Gray = (uchar)targetImg.data[ha_index];
				channels.data[ha_index] = Gray;
			}
		}

		/* 変数宣言 */
		Mat GRAY;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* 画素数を数える */
		calcHist(&channels, 1, 0, Mat(), GRAY, 1, &hist_size, &hist_range);

		/* 確認（ヒストグラム高さ固定のため）*/
		int MAX_COUNT = 0;
		double Min_count, Max_count;
		minMaxLoc(GRAY, &Min_count, &Max_count);
		if (Max_count > MAX_COUNT) {
			MAX_COUNT = (int)Max_count;
		}
		//MAX_COUNT = 80000;

		/* ヒストグラム生成用の画像を作成 */
		Image_hist = Mat(Size(276, 120), CV_8UC3, Scalar(255, 255, 255));

		/* 背景を描画（見やすくするためにヒストグラム部分の背景をグレーにする） */
		rectangle(Image_hist, Point(10, 10), Point(265, 100), Scalar(230, 230, 230), -1);

		for (int i = 0; i < 256; i++) {
			line(Image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)((float)(GRAY.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 0, 0), 1, 8, 0);

			if (i % 10 == 0) {		// 横軸10ずつラインを引く
				line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
					Scalar(170, 170, 170), 1, 8, 0);

				if (i % 50 == 0) {	// 横軸50ずつ濃いラインを引く
					line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
						Scalar(50, 50, 50), 1, 8, 0);
				}
			}
		}

		/* ヒストグラム情報表示 */
		cout << "--- ヒストグラム情報 ------------------------------" << endl;
		cout << " GrayScale画像" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "---------------------------------------------------" << endl;

		/* ヒストグラム画像の表示 & 保存 */
		string file_dst_hist = "dst_hist.jpg";		// 出力画像のファイル名
		namedWindow(win_dst_hist, WINDOW_AUTOSIZE);	// ウィンドウ生成
		imshow(win_dst_hist, Image_hist);				// 出力画像を表示
		imwrite(file_dst_hist, Image_hist);			// 出力画像の保存

		waitKey(0);			// キー入力待ち
		destroyWindow(win_dst_hist);
	}
	else {
		cout << "ERROR! drawHist_Color()  :  Can't draw Histgram because of its channel." << endl;
	}
	cout << endl;
}
void DrawHist(Mat& targetImg, int MaxCounter) {
	if (targetImg.channels() == 3) {
		Mat channels[3];
		int cha_index, ha_index;
		uchar Gray;
		for (int channel = 0; channel < 3; channel++) {
			channels[channel] = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
			for (int i = 0; i < targetImg.rows; i++) {
				for (int j = 0; j < targetImg.cols; j++) {
					cha_index = i * targetImg.cols * 3 + j * 3 + channel;
					ha_index = i * targetImg.cols + j;
					Gray = (uchar)targetImg.data[cha_index];
					channels[channel].data[ha_index] = Gray;
				}
			}
		}

		/* 変数宣言 */
		Mat R, G, B;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* 画素数を数える */
		calcHist(&channels[0], 1, 0, Mat(), B, 1, &hist_size, &hist_range);
		calcHist(&channels[1], 1, 0, Mat(), G, 1, &hist_size, &hist_range);
		calcHist(&channels[2], 1, 0, Mat(), R, 1, &hist_size, &hist_range);

		/* 確認（ヒストグラム高さ固定のため）*/
		int MAX_COUNT = 0;
		double Min_count[3], Max_count[3];
		for (int ch = 0; ch < 3; ch++) {
			if (ch == 0) { minMaxLoc(B, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 1) { minMaxLoc(G, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 2) { minMaxLoc(R, &Min_count[ch], &Max_count[ch]); }
			if (Max_count[ch] > MAX_COUNT) {
				MAX_COUNT = (int)Max_count[ch];
			}
		}
		if (MAX_COUNT > MaxCounter) { cout << "WARNING! DrawHist : MaxCounter is too small." << endl; }
		MAX_COUNT = MaxCounter;

		/* ヒストグラム生成用の画像を作成 */
		Image_hist = Mat(Size(276, 320), CV_8UC3, Scalar(255, 255, 255));

		/* 背景を描画（見やすくするためにヒストグラム部分の背景をグレーにする） */
		for (int i = 0; i < 3; i++) {
			rectangle(Image_hist, Point(10, 10 + 100 * i), Point(265, 100 + 100 * i), Scalar(230, 230, 230), -1);
		}

		for (int i = 0; i < 256; i++) {
			if(R.at<float>(i) <= MAX_COUNT * 1.15){ line(Image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)((float)(R.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 0, 255), 1, 8, 0); }
			else { line(Image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)(1.15 * 80)), Scalar(0, 0, 255), 1, 8, 0); }
			if (G.at<float>(i) <= MAX_COUNT * 1.15) { line(Image_hist, Point(10 + i, 200), Point(10 + i, 200 - (int)((float)(G.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 255, 0), 1, 8, 0); }
			else { line(Image_hist, Point(10 + i, 200), Point(10 + i, 200 - (int)(1.15 * 80)), Scalar(0, 255, 0), 1, 8, 0); }
			if (B.at<float>(i) <= MAX_COUNT * 1.15) { line(Image_hist, Point(10 + i, 300), Point(10 + i, 300 - (int)((float)(B.at<float>(i) / MAX_COUNT) * 80)), Scalar(255, 0, 0), 1, 8, 0); }
			else { line(Image_hist, Point(10 + i, 300), Point(10 + i, 300 - (int)(1.15 * 80)), Scalar(255, 0, 0), 1, 8, 0); }

			if (i % 10 == 0) {		// 横軸10ずつラインを引く
				line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
					Scalar(170, 170, 170), 1, 8, 0);
				line(Image_hist, Point(10 + i, 200), Point(10 + i, 110),
					Scalar(170, 170, 170), 1, 8, 0);
				line(Image_hist, Point(10 + i, 300), Point(10 + i, 210),
					Scalar(170, 170, 170), 1, 8, 0);

				if (i % 50 == 0) {	// 横軸50ずつ濃いラインを引く
					line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
						Scalar(50, 50, 50), 1, 8, 0);
					line(Image_hist, Point(10 + i, 200), Point(10 + i, 110),
						Scalar(50, 50, 50), 1, 8, 0);
					line(Image_hist, Point(10 + i, 300), Point(10 + i, 210),
						Scalar(50, 50, 50), 1, 8, 0);
				}
			}
		}

		/* ヒストグラム情報表示 */
		cout << "--- ヒストグラム情報 -------------------------" << endl;
		cout << " RGB画像" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "----------------------------------------------" << endl;

		/* ヒストグラム画像の表示 & 保存 */
		string file_dst_hist = "dst_hist.jpg";		// 出力画像のファイル名
		namedWindow(win_dst_hist, WINDOW_AUTOSIZE);	// ウィンドウ生成
		imshow(win_dst_hist, Image_hist);				// 出力画像を表示
		imwrite(file_dst_hist, Image_hist);			// 出力画像の保存

		waitKey(0);			// キー入力待ち
		destroyWindow(win_dst_hist);
	}
	else if (targetImg.channels() == 1) {
		Mat channels;
		int ha_index;
		uchar Gray;
		channels = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
		for (int i = 0; i < targetImg.rows; i++) {
			for (int j = 0; j < targetImg.cols; j++) {
				ha_index = i * targetImg.cols + j;
				Gray = (uchar)targetImg.data[ha_index];
				channels.data[ha_index] = Gray;
			}
		}

		/* 変数宣言 */
		Mat GRAY;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* 画素数を数える */
		calcHist(&channels, 1, 0, Mat(), GRAY, 1, &hist_size, &hist_range);

		/* 確認（ヒストグラム高さ固定のため）*/
		int MAX_COUNT = 0;
		double Min_count, Max_count;
		minMaxLoc(GRAY, &Min_count, &Max_count);
		if (Max_count > MAX_COUNT) {
			MAX_COUNT = (int)Max_count;
		}
		if (MAX_COUNT > MaxCounter) { cout << "WARNING! DrawHist : MaxCounter is too small." << endl; }
		MAX_COUNT = MaxCounter;

		/* ヒストグラム生成用の画像を作成 */
		Image_hist = Mat(Size(276, 120), CV_8UC3, Scalar(255, 255, 255));

		/* 背景を描画（見やすくするためにヒストグラム部分の背景をグレーにする） */
		rectangle(Image_hist, Point(10, 10), Point(265, 100), Scalar(230, 230, 230), -1);

		for (int i = 0; i < 256; i++) {
			if (GRAY.at<float>(i) <= MAX_COUNT * 1.15) { line(Image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)((float)(GRAY.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 0, 0), 1, 8, 0); }
			else { line(Image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)(1.15 * 80)), Scalar(0, 0, 0), 1, 8, 0); }

			if (i % 10 == 0) {		// 横軸10ずつラインを引く
				line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
					Scalar(170, 170, 170), 1, 8, 0);

				if (i % 50 == 0) {	// 横軸50ずつ濃いラインを引く
					line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
						Scalar(50, 50, 50), 1, 8, 0);
				}
			}
		}

		/* ヒストグラム情報表示 */
		cout << "--- ヒストグラム情報 ------------------------------" << endl;
		cout << " GrayScale画像" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "---------------------------------------------------" << endl;

		/* ヒストグラム画像の表示 & 保存 */
		string file_dst_hist = "dst_hist.jpg";		// 出力画像のファイル名
		namedWindow(win_dst_hist, WINDOW_AUTOSIZE);	// ウィンドウ生成
		imshow(win_dst_hist, Image_hist);				// 出力画像を表示
		imwrite(file_dst_hist, Image_hist);			// 出力画像の保存

		waitKey(0);			// キー入力待ち
		destroyWindow(win_dst_hist);
	}
	else {
		cout << "ERROR! drawHist_Color()  :  Can't draw Histgram because of its channel." << endl;
	}
	cout << endl;
}

// ヒストグラム計算&出力
void DrawHist(Mat& targetImg, Mat& dstHistImg) {
	dstHistImg = Mat::zeros(targetImg.size(), CV_8U);

	if (targetImg.channels() == 3) {
		Mat channels[3];
		int cha_index, ha_index;
		uchar Gray;
		for (int channel = 0; channel < 3; channel++) {
			channels[channel] = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
			for (int i = 0; i < targetImg.rows; i++) {
				for (int j = 0; j < targetImg.cols; j++) {
					cha_index = i * targetImg.cols * 3 + j * 3 + channel;
					ha_index = i * targetImg.cols + j;
					Gray = (uchar)targetImg.data[cha_index];
					channels[channel].data[ha_index] = Gray;
				}
			}
		}

		/* 変数宣言 */
		Mat R, G, B;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* 画素数を数える */
		calcHist(&channels[0], 1, 0, Mat(), B, 1, &hist_size, &hist_range);
		calcHist(&channels[1], 1, 0, Mat(), G, 1, &hist_size, &hist_range);
		calcHist(&channels[2], 1, 0, Mat(), R, 1, &hist_size, &hist_range);

		/* 確認（ヒストグラム高さ固定のため）*/
		int MAX_COUNT = 0;
		double Min_count[3], Max_count[3];
		for (int ch = 0; ch < 3; ch++) {
			if (ch == 0) { minMaxLoc(B, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 1) { minMaxLoc(G, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 2) { minMaxLoc(R, &Min_count[ch], &Max_count[ch]); }
			if (Max_count[ch] > MAX_COUNT) {
				MAX_COUNT = (int)Max_count[ch];
			}
		}
		//MAX_COUNT = 80000;

		/* ヒストグラム生成用の画像を作成 */
		Image_hist = Mat(Size(276, 320), CV_8UC3, Scalar(255, 255, 255));

		/* 背景を描画（見やすくするためにヒストグラム部分の背景をグレーにする） */
		for (int i = 0; i < 3; i++) {
			rectangle(Image_hist, Point(10, 10 + 100 * i), Point(265, 100 + 100 * i), Scalar(230, 230, 230), -1);
		}

		for (int i = 0; i < 256; i++) {
			line(Image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)((float)(R.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 0, 255), 1, 8, 0);
			line(Image_hist, Point(10 + i, 200), Point(10 + i, 200 - (int)((float)(G.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 255, 0), 1, 8, 0);
			line(Image_hist, Point(10 + i, 300), Point(10 + i, 300 - (int)((float)(B.at<float>(i) / MAX_COUNT) * 80)), Scalar(255, 0, 0), 1, 8, 0);

			if (i % 10 == 0) {		// 横軸10ずつラインを引く
				line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
					Scalar(170, 170, 170), 1, 8, 0);
				line(Image_hist, Point(10 + i, 200), Point(10 + i, 110),
					Scalar(170, 170, 170), 1, 8, 0);
				line(Image_hist, Point(10 + i, 300), Point(10 + i, 210),
					Scalar(170, 170, 170), 1, 8, 0);

				if (i % 50 == 0) {	// 横軸50ずつ濃いラインを引く
					line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
						Scalar(50, 50, 50), 1, 8, 0);
					line(Image_hist, Point(10 + i, 200), Point(10 + i, 110),
						Scalar(50, 50, 50), 1, 8, 0);
					line(Image_hist, Point(10 + i, 300), Point(10 + i, 210),
						Scalar(50, 50, 50), 1, 8, 0);
				}
			}
		}

		/* ヒストグラム情報表示 */
		cout << "--- ヒストグラム情報 -------------------------" << endl;
		cout << " RGB画像" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "----------------------------------------------" << endl;

		/* ヒストグラム画像の出力 */
		Image_hist.copyTo(dstHistImg);
	}
	else if (targetImg.channels() == 1) {
		Mat channels;
		int ha_index;
		uchar Gray;
		channels = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
		for (int i = 0; i < targetImg.rows; i++) {
			for (int j = 0; j < targetImg.cols; j++) {
				ha_index = i * targetImg.cols + j;
				Gray = (uchar)targetImg.data[ha_index];
				channels.data[ha_index] = Gray;
			}
		}

		/* 変数宣言 */
		Mat GRAY;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* 画素数を数える */
		calcHist(&channels, 1, 0, Mat(), GRAY, 1, &hist_size, &hist_range);

		/* 確認（ヒストグラム高さ固定のため）*/
		int MAX_COUNT = 0;
		double Min_count, Max_count;
		minMaxLoc(GRAY, &Min_count, &Max_count);
		if (Max_count > MAX_COUNT) {
			MAX_COUNT = (int)Max_count;
		}
		//MAX_COUNT = 80000;

		/* ヒストグラム生成用の画像を作成 */
		Image_hist = Mat(Size(276, 120), CV_8UC3, Scalar(255, 255, 255));

		/* 背景を描画（見やすくするためにヒストグラム部分の背景をグレーにする） */
		rectangle(Image_hist, Point(10, 10), Point(265, 100), Scalar(230, 230, 230), -1);

		for (int i = 0; i < 256; i++) {
			line(Image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)((float)(GRAY.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 0, 0), 1, 8, 0);

			if (i % 10 == 0) {		// 横軸10ずつラインを引く
				line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
					Scalar(170, 170, 170), 1, 8, 0);

				if (i % 50 == 0) {	// 横軸50ずつ濃いラインを引く
					line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
						Scalar(50, 50, 50), 1, 8, 0);
				}
			}
		}

		/* ヒストグラム情報表示 */
		cout << "--- ヒストグラム情報 ------------------------------" << endl;
		cout << " GrayScale画像" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "---------------------------------------------------" << endl;

		/* ヒストグラム画像の出力 */
		Image_hist.copyTo(dstHistImg);
	}
	else {
		cout << "ERROR! drawHist_Color()  :  Can't draw Histgram because of its channel." << endl;
	}
	cout << endl;
}
void DrawHist(Mat& targetImg, Mat& dstHistImg, int MaxCounter) {
	dstHistImg = Mat::zeros(targetImg.size(), CV_8U);

	if (targetImg.channels() == 3) {
		Mat channels[3];
		int cha_index, ha_index;
		uchar Gray;
		for (int channel = 0; channel < 3; channel++) {
			channels[channel] = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
			for (int i = 0; i < targetImg.rows; i++) {
				for (int j = 0; j < targetImg.cols; j++) {
					cha_index = i * targetImg.cols * 3 + j * 3 + channel;
					ha_index = i * targetImg.cols + j;
					Gray = (uchar)targetImg.data[cha_index];
					channels[channel].data[ha_index] = Gray;
				}
			}
		}

		/* 変数宣言 */
		Mat R, G, B;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* 画素数を数える */
		calcHist(&channels[0], 1, 0, Mat(), B, 1, &hist_size, &hist_range);
		calcHist(&channels[1], 1, 0, Mat(), G, 1, &hist_size, &hist_range);
		calcHist(&channels[2], 1, 0, Mat(), R, 1, &hist_size, &hist_range);

		/* 確認（ヒストグラム高さ固定のため）*/
		int MAX_COUNT = 0;
		double Min_count[3], Max_count[3];
		for (int ch = 0; ch < 3; ch++) {
			if (ch == 0) { minMaxLoc(B, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 1) { minMaxLoc(G, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 2) { minMaxLoc(R, &Min_count[ch], &Max_count[ch]); }
			if (Max_count[ch] > MAX_COUNT) {
				MAX_COUNT = (int)Max_count[ch];
			}
		}
		if (MAX_COUNT > MaxCounter) { cout << "WARNING! DrawHist() : MaxCounter is too small." << endl; }
		MAX_COUNT = MaxCounter;

		/* ヒストグラム生成用の画像を作成 */
		Image_hist = Mat(Size(276, 320), CV_8UC3, Scalar(255, 255, 255));

		/* 背景を描画（見やすくするためにヒストグラム部分の背景をグレーにする） */
		for (int i = 0; i < 3; i++) {
			rectangle(Image_hist, Point(10, 10 + 100 * i), Point(265, 100 + 100 * i), Scalar(230, 230, 230), -1);
		}

		for (int i = 0; i < 256; i++) {
			if (R.at<float>(i) <= MAX_COUNT * 1.15) { line(Image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)((float)(R.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 0, 255), 1, 8, 0); }
			else { line(Image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)(1.15 * 80)), Scalar(0, 0, 255), 1, 8, 0); }
			if (G.at<float>(i) <= MAX_COUNT * 1.15) { line(Image_hist, Point(10 + i, 200), Point(10 + i, 200 - (int)((float)(G.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 255, 0), 1, 8, 0); }
			else { line(Image_hist, Point(10 + i, 200), Point(10 + i, 200 - (int)(1.15 * 80)), Scalar(0, 255, 0), 1, 8, 0); }
			if (B.at<float>(i) <= MAX_COUNT * 1.15) { line(Image_hist, Point(10 + i, 300), Point(10 + i, 300 - (int)((float)(B.at<float>(i) / MAX_COUNT) * 80)), Scalar(255, 0, 0), 1, 8, 0); }
			else { line(Image_hist, Point(10 + i, 300), Point(10 + i, 300 - (int)(1.15 * 80)), Scalar(255, 0, 0), 1, 8, 0); }

			if (i % 10 == 0) {		// 横軸10ずつラインを引く
				line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
					Scalar(170, 170, 170), 1, 8, 0);
				line(Image_hist, Point(10 + i, 200), Point(10 + i, 110),
					Scalar(170, 170, 170), 1, 8, 0);
				line(Image_hist, Point(10 + i, 300), Point(10 + i, 210),
					Scalar(170, 170, 170), 1, 8, 0);

				if (i % 50 == 0) {	// 横軸50ずつ濃いラインを引く
					line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
						Scalar(50, 50, 50), 1, 8, 0);
					line(Image_hist, Point(10 + i, 200), Point(10 + i, 110),
						Scalar(50, 50, 50), 1, 8, 0);
					line(Image_hist, Point(10 + i, 300), Point(10 + i, 210),
						Scalar(50, 50, 50), 1, 8, 0);
				}
			}
		}

		/* ヒストグラム情報表示 */
		cout << "--- ヒストグラム情報 -------------------------" << endl;
		cout << " RGB画像" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "----------------------------------------------" << endl;

		/* ヒストグラム画像の出力 */
		Image_hist.copyTo(dstHistImg);
	}
	else if (targetImg.channels() == 1) {
		Mat channels;
		int ha_index;
		uchar Gray;
		channels = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
		for (int i = 0; i < targetImg.rows; i++) {
			for (int j = 0; j < targetImg.cols; j++) {
				ha_index = i * targetImg.cols + j;
				Gray = (uchar)targetImg.data[ha_index];
				channels.data[ha_index] = Gray;
			}
		}

		/* 変数宣言 */
		Mat GRAY;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* 画素数を数える */
		calcHist(&channels, 1, 0, Mat(), GRAY, 1, &hist_size, &hist_range);

		/* 確認（ヒストグラム高さ固定のため）*/
		int MAX_COUNT = 0;
		double Min_count, Max_count;
		minMaxLoc(GRAY, &Min_count, &Max_count);
		if (Max_count > MAX_COUNT) {
			MAX_COUNT = (int)Max_count;
		}
		if (MAX_COUNT > MaxCounter) { cout << "WARNING! DrawHist() : MaxCounter is too small." << endl; }
		MAX_COUNT = MaxCounter;

		/* ヒストグラム生成用の画像を作成 */
		Image_hist = Mat(Size(276, 120), CV_8UC3, Scalar(255, 255, 255));

		/* 背景を描画（見やすくするためにヒストグラム部分の背景をグレーにする） */
		rectangle(Image_hist, Point(10, 10), Point(265, 100), Scalar(230, 230, 230), -1);

		for (int i = 0; i < 256; i++) {
			if (GRAY.at<float>(i) <= MAX_COUNT * 1.15) { line(Image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)((float)(GRAY.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 0, 0), 1, 8, 0); }
			else { line(Image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)(1.15 * 80)), Scalar(0, 0, 0), 1, 8, 0); }

			if (i % 10 == 0) {		// 横軸10ずつラインを引く
				line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
					Scalar(170, 170, 170), 1, 8, 0);

				if (i % 50 == 0) {	// 横軸50ずつ濃いラインを引く
					line(Image_hist, Point(10 + i, 100), Point(10 + i, 10),
						Scalar(50, 50, 50), 1, 8, 0);
				}
			}
		}

		/* ヒストグラム情報表示 */
		cout << "--- ヒストグラム情報 ------------------------------" << endl;
		cout << " GrayScale画像" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "---------------------------------------------------" << endl;

		/* ヒストグラム画像の出力 */
		Image_hist.copyTo(dstHistImg);
	}
	else {
		cout << "ERROR! drawHist_Color()  :  Can't draw Histgram because of its channel." << endl;
	}
	cout << endl;
}

// ヒストグラム計算&線画出力
void DrawHist_line(Mat& targetImg, Mat& dstHistImg) {
	dstHistImg = Mat::zeros(targetImg.size(), CV_8U);

	if (targetImg.channels() == 3) {
		Mat channels[3];
		int cha_index, ha_index;
		uchar Gray;
		for (int channel = 0; channel < 3; channel++) {
			channels[channel] = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
			for (int i = 0; i < targetImg.rows; i++) {
				for (int j = 0; j < targetImg.cols; j++) {
					cha_index = i * targetImg.cols * 3 + j * 3 + channel;
					ha_index = i * targetImg.cols + j;
					Gray = (uchar)targetImg.data[cha_index];
					channels[channel].data[ha_index] = Gray;
				}
			}
		}

		/* 変数宣言 */
		Mat R, G, B;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* 画素数を数える */
		calcHist(&channels[0], 1, 0, Mat(), B, 1, &hist_size, &hist_range);
		calcHist(&channels[1], 1, 0, Mat(), G, 1, &hist_size, &hist_range);
		calcHist(&channels[2], 1, 0, Mat(), R, 1, &hist_size, &hist_range);

		/* 確認（ヒストグラム高さ固定のため）*/
		int MAX_COUNT = 0;
		double Min_count[3], Max_count[3];
		for (int ch = 0; ch < 3; ch++) {
			if (ch == 0) { minMaxLoc(B, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 1) { minMaxLoc(G, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 2) { minMaxLoc(R, &Min_count[ch], &Max_count[ch]); }
			if (Max_count[ch] > MAX_COUNT) {
				MAX_COUNT = (int)Max_count[ch];
			}
		}

		/* ヒストグラム生成用の画像を作成 */
		Image_hist = Mat(Size(276, 320), CV_8UC3, Scalar(255, 255, 255));

		/* 縦横の軸線を引く */
		line(Image_hist, Point(10, 100), Point(265, 100), Scalar(170, 170, 170), 1, 8, 0);
		line(Image_hist, Point(10, 200), Point(265, 200), Scalar(170, 170, 170), 1, 8, 0);
		line(Image_hist, Point(10, 300), Point(265, 300), Scalar(170, 170, 170), 1, 8, 0);
		line(Image_hist, Point(10, 100), Point(10, 10), Scalar(50, 50, 50), 1, 8, 0);
		line(Image_hist, Point(10, 200), Point(10, 110), Scalar(50, 50, 50), 1, 8, 0);
		line(Image_hist, Point(10, 300), Point(10, 210), Scalar(50, 50, 50), 1, 8, 0);

		for (int i = 1; i < 256; i++) {
			line(Image_hist, Point(10 + (i - 1), 100 - (int)((float)(R.at<float>(i - 1) / MAX_COUNT) * 80)), Point(10 + i, 100 - (int)((float)(R.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 0, 255), 2, 8, 0);
			line(Image_hist, Point(10 + (i - 1), 200 - (int)((float)(G.at<float>(i - 1) / MAX_COUNT) * 80)), Point(10 + i, 200 - (int)((float)(G.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 255, 0), 2, 8, 0);
			line(Image_hist, Point(10 + (i - 1), 300 - (int)((float)(B.at<float>(i - 1) / MAX_COUNT) * 80)), Point(10 + i, 300 - (int)((float)(B.at<float>(i) / MAX_COUNT) * 80)), Scalar(255, 0, 0), 2, 8, 0);
		}

		/* ヒストグラム情報表示 */
		cout << "--- ヒストグラム情報 -------------------------" << endl;
		cout << " RGB画像" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "----------------------------------------------" << endl;

		/* ヒストグラム画像の出力 */
		Image_hist.copyTo(dstHistImg);
	}
	else if (targetImg.channels() == 1) {
		Mat channels;
		int ha_index;
		uchar Gray;
		channels = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
		for (int i = 0; i < targetImg.rows; i++) {
			for (int j = 0; j < targetImg.cols; j++) {
				ha_index = i * targetImg.cols + j;
				Gray = (uchar)targetImg.data[ha_index];
				channels.data[ha_index] = Gray;
			}
		}

		/* 変数宣言 */
		Mat GRAY;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* 画素数を数える */
		calcHist(&channels, 1, 0, Mat(), GRAY, 1, &hist_size, &hist_range);

		/* 確認（ヒストグラム高さ固定のため）*/
		int MAX_COUNT = 0;
		double Min_count, Max_count;
		minMaxLoc(GRAY, &Min_count, &Max_count);
		if (Max_count > MAX_COUNT) {
			MAX_COUNT = (int)Max_count;
		}

		/* ヒストグラム生成用の画像を作成 */
		Image_hist = Mat(Size(276, 120), CV_8UC3, Scalar(255, 255, 255));

		/* 縦横の軸線を引く */
		line(Image_hist, Point(10, 100), Point(265, 100), Scalar(170, 170, 170), 1, 8, 0);
		line(Image_hist, Point(10, 100), Point(10, 10), Scalar(50, 50, 50), 1, 8, 0);

		for (int i = 1; i < 256; i++) {
			line(Image_hist, Point(10 + (i - 1), 100 - (int)((float)(GRAY.at<float>(i - 1) / MAX_COUNT) * 80)), Point(10 + i, 100 - (int)((float)(GRAY.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 0, 0), 2, 8, 0);
		}

		/* ヒストグラム情報表示 */
		cout << "--- ヒストグラム情報 ------------------------------" << endl;
		cout << " GrayScale画像" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "---------------------------------------------------" << endl;

		/* ヒストグラム画像の出力 */
		Image_hist.copyTo(dstHistImg);
	}
	else {
		cout << "ERROR! drawHist_Color()  :  Can't draw Histgram because of its channel." << endl;
	}
	cout << endl;

	Mat Mat_RGBA(dstHistImg.size(), CV_8UC4);
	Mat img;
	img = dstHistImg;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			cv::Vec3b p = img.at<cv::Vec3b>(cv::Point(x, y));
			if (p[0] == 0 && p[1] == 0 && p[2] == 0) {	// 白の時透過
				Mat_RGBA.at<cv::Vec4b>(cv::Point(x, y))
					= cv::Vec4b(p[0], p[1], p[2], 0); //透過
			}
			else {
				Mat_RGBA.at<cv::Vec4b>(cv::Point(x, y))
					= cv::Vec4b(p[0], p[1], p[2], 255); //不透過
			}
		}
	}
	Mat_RGBA.copyTo(dstHistImg);
}
void DrawHist_line(Mat& targetImg, Mat& dstHistImg, int MaxCounter) {
	dstHistImg = Mat::zeros(targetImg.size(), CV_8U);

	if (targetImg.channels() == 3) {
		Mat channels[3];
		int cha_index, ha_index;
		uchar Gray;
		for (int channel = 0; channel < 3; channel++) {
			channels[channel] = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
			for (int i = 0; i < targetImg.rows; i++) {
				for (int j = 0; j < targetImg.cols; j++) {
					cha_index = i * targetImg.cols * 3 + j * 3 + channel;
					ha_index = i * targetImg.cols + j;
					Gray = (uchar)targetImg.data[cha_index];
					channels[channel].data[ha_index] = Gray;
				}
			}
		}

		/* 変数宣言 */
		Mat R, G, B;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* 画素数を数える */
		calcHist(&channels[0], 1, 0, Mat(), B, 1, &hist_size, &hist_range);
		calcHist(&channels[1], 1, 0, Mat(), G, 1, &hist_size, &hist_range);
		calcHist(&channels[2], 1, 0, Mat(), R, 1, &hist_size, &hist_range);

		/* 確認（ヒストグラム高さ固定のため）*/
		int MAX_COUNT = 0;
		double Min_count[3], Max_count[3];
		for (int ch = 0; ch < 3; ch++) {
			if (ch == 0) { minMaxLoc(B, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 1) { minMaxLoc(G, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 2) { minMaxLoc(R, &Min_count[ch], &Max_count[ch]); }
			if (Max_count[ch] > MAX_COUNT) {
				MAX_COUNT = (int)Max_count[ch];
			}
		}
		if (MAX_COUNT > MaxCounter) { cout << "WARNING! DrawHist() : MaxCounter is too small." << endl; }
		MAX_COUNT = MaxCounter;

		/* ヒストグラム生成用の画像を作成 */
		Image_hist = Mat(Size(276, 320), CV_8UC3, Scalar(255, 255, 255));

		/* 縦横の軸線を引く */
		line(Image_hist, Point(10, 100), Point(265, 100), Scalar(170, 170, 170), 1, 8, 0);
		line(Image_hist, Point(10, 200), Point(265, 200), Scalar(170, 170, 170), 1, 8, 0);
		line(Image_hist, Point(10, 300), Point(265, 300), Scalar(170, 170, 170), 1, 8, 0);
		line(Image_hist, Point(10, 100), Point(10, 10), Scalar(50, 50, 50), 1, 8, 0);
		line(Image_hist, Point(10, 200), Point(10, 110), Scalar(50, 50, 50), 1, 8, 0);
		line(Image_hist, Point(10, 300), Point(10, 210), Scalar(50, 50, 50), 1, 8, 0);

		Point before_R, before_G, before_B;
		Point after_R, after_G, after_B;
		if (R.at<float>(0) <= MAX_COUNT * 1.15) { before_R = Point(10, 100 - (int)((float)(R.at<float>(0) / MAX_COUNT) * 80)); }
		else { before_R = Point(10, 100 - (int)(1.15 * 80)); }
		if (G.at<float>(0) <= MAX_COUNT * 1.15) { before_G = Point(10, 200 - (int)((float)(G.at<float>(0) / MAX_COUNT) * 80)); }
		else { before_G = Point(10, 200 - (int)(1.15 * 80)); }
		if (B.at<float>(0) <= MAX_COUNT * 1.15) { before_B = Point(10, 300 - (int)((float)(B.at<float>(0) / MAX_COUNT) * 80)); }
		else { before_B = Point(10, 300 - (int)(1.15 * 80)); }
		for (int i = 1; i < 256; i++) {
			if (R.at<float>(i) <= MAX_COUNT * 1.15) { after_R = Point(10 + i, 100 - (int)((float)(R.at<float>(i) / MAX_COUNT) * 80)); }
			else { after_R = Point(10 + i, 100 - (int)(1.15 * 80)); }
			if (G.at<float>(i) <= MAX_COUNT * 1.15) { after_G = Point(10 + i, 200 - (int)((float)(G.at<float>(i) / MAX_COUNT) * 80)); }
			else { after_G = Point(10 + i, 200 - (int)(1.15 * 80)); }
			if (B.at<float>(i) <= MAX_COUNT * 1.15) { after_B = Point(10 + i, 300 - (int)((float)(B.at<float>(i) / MAX_COUNT) * 80)); }
			else { after_B = Point(10 + i, 300 - (int)(1.15 * 80)); }
			line(Image_hist, before_R, after_R, Scalar(0, 0, 255), 2, 8, 0);
			line(Image_hist, before_G, after_G, Scalar(0, 255, 0), 2, 8, 0);
			line(Image_hist, before_B, after_B, Scalar(255, 0, 0), 2, 8, 0);
			before_R = after_R;
			before_G = after_G;
			before_B = after_B;
		}

		/* ヒストグラム情報表示 */
		cout << "--- ヒストグラム情報 -------------------------" << endl;
		cout << " RGB画像" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "----------------------------------------------" << endl;

		/* ヒストグラム画像の出力 */
		Image_hist.copyTo(dstHistImg);
	}
	else if (targetImg.channels() == 1) {
		Mat channels;
		int ha_index;
		uchar Gray;
		channels = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
		for (int i = 0; i < targetImg.rows; i++) {
			for (int j = 0; j < targetImg.cols; j++) {
				ha_index = i * targetImg.cols + j;
				Gray = (uchar)targetImg.data[ha_index];
				channels.data[ha_index] = Gray;
			}
		}

		/* 変数宣言 */
		Mat GRAY;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* 画素数を数える */
		calcHist(&channels, 1, 0, Mat(), GRAY, 1, &hist_size, &hist_range);

		/* 確認（ヒストグラム高さ固定のため）*/
		int MAX_COUNT = 0;
		double Min_count, Max_count;
		minMaxLoc(GRAY, &Min_count, &Max_count);
		if (Max_count > MAX_COUNT) {
			MAX_COUNT = (int)Max_count;
		}
		if (MAX_COUNT > MaxCounter) { cout << "WARNING! DrawHist() : MaxCounter is too small." << endl; }
		MAX_COUNT = MaxCounter;

		/* ヒストグラム生成用の画像を作成 */
		Image_hist = Mat(Size(276, 120), CV_8UC3, Scalar(255, 255, 255));

		/* 縦横の軸線を引く */
		line(Image_hist, Point(10, 100), Point(265, 100), Scalar(170, 170, 170), 1, 8, 0);
		line(Image_hist, Point(10, 100), Point(10, 10), Scalar(50, 50, 50), 1, 8, 0);

		Point before, after;
		if (GRAY.at<float>(0) <= MAX_COUNT * 1.15) { before = Point(10, 100 - (int)((float)(GRAY.at<float>(0) / MAX_COUNT) * 80)); }
		else { before = Point(10, 100 - (int)(1.15 * 80)); }
		for (int i = 1; i < 256; i++) {
			if (GRAY.at<float>(i) <= MAX_COUNT * 1.15) { after = Point(10 + i, 100 - (int)((float)(GRAY.at<float>(i) / MAX_COUNT) * 80)); }
			else { after = Point(10 + i, 100 - (int)(1.15 * 80)); }
			line(Image_hist, before, after, Scalar(0, 0, 0), 2, 8, 0);
			before = after;
		}

		/* ヒストグラム情報表示 */
		cout << "--- ヒストグラム情報 ------------------------------" << endl;
		cout << " GrayScale画像" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "---------------------------------------------------" << endl;

		/* ヒストグラム画像の出力 */
		Image_hist.copyTo(dstHistImg);
	}
	else {
		cout << "ERROR! drawHist_Color()  :  Can't draw Histgram because of its channel." << endl;
	}
	cout << endl;

	Mat Mat_RGBA(dstHistImg.size(), CV_8UC4);
	Mat img;
	img = dstHistImg;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			cv::Vec3b p = img.at<cv::Vec3b>(cv::Point(x, y));
			if (p[0] == 0 && p[1] == 0 && p[2] == 0) {	// 白の時透過
				Mat_RGBA.at<cv::Vec4b>(cv::Point(x, y))
					= cv::Vec4b(p[0], p[1], p[2], 0); //透過
			}
			else {
				Mat_RGBA.at<cv::Vec4b>(cv::Point(x, y))
					= cv::Vec4b(p[0], p[1], p[2], 255); //不透過
			}
		}
	}
	Mat_RGBA.copyTo(dstHistImg);
}


#endif