#include "main.h"
#include "MakeNoiseImage.h"
#include "EstimatedParameterMRF.h"
#include "EstimatedParameterHMRF.h"
#include "OutputDataToExcel.h"
#include "Image_Histgram.h"


int main() {
	/* �摜�̓��� */
	Input_Image();

	clock_t start, start2, end, end2;	// �������ԕ\���p
	double time_difference = 0.0, time_difference2 = 0.0;
	//--- �摜���� ---------------------------------------------------------------------------
	if (IMAGE_NUMBER == 0) {
		Inpainting_and_OutputToExcel();
	}
	else {
		/* �K�E�X�m�C�Y�t�� */
		vector<Mat> K_Image_dst;		// k���̗򉻉摜
		cout << IMAGE_NUMBER << "���̗򉻉摜�����c" << endl;		// ���s�m�F�p
		GaussianNoiseImage(IMAGE_NUMBER, Image_src, K_Image_dst);
		K_Image_dst[0].copyTo(Image_dst);
		CreateAverageImage(IMAGE_NUMBER, Image_dst_average, K_Image_dst, MAX_INTENSE);	// ���ω摜����
		Image_dst.copyTo(Image_dst_MRF);
		Image_dst.copyTo(Image_dst_HMRF);
		Image_dst.copyTo(Image_dst_NLM);
		Image_dst.copyTo(Image_dst_NLMdef);

		/* MRF(�}���R�t�m����)�ɂ��m�C�Y���� */
		cout << "�p�����[�^���肠��(MRF)�̃m�C�Y�����c" << endl;	// ���s�m�F�p
		start = clock();
		DenoisingMRF(Image_dst_MRF, K_Image_dst, IMAGE_NUMBER, MAX_INTENSE);
		end = clock();
		time_difference = (double)end - (double)start;
		const double time = static_cast<double>(time_difference) / CLOCKS_PER_SEC * 1000.0;
		cout << "time : " << time << "[ms]" << endl;
		cout << endl;

		/* HMRF(�K�w�^�}���R�t�m����)�ɂ��m�C�Y���� */
		cout << "�p�����[�^���肠��(HMRF)�̃m�C�Y�����c" << endl;	// ���s�m�F�p
		start2 = clock();
		DenoisingHMRF(Image_dst_HMRF, K_Image_dst, IMAGE_NUMBER, MAX_INTENSE);
		end2 = clock();
		time_difference2 = (double)end2 - (double)start2;
		const double time2 = static_cast<double>(time_difference2) / CLOCKS_PER_SEC * 1000.0;
		cout << "time : " << time2 << "[ms]" << endl;
		cout << endl;

		/* NonLocalMeans�ɂ��m�C�Y���� */
		cout << "Non-Local Means�̃m�C�Y�����c" << endl;	// ���s�m�F�p
		//void cv::fastNlMeansDenoisingColored(src, dst, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
		fastNlMeansDenoisingColored(Image_dst_average, Image_dst_NLMdef);
		double best_h = (double)NoiseSigma / (double)sqrt(IMAGE_NUMBER);
		cout << " best_h = " << (double)best_h << endl;
		//fastNlMeansDenoisingColored(Image_dst_average, Image_dst_NLM, best_h, 3, 7, 21);
		fastNlMeansDenoisingColored(Image_dst_average, Image_dst_NLM, best_h, best_h, 7, 21);
		cout << endl;

		/* �摜�̕]�� */
		cout << "�m�C�Y�摜 �� ���摜" << endl;			// ���s�m�F�p
		MSE_PSNR_SSIM(Image_src, Image_dst);
		cout << "NLM�C���摜(�f�t�H���g) �� ���摜" << endl;		// ���s�m�F�p
		MSE_PSNR_SSIM(Image_src, Image_dst_NLMdef);
		cout << "NLM�C���摜(�œK��) �� ���摜" << endl;		// ���s�m�F�p
		MSE_PSNR_SSIM(Image_src, Image_dst_NLM);
		cout << "MRF�C���摜 �� ���摜" << endl;		// ���s�m�F�p
		MSE_PSNR_SSIM(Image_src, Image_dst_MRF);
		cout << "HMRF�C���摜 �� ���摜" << endl;		// ���s�m�F�p
		MSE_PSNR_SSIM(Image_src, Image_dst_HMRF);
		cout << "�m�C�Y���ω摜 �� ���摜" << endl;		// ���s�m�F�p
		MSE_PSNR_SSIM(Image_src, Image_dst_average);
	}

	/* �o�̓q�X�g�O�����쐬 */
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

	/* �摜�̏o�� */
	Output_Image();
	return 0;
}


// �摜�̓���
void Input_Image() {
	//string file_src = "C:\\Users\\mon25\\Desktop\\DenoisingImage\\src.jpg";		// ���͉摜�̃t�@�C����
	string file_src = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\src.jpg";		// ���͉摜�̃t�@�C����
	Image_src = imread(file_src, 1);						// ���͉摜�i�J���[�j�̓ǂݍ���

	/* �p�����[�^��` */
	WIDTH = Image_src.cols;
	HEIGHT = Image_src.rows;
	MAX_DATA = WIDTH * HEIGHT;
	Image_dst = Mat(Size(WIDTH, HEIGHT), CV_8UC3);			// �o�͉摜�i�J���[�j�̏������ݒ�
	Image_dst_average = Mat(Size(WIDTH, HEIGHT), CV_8UC3);
	Image_dst_MRF = Mat(Size(WIDTH, HEIGHT), CV_8UC3);
	Image_dst_HMRF = Mat(Size(WIDTH, HEIGHT), CV_8UC3);
}
// �摜�̏o��
void Output_Image() {
	//string file_dst = "C:\\Users\\mon25\\Desktop\\DenoisingImage\\dst.jpg";		// �o�͉摜�̃t�@�C����
	//string file_dst2 = "C:\\Users\\mon25\\Desktop\\DenoisingImage\\dst_MRF.jpg";
	//string file_dst3 = "C:\\Users\\mon25\\Desktop\\DenoisingImage\\dst_HMRF.jpg";
	//string file_dst4 = "C:\\Users\\mon25\\Desktop\\DenoisingImage\\dst_average.jpg";
	//string file_dst5 = "C:\\Users\\mon25\\Desktop\\DenoisingImage\\dst_NLM.jpg";
	string file_dst = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst.jpg";		// �o�͉摜�̃t�@�C����
	string file_dst2 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_MRF.jpg";
	string file_dst3 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_HMRF.jpg";
	string file_dst4 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_average.jpg";
	string file_dst5 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_NLM.jpg";
	string file_dst6 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_hist_src.jpg";	// �i�o�̓q�X�g�O�����j
	string file_dst7 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_hist_dst.jpg";
	string file_dst8 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_hist_dst_MRF.jpg";
	string file_dst9 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_hist_dst_HMRF.jpg";
	string file_dst10 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_hist_dst_average.jpg";
	string file_dst11 = "C:\\Users\\Yuki Momma\\Desktop\\DenoisingImage\\dst_hist_dst_NLM.jpg";

	/* �E�B���h�E���� */
	namedWindow(win_src, WINDOW_AUTOSIZE);
	namedWindow(win_dst, WINDOW_AUTOSIZE);
	namedWindow(win_dst2, WINDOW_AUTOSIZE);
	namedWindow(win_dst3, WINDOW_AUTOSIZE);
	namedWindow(win_dst4, WINDOW_AUTOSIZE);

	/* �摜�̕\�� & �ۑ� */
	imshow(win_src, Image_src);				// ���͉摜��\��
	imshow(win_dst, Image_dst);				// �o�͉摜��\��(�m�C�Y�摜)
	imwrite(file_dst, Image_dst);			// �������ʂ̕ۑ�
	imshow(win_dst2, Image_dst_MRF);		// �o�͉摜��\��(�m�C�Y�����摜MRF)
	imwrite(file_dst2, Image_dst_MRF);		// �������ʂ̕ۑ�
	imshow(win_dst3, Image_dst_HMRF);		// �o�͉摜��\��(�m�C�Y�����摜HMRF)
	imwrite(file_dst3, Image_dst_HMRF);		// �������ʂ̕ۑ�
	imwrite(file_dst4, Image_dst_average);	// �������ʂ̕ۑ�(�m�C�Y���ω摜)
	imshow(win_dst4, Image_dst_NLM);		// �o�͉摜��\��(�m�C�Y�����摜MRF)
	imwrite(file_dst5, Image_dst_NLM);		// �������ʂ̕ۑ�

	imwrite(file_dst6, Image_hist_src);		// �q�X�g�O�����̕ۑ�
	imwrite(file_dst7, Image_hist_dst);
	imwrite(file_dst8, Image_hist_dst_MRF);
	imwrite(file_dst9, Image_hist_dst_HMRF);
	imwrite(file_dst10, Image_hist_dst_average);
	imwrite(file_dst11, Image_hist_dst_NLM);

	waitKey(0); // �L�[���͑҂�
}


// MSE&PSNR&SSIM�ɂ��摜�]��
void MSE_PSNR_SSIM(Mat& Original, Mat& Inpaint) {
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
	SSIMcalc(SSIM, beforeIMG, afterIMG);

	/* �]�����ʕ\�� */
	cout << "--- �]�� ------------------------------------------" << endl;
	cout << " MSE  : " << MSE << endl;
	cout << " PSNR : " << PSNR << endl;
	cout << " SSIM : " << SSIM << endl;
	cout << "---------------------------------------------------" << endl;
	cout << endl;
}
// SSIM�Z�o
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

	/* SSIM����(RGB) */
	double SSIM;
	SSIM = (double)mssim[0] + (double)mssim[1] + (double)mssim[2];
	ssim = (double)SSIM / 3.0;
}