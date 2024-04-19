// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <opencv2/core/utils/logger.hpp>

wchar_t* projectPath;
int prag = 0;
int choice = -1;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		cv::imshow("image", src);
		cv::waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		cv::imshow(fg.getFoundFileName(), src);
		if (cv::waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	cv::imshow(WIN_SRC, src);
	cv::imshow(WIN_DST, dst);

	cv::waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		cv::imshow("input image", src);
		cv::imshow("negative image", dst);
		cv::waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		cv::imshow("input image", src);
		cv::imshow("negative image", dst);
		cv::waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		cv::imshow("input image", src);
		cv::imshow("gray image", dst);
		cv::waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		cv::imshow("input image", src);
		cv::imshow("H", H);
		cv::imshow("S", S);
		cv::imshow("V", V);

		cv::waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		cv::imshow("input image", src);
		cv::imshow("resized image (without interpolation)", dst1);
		cv::imshow("resized image (with interpolation)", dst2);
		cv::waitKey(0);
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		cv::imshow("input image", src);
		cv::imshow("canny", dst);
		cv::waitKey(0);
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		cv::waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		cv::imshow("source", frame);
		cv::imshow("gray", grayFrame);
		cv::imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		cv::imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				cv::imshow(WIN_DST, frame);
		}
	}

}



/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	cv::imshow(name, imgHist);
}

void add_gray_level(int additive_factor) {

	Mat img = imread("C:/Users/mike/Desktop/Images/kids.bmp", IMREAD_GRAYSCALE);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int new_value = img.at<uchar>(i, j) + additive_factor;
			img.at<uchar>(i, j) = min(max(new_value, 0), 255);
		}
	}


	cv::imshow("add gray level", img);
	cv::waitKey(0);
}

void schimba_nivele_gri(double factor) {
	cv::Mat img = cv::imread("C:/Users/mike/Desktop/Images/kids.bmp", cv::IMREAD_GRAYSCALE);

	img = img * factor;

	cv::imwrite("C:/Users/mike/Desktop/Images/kids2.bmp", img);

	cv::imshow("change gray level", img);
	cv::waitKey(0);
}

void creaza_imagine() {
	cv::Mat img(256, 256, CV_8UC3, cv::Scalar(0, 0, 0));

	cv::Rect cadran1(0, 0, 128, 128); // Stânga-sus
	cv::Rect cadran2(128, 0, 128, 128); // Dreapta-sus
	cv::Rect cadran3(0, 128, 128, 128); // Stânga-jos
	cv::Rect cadran4(128, 128, 128, 128); // Dreapta-jos

	img(cadran1).setTo(cv::Scalar(255, 255, 255)); // Alb
	img(cadran2).setTo(cv::Scalar(0, 0, 255)); // Roșu
	img(cadran3).setTo(cv::Scalar(0, 255, 0)); // Verde
	img(cadran4).setTo(cv::Scalar(0, 255, 255)); // Galben

	cv::imwrite("imagine_cadrane.png", img);

	cv::imshow("patrat cu patrate", img);
	cv::waitKey(0);
}


void grayScaleToBW(int threshold) {
	cv::Mat img = cv::imread("C:/Users/mike/Desktop/Images/kids.bmp", cv::IMREAD_GRAYSCALE);


	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) < threshold)
				img.at<uchar>(i, j) = 0;
			if (img.at<uchar>(i, j) >= threshold)
				img.at<uchar>(i, j) = 255;
		}
	}

	cv::imshow("grayScale to BW", img);
	cv::waitKey(0);

}

void RGBto3Matrices() {
	cv::Mat img = cv::imread("C:/Users/mike/Desktop/Images/flowers_24bits.bmp", cv::IMREAD_COLOR);


	Mat matBlue(img.rows, img.cols, CV_8UC1);
	Mat matGreen(img.rows, img.cols, CV_8UC1);
	Mat matRed(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b culoarePixel = img.at<Vec3b>(i, j);
			matBlue.at<uchar>(i, j) = culoarePixel[0];
			matGreen.at<uchar>(i, j) = culoarePixel[1];
			matRed.at<uchar>(i, j) = culoarePixel[2];
		}
	}

	cv::imshow("Original", img);
	cv::imshow("Blue Photo", matBlue);
	cv::imshow("Green Photo", matGreen);
	cv::imshow("Red Photo", matRed);
	cv::waitKey(0);
}

void RGBtoHSV() {
	cv::Mat img = cv::imread("C:/Users/mike/Desktop/Images/flowers_24bits.bmp", cv::IMREAD_COLOR);

	Mat matHue(img.rows, img.cols, CV_8UC1);
	Mat matSaturation(img.rows, img.cols, CV_8UC1);
	Mat matValue(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b culoarePixel = img.at<Vec3b>(i, j);

			float r = (float)culoarePixel[2] / 255;
			float g = (float)culoarePixel[1] / 255;
			float b = (float)culoarePixel[0] / 255;

			float Max = max(max(r, g), b);
			float Min = min(min(r, g), b);
			float C = Max - Min;

			float V = Max;
			float S;
			float H;

			if (V)
				S = C / V;
			else
				S = 0;

			if (C) {
				if (Max == r) H = 60 * (g - b) / C;
				if (Max == g) H = 120 + 60 * (b - r) / C;
				if (Max == b) H = 240 + 60 * (r - g) / C;
			}
			else
				H = 0;

			if (H < 0) H += 360;

			matHue.at<uchar>(i, j) = H * 255 / 360;
			matSaturation.at<uchar>(i, j) = S * 255;
			matValue.at<uchar>(i, j) = V * 255;
		}
	}

	cv::imshow("Original", img);
	cv::imshow("Hue", matHue);
	cv::imshow("Saturation", matSaturation);
	cv::imshow("Value", matValue);
	cv::waitKey(0);

}

bool isInside(int height, int width) {
	cv::Mat img = cv::imread("C:/Users/mike/Desktop/Images/flowers_24bits.bmp", cv::IMREAD_COLOR);

	if (height < 1 || height > img.rows)
		return false;
	if (width < 1 || width > img.cols)
		return false;
	return true;
}

void histograma() {
	cv::Mat img = cv::imread("C:/Users/mike/Desktop/Images/cameraman.bmp", cv::IMREAD_GRAYSCALE);

	int histograma[256] = {};

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int pixelValue = img.at<uchar>(i, j);
			histograma[pixelValue]++;
		}
	}

	for (int i = 0; i < 256; i++)
		std::cout << histograma[i] << " ";
}

void FDP() {
	cv::Mat img = cv::imread("C:/Users/mike/Desktop/Images/cameraman.bmp", cv::IMREAD_GRAYSCALE);

	int histograma[256] = {};
	int size = img.rows * img.cols;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int pixelValue = img.at<uchar>(i, j);
			histograma[pixelValue]++;
		}
	}

	float FDP[256] = {};

	for (int i = 0; i < 256; i++)
		FDP[i] = (float)histograma[i] / (float)size;

	for (int i = 0; i < 256; i++)
		std::cout << FDP[i] << " ";
}

int isInside2(int row, int col, Mat img) {
	if (row >= 0 && row <= img.rows && col >= 0 && col <= img.cols) {
		return 1;
	}
	return 0;
}

void praguriMultiple() {
	char fname[256] = "C:/Users/mike/Desktop/Images/cameraman.bmp";
	int WH = 5;
	float TH = 0.0003f;
		Mat src = imread(fname, cv::IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		int list[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar g = src.at<uchar>(i, j);
				list[g] = list[g] + 1;
			}
		}

		float FDP[256] = { 0.0 };
		int M = height * width;
		for (int i = 0; i < 256; i++) {
			FDP[i] = ((float)list[i]) / ((float)M);
		}
		std::vector<uchar> vf, mijloace;
		vf.push_back(0);
		mijloace.push_back(0);

		for (int i = WH; i <= 255 - WH; i++) {
			float medie = 0.0;
			int max = 1;
			for (int j = 0; j < 2 * WH + 1; j++) {
				medie += FDP[i + j - WH];
				if (FDP[i + j - WH] > FDP[i]) {
					max = 0;
				}
			}

			medie = medie / ((float)(2 * WH + 1));

			if (max && FDP[i] > medie + TH) {
				vf.push_back(i);
			}
		}

		vf.push_back(255);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar current = src.at<uchar>(i, j);
				for (int k = 0; k < vf.size(); k++)
				{
					if (current >= vf[k] && current <= vf[k + 1])
					{
						if (current - vf[k] < vf[k + 1] - current)
							dst.at<uchar>(i, j) = vf[k];
						else
							dst.at<uchar>(i, j) = vf[k + 1];
					}

				}
			}
		}
		showHistogram("Histograma", list, width, height);
		cv::imshow("Praguri Multiple", dst);
		cv::imshow("Initial", src);
		cv::waitKey();
	
}

void floydSteinberg() {
	int WH = 5;
	float TH = 0.0003f;
	char fname[MAX_PATH] = "C:/Users/mike/Desktop/Images/saturn.bmp";

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		Mat dstFin = Mat(height, width, CV_8UC1);

		int list[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar g = src.at<uchar>(i, j);
				list[g] = list[g] + 1;
			}
		}

		float FDP[256] = { 0.0 };
		int M = height * width;
		for (int i = 0; i < 256; i++) {
			FDP[i] = ((float)list[i]) / ((float)M);
		}
		std::vector<uchar> vf, mijloace;
		vf.push_back(0);
		mijloace.push_back(0);

		for (int i = WH; i <= 255 - WH; i++) {
			float medie = 0.0;
			int max = 1;
			for (int j = 0; j < 2 * WH + 1; j++) {
				medie += FDP[i + j - WH];
				if (FDP[i + j - WH] > FDP[i]) {
					max = 0;
				}
			}

			medie = medie / ((float)(2 * WH + 1));

			if (max && FDP[i] > medie + TH) {
				vf.push_back(i);
			}
		}

		vf.push_back(255);

		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				uchar current = src.at<uchar>(i, j);
				for (int k = 0; k < vf.size(); k++){
					if (current >= vf[k] && current <= vf[k + 1]){
						if (current - vf[k] < vf[k + 1] - current)
							dst.at<uchar>(i, j) = vf[k];
						else
							dst.at<uchar>(i, j) = vf[k + 1];
					}

				}
			}
		}

		dstFin = dst.clone();

		int a;
		for (int i = 0; i < height - 1; i++) {
			for (int j = 0; j < width - 1; j++) {
				uchar oldpixel = src.at<uchar>(i, j);
				uchar newpixel = dst.at<uchar>(i, j);

				int error = oldpixel - newpixel;
				
				if (isInside2(i, j + 1, dstFin) == 1) {
					a = dstFin.at<uchar>(i, j + 1) + (7 * error / 16);
					if (a > 255) {
						dstFin.at<uchar>(i, j + 1) = 255;
					}
					else if (a < 0) {
						dstFin.at<uchar>(i, j + 1) = 0;
					}
					else {
						dstFin.at<uchar>(i, j + 1) = a;
					}
				}

				if (isInside2(i + 1, j - 1, dstFin) == 1) {
					a = dstFin.at<uchar>(i + 1, j - 1) + (3 * error / 16);
					if (a > 255) {
						dstFin.at<uchar>(i + 1, j - 1) = 255;
					} else if (a < 0) {
						dstFin.at<uchar>(i + 1, j - 1) = 0;
					} else {
						dstFin.at<uchar>(i + 1, j - 1) = a;
					}
				}

				if (isInside2(i + 1, j, dstFin) == 1) {
					a = dstFin.at<uchar>(i + 1, j) + (5 * error / 16);
					if (a > 255) {
						dstFin.at<uchar>(i + 1, j) = 255;
					} else if (a < 0) {
						dstFin.at<uchar>(i + 1, j) = 0;
					} else {
						dstFin.at<uchar>(i + 1, j) = a;
					}
				}

				if (isInside2(i + 1, j + 1, dstFin) == 1) {
					a = dstFin.at<uchar>(i + 1, j + 1); +(7 * error / 16);
					if (a > 255) {
						dstFin.at<uchar>(i + 1, j + 1) = 255;
					} else if (a < 0) {
						dstFin.at<uchar>(i + 1, j + 1) = 0;
					} else {
						dstFin.at<uchar>(i + 1, j + 1) = a;
					}
				}
			}

		}


		showHistogram("Histograma", list, width, height);
		cv::imshow("Praguri Multiple", dst);
		cv::imshow("Floyd Steinberg", dstFin);
		cv::imshow("Initial", src);
		cv::waitKey();
}

int arieObiect(Vec3b pixel, Mat img) {
	int arie = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixelCurent = img.at<Vec3b>(i, j);
			if (pixel == pixelCurent)
				arie++;

		}
	}
	return arie;
}

void centruMasaObiect(Vec3b pixel, Mat img, int arie, int* r, int* c) {
	int rLocal = 0; int cLocal = 0;
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixelCurent = img.at<Vec3b>(i, j);
			if (pixel == pixelCurent) {
				cLocal += i;
				rLocal += j;
			}
		}
	}

	*r = rLocal / (float)arie;
	*c = cLocal / (float)arie;
}

float axaDeAlungire(Vec3b pixel, Mat pic, float rLocal, float cLocal) {
	float numarator = 0.0;
	float numitor = 0.0;
	for (int i = 0; i < pic.rows; i++){
		for (int j = 0; j < pic.cols; j++){
			Vec3b pixelCurent = pic.at<Vec3b>(i, j);
			if (pixel[0] == pixelCurent[0] && pixel[1] == pixelCurent[1] && pixel[2] == pixelCurent[2]) {
				numarator += 2 * (i - rLocal) * (j - cLocal);
				numitor += ((i - rLocal) * (i - rLocal)) - ((j - cLocal) * (j - cLocal));
			}

		}
	}

	return atan2(2 * numarator, numitor) / 2.0;
}

bool esteConturObiect(Vec3b pixel, Mat img, int x, int y) {
	if (isInside2(x + 1, y + 1, img) == 1 && img.at<Vec3b>(x + 1, y + 1) != pixel)
		return true;
	if (isInside2(x, y + 1, img) == 1 && img.at<Vec3b>(x, y + 1) != pixel)
		return true;
	if (isInside2(x - 1, y + 1, img) == 1 && img.at<Vec3b>(x - 1, y + 1) != pixel)
		return true;
	if (isInside2(x - 1, y, img) == 1 && img.at<Vec3b>(x - 1, y) != pixel)
		return true;
	if (isInside2(x - 1, y - 1, img) == 1 && img.at<Vec3b>(x - 1, y - 1) != pixel)
		return true;
	if (isInside2(x, y - 1, img) == 1 && img.at<Vec3b>(x, y - 1) != pixel)
		return true;
	if (isInside2(x + 1, y - 1, img) == 1 && img.at<Vec3b>(x + 1, y - 1) != pixel)
		return true;
	if (isInside2(x + 1, y, img) == 1 && img.at<Vec3b>(x + 1, y) != pixel)
		return true;
	return false;

}

int perimetruObiect(Vec3b pixel, Mat img) {
	int perimetru = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixelCurent = img.at<Vec3b>(i, j);
			if (pixel == pixelCurent && esteConturObiect(pixelCurent, img, i, j) == true) {
				perimetru++;
			}
		}
	}

	return perimetru;
}

float factorSubtiere(Vec3b pixel, Mat img) {
	float factor = 0.0;
	int arie = arieObiect(pixel, img); std::cout << arie << '\n';
	int perimetru = perimetruObiect(pixel, img); std::cout << perimetru << '\n';

	factor = (4 * 3.14 * arie) / (float)(perimetru * perimetru);
	return factor;
}


float aspectRatio(Vec3b pixel, Mat img) {
	int ok = 0;
	int cMin = 9999; int cMax = -1;
	int rMin = 9999; int rMax = -1;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (ok == 1) continue;
			if (esteConturObiect(pixel, img, i, j) == true) rMin = i;
		}
	}
	
	ok = 0;
	for (int i = img.rows; i > 0; i--) {
		for (int j = 0; j < img.cols; j++) {
			if (ok == 1) continue;
			if (esteConturObiect(pixel, img, i, j) == true) rMax = i;
		}
	}

	ok = 0;
	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {
			if (ok == 1) continue;
			if (esteConturObiect(pixel, img, i, j) == true) cMin = j;
		}
	}

	ok = 0;
	for (int i = img.cols; i > 0; i--) {
		for (int j = 0; j < img.rows; j++) {
			if (ok == 1) continue;
			if (esteConturObiect(pixel, img, i, j) == true) cMax = j;
		}
	}

	float numitor = (float)(rMin - rMax + 1);
	float numarator = (float)(cMin - cMax + 1);

	return numarator / numitor;
}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN) {

		Vec3b pixelColor = src->at<Vec3b>(y, x);
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			pixelColor[2],
			pixelColor[1],
			pixelColor[0]);

		int arie = arieObiect(pixelColor, *src);
		std::cout << "Arie: " << arie << '\n';
		int rLocal = 0, cLocal = 0;
		centruMasaObiect(pixelColor, *src, arie, &rLocal, &cLocal);
		std::cout << "CG Rand: " << rLocal << '\n' << "CG Coloana: " << cLocal << '\n';
		float axaAlungire = axaDeAlungire(pixelColor, *src, rLocal, cLocal);
		std::cout << "Axa de Alungire: " << axaAlungire << '\n';
		int perimetru = perimetruObiect(pixelColor, *src);
		std::cout << "Perimetru: " << perimetru << '\n';
		float factorSubtiereVal = factorSubtiere(pixelColor, *src);
		std::cout << "Factor subtiere: " << factorSubtiereVal << '\n';
		float aspectR = aspectRatio(pixelColor, *src);
		std::cout << "Aspect Ratio: " << aspectR << '\n';

		std::cout << '\n';
	}


}

void labelImage(cv::Mat img) {
	int label = 0;
	//Mat labels = Mat::zeros(img.size(), CV_32SC1);
	Mat labels(img.rows, img.cols, CV_32SC1, Scalar(0));

	Mat dst(img.rows, img.cols, CV_8UC3);



	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
				if (img.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
					int di[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
					int dj[8] = { -1, 0, 1, -1 , 1, -1, 0, 1 };
					label++;
					std::queue<Point2i> Q;
					labels.at<int>(i, j) = label;
					Q.push({ i, j });

					while (!Q.empty()) {
						Point q = Q.front();
						Q.pop();

						for (int k = 0; k < 8; k++) {
							//Point2i neighbor(q.x + di[k], q.y + dj[k]);
							if (isInside2(q.x + di[k], q.y + dj[k], img) == true) {
								if (img.at<uchar>(q.x + di[k], q.y + dj[k]) == 0 && labels.at<int>(q.x + di[k], q.y + dj[k]) == 0) {
									labels.at<int>(q.x + di[k], q.y + dj[k]) = label;
									Q.push(Point(q.x + di[k], q.y + dj[k]));
								}
							}
						}
					}
				}
		}
	}

	std::vector<cv::Vec3b> colors;
	for (int i = 0; i <= label; i++) {
		colors.push_back(cv::Vec3b(rand() % 256, rand() % 256, rand() % 256));
	}

	std::cout << label << '\n';

	// Create a color image from the labels
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) != 0 && labels.at<int>(i, j) == 0)
				dst.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
			else dst.at<cv::Vec3b>(i, j) = colors[labels.at<int>(i, j)];
		}
	}

	cv::imshow("Original", img);
	cv::imshow("Labeled", dst);
	cv::waitKey(0);
}

void twoPasses(cv::Mat img) {
	int label = 0;

	Mat labels(img.rows, img.cols, CV_32SC1, Scalar(0));

	Mat dst(img.rows, img.cols, CV_8UC3);

	std::vector<std::vector<int>> edges;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
				std::vector<int> L;

			}
		}
	}
}

void trackBorder() {
	char path[MAX_PATH];
	while (openFileDlg(path)) {
		Mat src = imread(path, IMREAD_GRAYSCALE);
		Mat dst(src.size(), CV_8UC1, Scalar(255));

		int height = src.rows;
		int width = src.cols;
		Point2i P0, P1;
		Point2i actual, anterior;
		bool exist = false;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 0) {
					P0.x = i;
					P0.y = j;
					exist = true;
					break;
				}
			}
			if (exist) break;
		}

		std::cout << "Primul punct gasit: " << P0 << '\n';


		int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
		int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

		int dir = 7;
		std::vector<int> codInlantuit, codDerivat;
		int n = 0;
		actual = P0;
		int ii, jj;

		do {
			n++;

			if (dir % 2 == 1)
				dir = (dir + 6) % 8;
			else if (dir % 2 == 0)
				dir = (dir + 7) % 8;

			ii = actual.x + di[dir];
			jj = actual.y + dj[dir];

			while (src.at<uchar>(ii, jj) == 255) {
				dir = (dir + 1) % 8;
				ii = actual.x + di[dir];
				jj = actual.y + dj[dir];
			}

			if (n == 1) {
				P1.x = P0.x + di[dir];
				P1.y = P0.y + dj[dir];
				actual = P1;
			}
			else {
				anterior = actual;
				actual.x += di[dir];
				actual.y += dj[dir];
			}

			codInlantuit.push_back(dir);
			dst.at<uchar>(actual.x, actual.y) = 0;

		} while (!((actual == P1) && (anterior == P0)));


		for (int i = 0; i < codInlantuit.size() - 1; i++) {
			codDerivat.push_back((codInlantuit[i + 1] - codInlantuit[i] + 8) % 8);
		}

		std::cout << "Codul inlantuit: ";
		for (int i = 0; i < codInlantuit.size() - 1; i++) {
			std::cout << codInlantuit[i] << " ";
		}

		std::cout << "\nDerivata: ";
		for (int i = 0; i < codDerivat.size() - 1; i++) {
			std::cout << codDerivat[i] << " ";
		}
		std::cout << '\n';

		imshow("Original", src);
		imshow("Destinatie", dst);
		waitKey(0);

	}
}

Mat diluare(Mat src) {
	Mat dst = src.clone();
	int height = src.rows;
	int width = src.cols;

	int di[4] = { 0, -1,  0, 1 };
	int dj[4] = { 1,  0, -1, 0 };

	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 0; j < width - 1; j++)
		{
			uchar pixel = src.at<uchar>(i, j);
			if (pixel == 0) {

				for (int k = 0; k < 4; k++)
				{
					if (isInside2(i + dj[k], j + di[k], dst))
						dst.at<uchar>(i + dj[k], j + di[k]) = 0;

				}

				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return dst;
}

Mat eroziune(Mat src) {
	Mat dst = src.clone();
	int height = src.rows;
	int width = src.cols;

	int di[4] = { 0, -1,  0, 1};
	int dj[4] = { 1,  0, -1, 0};

	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 0; j < width - 1; j++)
		{
			uchar pixel = src.at<uchar>(i, j);
			if (pixel == 255) {

				for (int k = 0; k < 4; k++)
				{
					if (isInside2(i + dj[k], j + di[k], dst))
						dst.at<uchar>(i + dj[k], j + di[k]) = 255;

				}

				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return dst;
}

Mat deschidere(Mat src) {
	Mat mat1 = diluare(src);
	Mat mat2 = eroziune(mat1);
	return mat2;
}

Mat inchidere(Mat src) {
	Mat mat1 = eroziune(src);
	Mat mat2 = diluare(mat1);
	return mat2;
}

void medieDeviatie() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int M = height * width;

		int hist[256] = { 0 };
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hist[src.at<uchar>(i, j)]++;
			}
		}

		float g = 0;

		for (int i = 0; i < 256; i++)
		{
			g = g + (i * hist[i]);
		}
		g = (float)g / M;


		float deviation = 0.0;
		for (int i = 0; i < 256; i++)
		{
			deviation = deviation + ((float)(i - g) * (i - g) * hist[i]);

		}
		deviation = (float)deviation / M;
		deviation = sqrt(deviation);

		int histC[256] = { 0 };
		for (int i = 0; i < 256; i++)
		{
			for (int j = 0; j < i; j++)
			{
				histC[i] += hist[j];
			}
		}


		std::cout << "Media: " << g << '\n';
		std::cout << "Deviatia: " << deviation << '\n';

		showHistogram("Histograma", hist, 255, 255);
		showHistogram("Histograma cumulativa", histC, 255, 255);

	}
}

void binarizareImagine() {
	char fname[MAX_PATH];
	
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		int hist[256] = { 0 };
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				hist[src.at<uchar>(i, j)]++;
			}
		}

		int iMin = INT_MAX;
		int iMax = INT_MIN;

		for (int i = 0; i < 256; i++) {
			if (hist[i] > 0) {
				iMin = i;
				break;
			}
		}

		for (int i = 255; i >= 0; i--) {
			if (hist[i] > 0) {
				iMax = i;
				break;
			}
		}

		std::cout << "iMin: " << iMin << '\n';
		std::cout << "iMax: " << iMax << '\n';


		float treshold = (iMin + iMax) / 2.0;
		float prevTreshold;
		do{
			float u1 = 0, u2 = 0;
			float g1 = 0, g2 = 0;
			for (int i = iMin; i < treshold; i++) {
				g1 += i * hist[i];
				u1 += hist[i];
			}

			for (int i = treshold; i <= iMax; i++) {
				g2 += i * hist[i];
				u2 += hist[i];
			}

			g1 /= u1;
			g2 /= u2;
			prevTreshold = treshold;
			treshold = (g1 + g2) / 2.0;

		} while (abs(treshold - prevTreshold) > 0.1);

		Mat dst(height, width, CV_8UC1, Scalar(255));

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) >= treshold) {
					dst.at<uchar>(i, j) = 255;
				}
				else if (src.at<uchar>(i, j) < treshold) {
					dst.at<uchar>(i, j) = 0;
				}

			}
		}

		imshow("Original", src);
		imshow("Binarizata", dst);
		waitKey(0);
	}
}

void brightnessContrast(int gOutMin, int gOutMax, int brighnessAmount) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dstAfterBrightness(src.rows, src.cols, CV_8UC1, Scalar(255));

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int pixel = src.at<uchar>(i, j) + brighnessAmount;
				if (pixel > 255)  dstAfterBrightness.at<uchar>(i, j) = 255;
				else {
					if (sum < 0) dstAfterBrightness.at<uchar>(i, j) = 0;
					else dstAfterBrightness.at<uchar>(i, j) = pixel;
				}
			}
		}

		int hist[256] = { 0 };
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hist[src.at<uchar>(i, j)]++;
			}
		}

		int gInMin = INT_MAX;
		int gInMax = INT_MIN;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) < gInMin) {
					gInMin = src.at<uchar>(i, j);
				}
				if (src.at<uchar>(i, j) > gInMax) {
					gInMax = src.at<uchar>(i, j);
				}

			}
		}

		Mat dstAfterContrast(src.rows, src.cols, CV_8UC1, Scalar(255));

		float rap = (float)(gOutMax - gOutMin) / (gInMax - gInMin);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int gIn = src.at<uchar>(i, j);
				int gOut = gOutMin + (gIn - gInMin) * rap;
				if (gOut > 255)
				{
					dstAfterContrast.at<uchar>(i, j) = 255;

				}
				else {

					if (sum < 0)
					{
						dstAfterContrast.at<uchar>(i, j) = 0;

					}
					else
					{
						dstAfterContrast.at<uchar>(i, j) = gOut;
					}
				}
			}
		}

		int histBrightness[256] = { 0 };
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				histBrightness[dstAfterBrightness.at<uchar>(i, j)]++;
			}
		}

		int histContrast[256] = { 0 };
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				histContrast[dstAfterContrast.at<uchar>(i, j)]++;
			}
		}
		imshow("Initial image", src);
		imshow("After Brightness", dstAfterBrightness);
		imshow("After Contrast", dstAfterContrast);
		showHistogram("MyHist", hist, 256, 256);
		showHistogram("MyHist Brightness", histBrightness, 256, 256);
		showHistogram("MyHist Contrast", histContrast, 256, 256);
		waitKey(0);

	}
}


void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		cv::imshow("My Window", src);

		// Wait until user press some key
		cv::waitKey(0);
	}
}


int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - Add Gray Level\n");
		printf(" 14 - Change Gray Level\n");
		printf(" 15 - Create image\n");
		printf(" 16 - Gray Scale to B&W\n");
		printf(" 17 - 3 R G B Photos\n");
		printf(" 18 - RGB to HSV\n");
		printf(" 19 - isInside\n");
		printf(" 20 - Histograma sau FDP\n");
		printf(" 21 - Praguri multiple\n");
		printf(" 22 - Floyd Steinberg\n");
		printf(" 23 - Arii si toate alea\n");
		printf(" 24 - Algoritm de traversare in latime\n");
		printf(" 25 - Algoritm de urmarire a conturului\n");
		printf(" 26 - Dilatare\n");
		printf(" 27 - Eroziune\n");
		printf(" 28 - Deschidere\n");
		printf(" 29 - Inchidere\n");
		printf(" 30 - Dilatare, Eroziune, Deschidere, Inchidere\n");
		printf(" 31 - Standard Deviation\n");
		printf(" 32 - Binarizare imagine\n");
		printf(" 33 - Brightness & Contrast\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testNegativeImage();
			break;
		case 4:
			testNegativeImageFast();
			break;
		case 5:
			testColor2Gray();
			break;
		case 6:
			testImageOpenAndSave();
			break;
		case 7:
			testBGR2HSV();
			break;
		case 8:
			testResize();
			break;
		case 9:
			testCanny();
			break;
		case 10:
			testVideoSequence();
			break;
		case 11:
			testSnap();
			break;
		case 12:
			testMouseClick();
			break;
		case 13:
			add_gray_level(1);
			break;
		case 14:
			schimba_nivele_gri(1.2f);
			break;
		case 15:
			creaza_imagine();
			break;
		case 16:
			std::cout << "prag =";
			std::cin >> prag;
			grayScaleToBW(prag);
			break;
		case 17:
			RGBto3Matrices();
			break;
		case 18:
			RGBtoHSV();
			break;
		case 19:
			int i, j;
			std::cout << "rand = "; std::cin >> i;
			std::cout << "coloana = "; std::cin >> j;
			if (isInside(i, j) == true) {
				cv::Mat imgMerge = imread("C:/Users/mike/Desktop/Images/inauntru.bmp", cv::IMREAD_COLOR);
				cv::imshow("Is inside", imgMerge);
				cv::waitKey(0);
			}
			else {
				cv::Mat imgNuMerge = imread("C:/Users/mike/Desktop/Images/inafara.bmp", cv::IMREAD_COLOR);
				cv::imshow("Not inside", imgNuMerge);
				cv::waitKey(0);
			}
			getchar();  
			break;
		case 20:
			std::cout << "0 -> histograma\n1 -> FDP\n"; std::cin >> choice;
			if (choice == 0) histograma();
			if (choice == 1) FDP();
			std::cin >> i;
			break;
		case 21:
			praguriMultiple();
			break;
		case 22:
			floydSteinberg();
			break;
		case 23: 
			testMouseClick();
			break;
		case 24: {
			Mat imgC24 = imread("C:/Users/mike/Desktop/Images/Etichete/letters.bmp", IMREAD_GRAYSCALE);
			labelImage(imgC24);
			break;
		}
		case 25:
			trackBorder();
			break;
		case 26: {
			char fname[MAX_PATH];
			while (openFileDlg(fname)) {

				Mat imgDil = imread(fname, IMREAD_GRAYSCALE);

				Mat dstDil = diluare(imgDil);

				Mat dstDil5x;
				for (int i = 0; i < 4; i++)
					dstDil5x = diluare(dstDil);

				imshow("Original", imgDil);
				imshow("Diluata", dstDil);
				imshow("Diluata 5x", dstDil5x);
				waitKey(0);
			}
		}

		case 27: {
			char fname[MAX_PATH];
			while (openFileDlg(fname)) {

				Mat imgEroz = imread(fname, IMREAD_GRAYSCALE);

				Mat dstEroz = eroziune(imgEroz);

				Mat dstEroz5x;
				for (int i = 0; i < 4; i++) {
					dstEroz5x = eroziune(dstEroz);
				}

				imshow("Original", imgEroz);
				imshow("Erodata", dstEroz);
				imshow("Erodata 5x", dstEroz5x);
				waitKey(0);
			}
		}

		case 28: {
			char fname[MAX_PATH];
			while (openFileDlg(fname)) {
				Mat imgDeschidere = imread(fname, IMREAD_GRAYSCALE);

				Mat dstDeschidere = deschidere(imgDeschidere);

				imshow("Original", imgDeschidere);
				imshow("Deschidere", dstDeschidere);
				waitKey(0);
			}
		}

		case 29: {
			char fname[MAX_PATH];
			while (openFileDlg(fname)) {
				Mat imgInchidere = imread(fname, IMREAD_GRAYSCALE);

				Mat dstInchidere = inchidere(imgInchidere);

				imshow("Original", imgInchidere);
				imshow("Inchidere", dstInchidere);
				waitKey(0);
			}
		}

		case 30: {
			int n;
			std::cout << "Insert number of operations: "; std::cin >> n;

			std::string dilString = "Diluare"; dilString += std::to_string(n); dilString += "x";
			std::string erozString = "Eroziune"; erozString += std::to_string(n); erozString += "x";

			char fname[MAX_PATH];

			while (openFileDlg(fname)) {
				Mat imgOp = imread(fname, IMREAD_GRAYSCALE);

				Mat dil1 = diluare(imgOp);
				Mat eroz1 = eroziune(imgOp);
				Mat inch1 = inchidere(imgOp);
				Mat desch1 = deschidere(imgOp);

				Mat dilN;
				Mat erozN;

				for (int i = 0; i < n - 1; i++) {
					dilN = diluare(dil1);
					erozN = eroziune(eroz1);
				}

				imshow("Original", imgOp);
				imshow("Diluare", dil1);
				imshow(dilString, dilN);
				imshow(erozString, erozN);
				imshow("Eroziune", eroz1);
				imshow("Deschidere", desch1);
				imshow("Inchidere", inch1);
				waitKey(0);
			} 
		}

		case 31: {
			medieDeviatie();
			break;
		}

		case 32: {
			binarizareImagine();
			break;
		}

		case 33: {
			int gIn;
			int gOut;
			int brightnessAmount;
			std::cout << "gIn: "; std::cin >> gIn;
			std::cout << "gOut: "; std::cin >> gOut;
			std::cout << "Brightness Amount: "; std::cin >> brightnessAmount;
			brightnessContrast(gIn, gOut, brightnessAmount);
			break;
		}
		
		}
	} while (op != 0);
	return 0;
}