/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>
#include <numeric>

#define MIN_RAD 23
#define MAX_RAD 125
#define RADIUS_RANGE 103

using namespace std;
using namespace cv;

/** Function Headers */
std::vector<Rect> detectAndDisplay( Mat frame,
											 std::vector<Rect> dartboards );

std::vector<Rect> chooseGroundTruths(int imageNumber);

float f1( std::vector<Rect> dartboards,
				  std::vector<Rect> groundTruths);

void sobel( cv::Mat &input,
							cv::Mat &sobelX,
							cv::Mat &sobelY,
							cv::Mat &sobelMag,
							cv::Mat &sobelGr );

void GaussianBlur(cv::Mat &input,
									int size,
									cv::Mat &blurredOutput);

void thresholdMag(cv::Mat &input,
							 int threshVal);

void houghCircle(cv::Mat &edges,
										cv::Mat &thetas,
										cv::Mat &grey,
										cv:: Mat &space);

/*void houghLines(cv::Mat&sobelMag,
								cv::&sobelGrad,
								cv::&lines,
								cv::&houghSpaceLines); */

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;
int imageNumber;

/** @function main */
int main( int argc, const char** argv )
{
	std::vector<Rect> dartboards;
  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	//Get the image number and gTNumber
	imageNumber = atoi (argv[2]);

	std::vector<Rect> groundTruths = chooseGroundTruths(imageNumber);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect dartboards and Display Result
	dartboards = detectAndDisplay( frame , dartboards );
	// 3.5 f1Score
	float f1Score = f1( dartboards, groundTruths );

	//print f1 score
	printf( "f1 Score %f\n", f1Score );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

//function to select correct ground truths
std::vector<Rect> chooseGroundTruths(int imageNumber)
{
	std::vector<std::vector<Rect> > allGroundTruths =
	{
    {Rect(444,15,153,175)}, //0
		{Rect(196,132,195,191)}, //1
    {Rect(101,95,91,93)}, //2
    {Rect(323,148,67,72)}, //3
    {Rect(185,94,213,203)}, //4
    {Rect(433,141,110,111)}, //5
    {Rect(210,115,63,66)}, //6
    {Rect(254,170,150,144)}, //7
    {Rect(842,218,117,119), Rect(67,252,60,89)}, //8
    {Rect(203,48,231,232)}, //9
    {Rect(92,104,95,109), Rect(585,127,56,86), Rect(916,149,37,65)}, //10
    {Rect(174,105,59,56), Rect(433,113,40,74)}, //11
    {Rect(156,77,60,137)}, //12
    {Rect(272,120,131,131)}, //13
    {Rect(120,101,125,127), Rect(989,95,122,125)}, //14
    {Rect(154,56,129,138)}, //15
	};
	//get ground truths
	std::vector<Rect> groundTruths = allGroundTruths[imageNumber];
	return groundTruths;
}

/** @function detectAndDisplay */
std::vector<Rect> detectAndDisplay( Mat frame , std::vector<Rect> dartboards )
{
	Mat frame_gray;
	Mat sobelX;
	Mat sobelY;
  Mat sobelMag;
  Mat sobelGr;
	Mat canny;
	Mat circles;
	Mat hough;
	Mat houghSpaceCircle;
	Mat lines;
	Mat houghSpaceLines;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	GaussianBlur(frame_gray,15, frame_gray);

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	//perform canny edge Detection
	Canny(frame_gray, canny, 80, 40, 3);

	//2.5 Perform sobel transform
	sobel(frame_gray,sobelX,sobelY,sobelMag,sobelGr);

	//threshold image
	thresholdMag(sobelMag,70); //50

	//hough transform circles
	circles.create(frame_gray.size(), frame_gray.type());
	circles = frame_gray;

	houghCircle(sobelMag, sobelGr, circles, houghSpaceCircle);
	//houghLines(sobelMag, sobelGr, lines, houghSpaceLines);

	//write out images
	imwrite("sobelMag.jpg", sobelMag);
	imwrite("sobelGr.jpg", sobelGr);
	imwrite("houghspacecircle.jpg", houghSpaceCircle);
  imwrite("circles.jpg", circles);
	imwrite("canny.jpg", canny);

  // 3. Print number of dartboards found
	std::cout << dartboards.size() << std::endl;

  // 4. Draw box around dartboards found
	for( int i = 0; i < dartboards.size(); i++ )
	{
		rectangle(frame, Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height), Scalar( 0, 255, 0 ), 2);
	}
	return dartboards;
}

void thresholdMag(cv::Mat &input,int threshVal)
{
	for(int i = 0;i < input.rows;i++)
	{
		for(int j = 0;j < input.cols;j++)
		{
			if(input.at<uchar>(i,j) > threshVal)
			{
				input.at<uchar>(i,j) = (uchar) 255;
			}
			else
			{
				input.at<uchar>(i,j) = (uchar) 0;
			}
		}
	}
}

int*** malloc3dArray(int dim1, int dim2, int dim3)
{
	int i,j,k;
	int*** array = (int ***)malloc(dim1*sizeof(int**));
	for (i = 0; i< dim1; i++)
	{
 		array[i] = (int **) malloc(dim2*sizeof(int *));
		for (j = 0; j < dim2; j++)
		{
			array[i][j] = (int *)malloc(dim3*sizeof(int));
    }
  }
	return array;
}

/*void houghLines(cv::Mat&sobelMag, cv::&sobelGrad, cv::&lines, cv::&houghSpaceLines)
{
	space.create(sobelGrad.size(), sobelGrad.height());
	houghSpace[sobelGrad.cols][sobelGrad.rows];
	for(int i = 0; i < sobelGrad.cols; i++)
	{
		for(int j = 0; j < sobelGrad.rows; j++)
		{
			houghSpace[i][j] = 0;
		}
	}
	for(int i = 0; i < sobelGrad.cols; i++)
	{
		for(int j = 0; j < sobelGrad.rows; j++)
		{
			houghSpaceLines.a
		}
	}
} */

void houghCircle(cv::Mat &edges,cv::Mat &thetas,cv::Mat &grey,cv:: Mat &space)
{
	space.create(edges.size(), edges.type());
	int*** houghSpace = malloc3dArray(edges.cols, edges.rows, RADIUS_RANGE);
	for(int i = 0;i < edges.cols;i++)
	{
		for(int j = 0; j < edges.rows;j++)
		{
			for(int k = 0;k < RADIUS_RANGE;k++)
			{
				houghSpace[i][j][k] = 0;
			}
		}
	}

	for(int i = 0;i < edges.rows;i++)
	{
		for(int j = 0;j < edges.cols;j++)
		{
			space.at<uchar>(i,j) = 0;
			int imageVal = edges.at<uchar>(i,j);
			int theta = thetas.at<uchar>(i,j);
			theta = (((theta - 0) * 360 / 255) + 0);
			if(imageVal == 255 && theta > 30 && theta < 330)
			{
				for(int r = MIN_RAD;r < MAX_RAD;r++)
				{
					for(int q = 0;q < 2;q++)
					{
						int x0;
						int y0;
						if(q == 0)
						{
							x0 = (int) j - (r)*cos(theta);
							y0 = (int) i - (r)*sin(theta);
						}
						else
						{
							x0 = (int) j + (r)*cos(theta);
							y0 = (int) i + (r)*sin(theta);
						}
						int rad = r - MIN_RAD;
						if((x0 < 1) || (y0 < 1)){ }
						else if(x0 > (edges.cols - 1)){ }
						else if(rad > 103){ }
						else if(y0 > edges.rows - 1){ }
						else
						{
							if(houghSpace[x0][y0][rad] == 0)
							{
								houghSpace[x0][y0][rad] += 1;
							}
							else
							{
								houghSpace[x0][y0][rad] *= 3;
							}
						}
					}
				}
			}
		}
	}
	for(int i = 1; i < edges.rows - 1;i++){
		for(int j = 1; j < edges.cols - 1;j++){
			int imval = 0;
			imval = space.at<uchar>(i,j);
			for(int k = 0;k < RADIUS_RANGE;k++){
				int votes = 0;
				imval += houghSpace[j][i][k]*1;
				if(imval > 255)imval = 255;

				votes += houghSpace[j][i][k];
				votes += houghSpace[j+1][i][k];
				votes += houghSpace[j-1][i][k];
				votes += houghSpace[j][i+1][k];
				votes += houghSpace[j][i-1][k];
				if (votes > 600)
				{
					printf("votes = %i\n",votes );
				}

				if(votes > 2000)
				{
					int radius = k+ MIN_RAD;
					grey.at<uchar>(i,j) = 40;
					grey.at<uchar>(i,j-1) = 40;
					grey.at<uchar>(i,j+1) = 40;
					grey.at<uchar>(i-1,j) = 40;
					grey.at<uchar>(i+1,j) = 40;
					for(int y = -(radius);y < (radius+1);y++)
					{
						for(int x = -(radius);x < (radius+1);x++)
						{
							if(sqrt((x*x) + (y*y)) <= (radius) && sqrt((x*x) + (y*y)) > (radius-1))
							{
								if (i+y < 0 || i+y > edges.rows || j+x < 0 || j+x > edges.cols){ }
								else
								{
									grey.at<uchar>(i + y,j + x) = 255;
								}
							}
						}
					}
				}
			}
			space.at<uchar>(i,j) =(uchar) imval;
		}
	}
}

void sobel(cv::Mat &input, cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &sobelMag, cv::Mat &sobelGr)
{
	sobelX.create(input.size(), input.type());
	sobelY.create(input.size(), input.type());
	sobelMag.create(input.size(), input.type());
	sobelGr.create(input.size(), input.type());

	int xKernel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
	int yKernel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};


	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 1, 1, 1, 1,cv::BORDER_REPLICATE);

	for (int i = 0; i < input.rows;i++)
	{
		for (int j = 0;j < input.cols;j++)
		{
			int sum[2] = {0,0};
			for (int m = -1;m<2;m++)
			{
				for (int n = -1;n<2;n++)
				{
					int imagey = i + m + 1;
					int imagex = j + n + 1;

					int kerny = 1 - m;
					int kernx = 1 - n;

					int imageVal = (int) paddedInput.at<uchar>(imagey,imagex);
					int kernValx = xKernel[kerny][kernx];
					int kernValy = yKernel[kerny][kernx];

					sum[0] += imageVal * kernValx;
					sum[1] += imageVal * kernValy;
				}
			}

			double dir = 0;
			if(sum[0] < 19 && sum[0] > -19 && sum[1] < 19 && sum[1] > -19){	}
			else
			{
					dir = atan2(sum[1],sum[0])*360/M_PI;
			}
			if(dir < 0){dir = 360 - (dir* -1);}
			if(sum[0] < 0)sum[0] = sum[0]*-1;
			if(sum[1] < 0)sum[1] = sum[1]*-1;

			if(sum[0] > 255)sum[0]= 255;
			if(sum[1] > 255)sum[1] = 255;

			int mag = sqrt((sum[0]*sum[0]) + (sum[1]*sum[1]));
			if(mag > 255){mag = 255;}
			dir = (((dir - 0) * 255 / 360) + 0);

			sobelMag.at<uchar>(i,j) = (uchar) mag;
			sobelGr.at<uchar>(i,j) = (uchar) dir;

			sobelX.at<uchar>(i, j) = (uchar) sum[0];
			sobelY.at<uchar>(i, j) = (uchar) sum[1];

		}
	}
}

float f1( std::vector<Rect> dartboards, std::vector<Rect> groundTruths)
{
	float tpr = 0;
	float tp = 0;
	printf("size of ground truths %f\n", (float)groundTruths.size());
	float fp = (float)dartboards.size();
	int boardCount[10] = {};
	for(int a = 0; a < (int)groundTruths.size(); a++) {
		for( int i = 0; i < (int)dartboards.size(); i++ )
		{
			//if the box intersects check the level of intersection
			if( (groundTruths[a].x + groundTruths[a].width) < dartboards[i].x || (dartboards[i].x + dartboards[i].width) < groundTruths[a].x
					|| (groundTruths[a].y + groundTruths[a].height) < dartboards[i].y || (dartboards[i].y + dartboards[i].height) < groundTruths[a].y)
			{ /*doesn't intersects*/ }
			else
			{
				//intersection area
				float intersectWidth = min(dartboards[i].x + dartboards[i].width, groundTruths[a].x + groundTruths[a].width) - max(dartboards[i].x, groundTruths[a].x);
				float intersectHeight = min(dartboards[i].y + dartboards[i].height, groundTruths[a].y + groundTruths[a].height) - max(dartboards[i].y, groundTruths[a].y);
				float intersectArea = intersectWidth*intersectHeight;
				//union area
				float unionArea = dartboards[i].area() + groundTruths[a].area() - intersectArea;
				//intersection over union
				float jaccard = intersectArea/unionArea;
				//printf("Jaccard %f\n", jaccard);
				float threshold = 0.5;
				if (jaccard > threshold) {
				//make dartboards detected = loop number + 1
					boardCount[a] = 1;
					//tp + 1
					tp++;
					//reduce fpr by 1
					fp--;
				}
		}
	}
}
	int detectedDartboards = 0;
	for (int i = 0; i < 10; i++)
	{
		detectedDartboards += boardCount[i];
	}
	printf("Detected Dartboards%i\n", detectedDartboards);
	float fn = (float)groundTruths.size() - (float)detectedDartboards;
	//tpr
	tpr = detectedDartboards;
	//precision
	float precision = tp / (float)dartboards.size();
	//recall
	float recall = tp / (tp + fn);
	//f1
	float f1Score = 0;
	if (precision == 0 || recall == 0)
	{
		f1Score = 0;
	}
	else
	{
		f1Score = 2*(( precision * recall ) / (precision + recall));
	 }
	 printf("TP = %f\n", tp);
	 printf("FP = %f\n", fp);
	 printf("FN = %f\n", fn);
	//add f1Score to total
	return f1Score;
}

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);

	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar) sum;
		}
	}
}
