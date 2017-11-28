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

#define MIN_RAD 25
#define MAX_RAD 125
#define RADIUS_RANGE 76

//circle tester
/*#define MIN_RAD 160
#define MAX_RAD 170
#define RADIUS_RANGE 11 */
#define SIGN(a) ((a)>0 ? 1 : -1)

using namespace std;
using namespace cv;

/** Function Headers */
vector<Rect> detectAndDisplay(Mat frame,
											 				vector<Rect> dartboards );

vector<Rect> chooseGroundTruths(int imageNumber);

float f1( vector<Rect> dartboards,
				  vector<Rect> groundTruths);

void sobel( Mat &input,
						Mat &sobelX,
						Mat &sobelY,
						Mat &sobelMag,
						Mat &sobelGr);

void GaussianBlur(Mat &input,
									int size,
									Mat &blurredOutput);

void thresholdMag(Mat &input,
							 		int threshVal);

void houghCircle(Mat &edges,
								 Mat &thetas,
								 Mat &grey,
								 Mat &space);

void houghLines(Mat&sobelMag,
								Mat&sobelGrad,
								Mat&lines,
								Mat&houghSpaceLines);

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;
int imageNumber;

/** @function main */
int main( int argc, const char** argv )
{
	vector<Rect> dartboards;
  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	//Get the image number and gTNumber
	imageNumber = atoi (argv[2]);

	vector<Rect> groundTruths = chooseGroundTruths(imageNumber);

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
vector<Rect> chooseGroundTruths(int imageNumber)
{
	vector<vector<Rect> > allGroundTruths =
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
	vector<Rect> groundTruths = allGroundTruths[imageNumber];
	return groundTruths;
}

/** @function detectAndDisplay */
vector<Rect> detectAndDisplay( Mat frame , vector<Rect> dartboards )
{
  Mat frame_gray;
  Mat sobelX;
	Mat sobelY;
  Mat sobelMag;
  Mat sobelGr;
	Mat canny;
	Mat cannyBlurred;
	Mat circles;
	Mat hough;
	Mat houghSpaceCircle;
	Mat lines;
	Mat houghSpaceLines;
	Mat labColours[3];

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	GaussianBlur(frame_gray, 10, frame_gray);

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	//perform canny edge Detection
	Canny(frame_gray, canny, 80, 40, 3);
	imwrite("canny.jpg", canny);

	GaussianBlur(canny, 2, cannyBlurred);

	//2.5 Perform sobel transform
	sobel(cannyBlurred,sobelX,sobelY,sobelMag,sobelGr);
	imwrite("sobelGr.jpg", sobelGr);

	//threshold image
	thresholdMag(sobelMag,70); //50


	imwrite("sobelMag.jpg", sobelMag);

	//hough transform circles
	circles.create(frame_gray.size(), frame_gray.type());
	circles = frame_gray;

	houghCircle(sobelMag, sobelGr, circles, houghSpaceCircle);

	//houghLines(sobelMag, sobelGr, lines, houghSpaceLines);
	//imwrite("houghspacelines.jpg", houghSpaceLines);

	printf("writing houghspace\n");

  imwrite("houghspacecircle.jpg", houghSpaceCircle);
  imwrite("circles.jpg", circles);
  printf("finsihed\n");
  //imwrite("canny.jpg", canny);

  // 3. Print number of dartboards found
  cout << dartboards.size() << endl;

  // 4. Draw box around dartboards found
	for( int i = 0; i < dartboards.size(); i++ )
	{
		rectangle(frame, Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height), Scalar( 0, 255, 0 ), 2);
	}
	return dartboards;
}

void thresholdMag(Mat &input,int threshVal)
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

int*** create3dArray(int dim1, int dim2, int dim3)
{
	int i,j;
	int*** array = new int**[dim1];
	for (i = 0; i< dim1; i++)
	{
 		array[i] = new int*[dim2];
		for (j = 0; j < dim2; j++)
		{
			array[i][j] = new int[dim3];
    }
  }
	return array;
}

void houghLines(Mat &sobelMag, Mat &sobelGrad, Mat &lines, Mat &houghSpaceLines)
{
	int max_length = (int)sqrt((sobelMag.cols*sobelMag.cols) + (sobelMag.rows*sobelMag.rows));
	printf("max leng = %i\n", max_length);
	houghSpaceLines.create(max_length, 900, sobelMag.type());
	int houghSpace[max_length][180];
	for(int i = 0; i < max_length; i++)
	{
		for(int j = 0; j < 180; j++)
		{
			houghSpace[i][j] = 0;
		}
	}
	for(int i = 0; i < sobelGrad.cols; i++)
	{
		for(int j = 0; j < sobelGrad.rows; j++)
		{
			int imageVal = sobelMag.at<uchar>(j,i);
			float theta = sobelGrad.at<uchar>(j,i);
			theta = (theta / 255) * 180;
			if (imageVal == 255)
			{
				float tolerance = 10;
				float gradient = theta + 90;
				if (gradient > 180)
				{
					gradient = gradient - 180;
				}
				float minGrad = gradient - tolerance;
				if (minGrad < 0)
				{
					minGrad = 180 + minGrad;
				}
				float maxGrad = gradient + tolerance;
				if(maxGrad > 180)
				{
					maxGrad = maxGrad - 180;
				}
				for(int k = 0; k < 180; k++)
				{
					if(k >= minGrad && k <= maxGrad)
					{
						float angle = k * (M_PI / 180);
						float rho = i*cos(angle) + j*sin(angle);
						houghSpace[(int)rho][k] += 1;

					}
				}
			}
		}
	}
	for(int i = 0; i < max_length; i++)
	{
		for(int j = 0; j < 900; j++)
		{
			int desscaledAngle = j/5;
			int imval = houghSpace[i][desscaledAngle];
			if (imval > 255)
			{
				imval = 255;
			}
			houghSpaceLines.at<uchar>(i,j) = imval;
			}
		}
	}



void houghCircle(Mat &edges, Mat &thetas, Mat &grey, Mat &space)
{
	float x, y, dx, dy;
	int x1, y1, x2, y2;

	space.create(edges.size(), edges.type());
	int*** houghSpace = create3dArray(edges.cols, edges.rows, RADIUS_RANGE);
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

	for(int i = 0;i < edges.cols;i++)
	{
		for(int j = 0;j < edges.rows;j++)
		{
			space.at<uchar>(j,i) = 0;
			for(int k = 0;k < RADIUS_RANGE;k++)
			{
				int r = k+MIN_RAD;
				int imageVal = edges.at<uchar>(j,i);
				float theta = thetas.at<uchar>(j,i);
				theta = (theta / 255) * 180;
				theta = theta * (M_PI / 180);
				if(imageVal == 255)
				{
					x = (r)*cos(theta);
					y = (r)*sin(theta);
					x1 = (int) (i + x); y1 = (int) (j + y);
					x2 = (int) (i - x); y2 = (int) (j - y);
					if((x1 > 1) && (y1 > 1) && (x1 < edges.cols - 1) && (y1 < edges.rows - 1))
					{
						if(houghSpace[x1][y1][r] == 0)
						{
							houghSpace[x1][y1][r] += 1;
						}
						else
						{
							houghSpace[x1][y1][r] *= 2;
						}
					}
					if((x2 > 1) && (y2 > 1) && (x2 < edges.cols - 1) && (y2 < edges.rows - 1))
					{
						if(houghSpace[x2][y2][r] == 0)
						{
							houghSpace[x2][y2][r] += 1;
						}
						else
						{
							houghSpace[x2][y2][r] *= 2;
						}
					}
				}
			}
		}
	}
	int highestVotes = 0;
	for(int i = 1; i < edges.cols - 1;i++)
	{
		for(int j = 1; j < edges.rows - 1;j++)
		{
			int imval = 0;
			imval = space.at<uchar>(j,i);
			for(int k = 0;k < RADIUS_RANGE;k++)
			{
				int votes = 0;
				imval += houghSpace[i][j][k]*1;
				if(imval > 255)imval = 255;
				votes += houghSpace[i][j][k];
				votes += houghSpace[i+1][j][k];
				votes += houghSpace[i-1][j][k];
				votes += houghSpace[i][j+1][k];
				votes += houghSpace[i][j-1][k];
				if (votes > highestVotes)
				{
					highestVotes = votes;
				}
			}
		}
	}
	for(int i = 1; i < edges.cols - 1;i++)
	{
		for(int j = 1; j < edges.rows - 1;j++)
		{
			int imvalPrime = 0;
			imvalPrime = space.at<uchar>(j,i);
			int bestCirc[4] = {};
			int concentric[2] = {};
			for(int k = 0;k < RADIUS_RANGE;k++)
			{
				int votes = 0;
				imvalPrime += houghSpace[i][j][k]*1;
				if(imvalPrime > 255)imvalPrime = 255;
				votes += houghSpace[i][j][k];
				votes += houghSpace[i+1][j][k];
				votes += houghSpace[i-1][j][k];
				votes += houghSpace[i][j+1][k];
				votes += houghSpace[i][j-1][k];
				if (votes != 0)
				{
					votes = (votes * 1000) / (highestVotes);
				}
				if(votes > 700)
				{
					if (votes > bestCirc[0])
					{
						bestCirc[0] = votes;
						bestCirc[1] = i;
						bestCirc[2] = j;
						bestCirc[3] = k+MIN_RAD;
					}
				}
			}
			// Bigger cicle = bestcirc radius*1.5 -> max radius
			//Smaller circle = bestcirc radius*2/3 -> min radius
		/*int imval = 0;
		imval = space.at<uchar>(j,i);
		 for(int k = 0;k < (bestCirc[3]-MIN_RAD)*(2/3);k++)
			{
				int votes = 0;
				imval += houghSpace[i][j][k]*1;
				if(imval > 255)imval = 255;
				votes += houghSpace[i][j][k];
				votes += houghSpace[i+1][j][k];
				votes += houghSpace[i-1][j][k];
				votes += houghSpace[i][j+1][k];
				votes += houghSpace[i][j-1][k];
				if (votes != 0)
				{
					votes = (votes * 1000) / (highestVotes);
				}
				if(votes > 700)
				{
					if (votes > concentric[0])
					{
						concentric[0] = votes;
						concentric[1] = k+MIN_RAD;
					}
				}
			}
			imval = 0;
			imval = space.at<uchar>(j,i);
			for(int k = (bestCirc[3]-MIN_RAD)*1.5;k < RADIUS_RANGE;k++)
			{
				int votes = 0;
				imval += houghSpace[i][j][k]*1;
				if(imval > 255)imval = 255;
				votes += houghSpace[i][j][k];
				votes += houghSpace[i+1][j][k];
				votes += houghSpace[i-1][j][k];
				votes += houghSpace[i][j+1][k];
				votes += houghSpace[i][j-1][k];
				if (votes != 0)
				{
					votes = (votes * 1000) / (highestVotes);
				}
				if(votes > 700)
				{
					if (votes > concentric[0])
					{
						concentric[0] = votes;
						concentric[1] = k+MIN_RAD;
					}
				}
			}
			int bestCircX = bestCirc[1];
			int bestCircY = bestCirc[2];
			int bestCircRad = bestCirc[3];
			grey.at<uchar>(bestCircY,bestCircX) = 40;
			grey.at<uchar>(bestCircY,bestCircX-1) = 40;
			grey.at<uchar>(bestCircY,bestCircX+1) = 40;
			grey.at<uchar>(bestCircY-1,bestCircX) = 40;
			grey.at<uchar>(bestCircY+1,bestCircX) = 40;
			for(int y = -(bestCircRad);y < (bestCircRad+1);y++)
			{
				for(int x = -(bestCircRad);x < (bestCircRad+1);x++)
				{
					if(sqrt((x*x) + (y*y)) <= (bestCircRad) && sqrt((x*x) + (y*y)) > (bestCircRad-1))
					{
						if (bestCircX+x < 0 || bestCircX+x > edges.cols || bestCircY+y < 0 || bestCircY+y > edges.rows){ }
						else
						{
							grey.at<uchar>(bestCircY + y, bestCircX + x) = 255;
						}
					}
				}
			}
			printf("concentric rad %d \n",concentric[1]);
			for(int y = -(concentric[1]);y < (concentric[1]+1);y++)
			{
				for(int x = -(concentric[1]);x < (concentric[1]+1);x++)
				{
					if(sqrt((x*x) + (y*y)) <= (concentric[1]) && sqrt((x*x) + (y*y)) > (concentric[1]-1))
					{
						if (bestCircX+x < 0 || bestCircX+x > edges.cols || bestCircY+y < 0 || bestCircY+y > edges.rows){ }
						else
						{
							grey.at<uchar>(bestCircY + y, bestCircX + x) = 255;
						}
					}
				}
			}*/
			space.at<uchar>(j, i) = (uchar) imvalPrime;
		}
	}
	delete[] houghSpace;
}

void sobel(Mat &input, Mat &sobelX, Mat &sobelY, Mat &sobelMag, Mat &sobelGr)
{
	sobelX.create(input.size(), input.type());
	sobelY.create(input.size(), input.type());
	sobelMag.create(input.size(), input.type());
	sobelGr.create(input.size(), input.type());

	int xKernel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
	int yKernel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};


	Mat paddedInput;
	copyMakeBorder( input, paddedInput, 1, 1, 1, 1,BORDER_REPLICATE);

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
					dir = atan2(sum[1],sum[0])*180/M_PI;
			}
			if(dir < 0){dir = 180 - (dir* -1);}
			if (dir != 0)
			{
				//printf("direction = %f, x = %i, y = %i\n",dir, j, i);
			}
			if(sum[0] < 0)sum[0] = sum[0]*-1;
			if(sum[1] < 0)sum[1] = sum[1]*-1;

			if(sum[0] > 255)sum[0]= 255;
			if(sum[1] > 255)sum[1] = 255;

			int mag = sqrt((sum[0]*sum[0]) + (sum[1]*sum[1]));
			if(mag > 255){mag = 255;}
			dir = (dir / 180) * 255;

			sobelMag.at<uchar>(i,j) = (uchar) mag;
			sobelGr.at<uchar>(i,j) = (uchar) dir;

			sobelX.at<uchar>(i, j) = (uchar) sum[0];
			sobelY.at<uchar>(i, j) = (uchar) sum[1];

		}
	}
}

float f1( vector<Rect> dartboards, vector<Rect> groundTruths)
{
	float tpr = 0;
	float tp = 0;
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

void GaussianBlur(Mat &input, int size, Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1D
	Mat kX = getGaussianKernel(size, -1);
	Mat kY = getGaussianKernel(size, -1);

	// make it 2D multiply one by the transpose of the other
	Mat kernel = kX * kY.t();

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	Mat paddedInput;
	copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		BORDER_REPLICATE );

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
