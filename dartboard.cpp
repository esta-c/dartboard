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

#define MIN_RAD 23 //23
#define MAX_RAD 125 //125
#define RADIUS_RANGE 103

//circle tester
/*#define MIN_RAD 160
#define MAX_RAD 170
#define RADIUS_RANGE 11*/
#define SIGN(a) ((a)>0 ? 1 : -1)

using namespace std;
using namespace cv;

struct myCircle { int x;
									int y;
									int radius1;
							 		int radius2;};

/** Function Headers */
vector<Rect> detectAndDisplay(Mat frame,
											 				vector<Rect> dartboards );

vector<Rect> chooseGroundTruths(int imageNumber);

void GaussianBlur(Mat &input,
									int size,
									Mat &blurredOutput);

void sobel( Mat &input,
						Mat &sobelX,
						Mat &sobelY,
						Mat &sobelMag,
						Mat &sobelGr,
						Mat &linesGrad);

vector<myCircle> houghCircle(Mat &edges,
														 Mat &thetas,
														 Mat &grey,
														 Mat &space);

vector<Point> houghLines(Mat&sobelMag,
								Mat&slinesGrad,
								Mat&lines,
								Mat&houghSpaceLines,
								Mat&blines,
								Mat&temp);

vector<Rect> refineDartboards(vector<Rect> dartboards,
														 	vector<myCircle> circleCentres,
														 	vector<Point> intersetcs);

float f1( vector<Rect> dartboards,
					vector<Rect> groundTruths);

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

	//Get the image number from the name
	string imageFileName = argv[1];
	int startDigit = imageFileName.find_first_of("0123456789");
	int endDigit = imageFileName.find_first_of(".");
	string imageNumberString = imageFileName.substr(startDigit, (endDigit-startDigit));
	int imageNumber = atoi(imageNumberString.c_str());

	vector<Rect> groundTruths = chooseGroundTruths(imageNumber);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect dartboards and Display Result
	dartboards = detectAndDisplay( frame , dartboards );

	//get f1 Score
	float f1Score = f1( dartboards, groundTruths );

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
    {Rect(77,89,121,139), Rect(585,127,56,86), Rect(916,149,37,65)}, //10
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
	Mat frame_grayReal;
	Mat frame_lab;
	Mat frame_temp;
	Mat changed_lab;
	Mat frame_gray_blurred;
	Mat eq_frame_gray;
  Mat sobelX;
	Mat sobelY;
	Mat abssobelX;
	Mat abssobelY;
  Mat sobelMag;
  Mat sobelGr;
	Mat canny_edges;
	Mat circles;
	Mat houghSpaceCircle;
	Mat lines;
	Mat houghSpaceLines;
	Mat linesGrad;
	Mat blines;
	Mat temp;
	Mat frame_grayReal_blurred;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_grayReal, CV_BGR2GRAY);
	cvtColor( frame, frame_lab, CV_BGR2Lab);
	Mat lab[3];
	split(frame_lab, lab);
	lab[2] = 0;
	merge(lab,3,changed_lab);
	cvtColor( changed_lab, frame_temp, CV_Lab2BGR);
	cvtColor( frame_temp, frame_gray, CV_BGR2GRAY );

	//GaussianBlur( frame_gray, frame_gray_blurred, Size(5,5), 0, 0, BORDER_DEFAULT );
	//medianBlur(frame_gray, frame_gray_blurred, 7);

	equalizeHist( frame_gray, eq_frame_gray );
	GaussianBlur( eq_frame_gray, 7, frame_gray_blurred);
	GaussianBlur( frame_grayReal, 7, frame_grayReal_blurred);
	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_grayReal_blurred, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	//2.5 Perform sobel transform
	linesGrad.create(frame.size(), CV_64F);
	sobel(frame_gray_blurred, sobelX, sobelY, sobelMag, sobelGr, linesGrad);
	imwrite("linesgrad.jpg", linesGrad);
	imwrite("sobelX.jpg", sobelX);
	imwrite("sobelY.jpg", sobelY);
	imwrite("sobelGr.jpg", sobelGr);
	//threshold image
	int threshVal1 = 75;
	for(int i = 0;i < sobelMag.cols;i++)
	{
		for(int j = 0;j < sobelMag.rows;j++)
		{
			if(sobelMag.at<double>(j,i) < threshVal1)
			{
				sobelMag.at<double>(j,i) = 0;
			}
			else
			{
				sobelMag.at<double>(j,i) = 255;
			}
		}
	}

	imwrite("sobelMag.jpg", sobelMag);
	//hough transform circles
	cvtColor(frame_grayReal, circles, COLOR_GRAY2BGR);
	vector<myCircle> circleCentres = houghCircle(sobelMag, sobelGr, circles, houghSpaceCircle);

	imwrite("linesgrad.jpg", linesGrad);
	vector<Point> highestvals = houghLines(sobelMag, linesGrad, lines, houghSpaceLines,blines,temp);
	imwrite("tempnotthresh.jpg",temp);

	imwrite("temp.jpg",temp);
	imwrite("lines.jpg", lines);
	imwrite("houghSpaceLines.jpg", houghSpaceLines);
	imwrite("houghSpaceCircle.jpg", houghSpaceCircle);
	imwrite("circles.jpg", circles);
  // 3. Print number of dartboards found
  //cout << dartboards.size() << endl;

  // 4. Draw box around dartboards found
	//normal dartboards
	/*for( int i = 0; i < dartboards.size(); i++ )
	{
		rectangle(frame, Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height), Scalar( 0, 255, 0 ), 2);
	}*/
	//refined
	vector<Rect> acceptedDartboards = refineDartboards(dartboards, circleCentres,highestvals);
	for( int i = 0; i < acceptedDartboards.size(); i++ )
	{
		rectangle(frame, Point(acceptedDartboards[i].x, acceptedDartboards[i].y), Point(acceptedDartboards[i].x + acceptedDartboards[i].width, acceptedDartboards[i].y + acceptedDartboards[i].height), Scalar( 0, 255, 0 ), 2);
	}
	cout << acceptedDartboards.size() << endl;
	return acceptedDartboards;
}

void sobel(Mat &input, Mat &sobelX, Mat &sobelY, Mat &sobelMag, Mat &sobelGr, Mat &linesGrad)
{
	sobelX.create(input.size(), CV_64F);
	sobelY.create(input.size(), CV_64F);
	sobelMag.create(input.size(), CV_64F);
	sobelGr.create(input.size(), CV_64F);

	int xKernel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
	int yKernel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

	Mat paddedInput;
	copyMakeBorder( input, paddedInput, 1, 1, 1, 1, BORDER_REPLICATE);

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
					dir = (atan2(sum[1],sum[0]))*180/M_PI;
			}

			float dirTemp = dir;
			if(dirTemp < 0){dirTemp = 360 + dirTemp;}

			if(dir < 0){dir = 180 - (dir* -1);}
			if(sum[0] < 0)sum[0] = sum[0]*-1;
			if(sum[1] < 0)sum[1] = sum[1]*-1;

			if(sum[0] > 255)sum[0]= 255;
			if(sum[1] > 255)sum[1] = 255;

			int mag = sqrt((sum[0]*sum[0]) + (sum[1]*sum[1]));
			if(mag > 255){mag = 255;}
			dir = (dir / 180) * 255;

			linesGrad.at<double>(i,j) = dirTemp;
			sobelMag.at<double>(i,j) = mag;
			sobelGr.at<double>(i,j) = dir;

			sobelX.at<double>(i, j) = sum[0];
			sobelY.at<double>(i, j) = sum[1];

		}
	}
}

vector<Rect> refineDartboards(vector<Rect> dartboards, vector<myCircle> circleCentres, vector<Point> intersects)
{
	vector<Rect> acceptedDartboards;
	vector<bool> circleCheck;
	vector<bool> interCheck;
	vector<bool> dartCheck;
	for (int i = 0;i < circleCentres.size();i++)
	{
		circleCheck.push_back(false);
	}
	for(int i = 0;i < dartboards.size();i++){
		dartCheck.push_back(false);
	}
	for(int i = 0;i < intersects.size();i++){
		interCheck.push_back(false);
	}
	for (int i = 0; i < dartboards.size(); i++)
	{
		for(int j = 0; j < circleCentres.size(); j++)
		{
			Rect centralRegion = Rect((dartboards[i].x + dartboards[i].width/3), (dartboards[i].y + dartboards[i].height/3), (dartboards[i].width/3), (dartboards[i].height/3));
			int x = circleCentres[j].x;
			int y = circleCentres[j].y;
			int radius1 = circleCentres[j].radius1;
			int radius2 = circleCentres[j].radius2;
			if(!(x > centralRegion.x && x < (centralRegion.x+centralRegion.width) && y > centralRegion.y && y < (centralRegion.y + centralRegion.height)))
			{
				for(int k = 0;k < intersects.size();k++)
				{
					int intX = intersects[k].x;
					int intY = intersects[k].y;
					if (intX > x - 10 && intX < x + 10 && intY > y - 10 && intY < y + 10)
					{
						if(!circleCheck[j] && !interCheck[k])
						{
							printf("circleint\n");
							if(radius1 > radius2)
							{
								if(radius2 > radius1*0.5)
								{
									Rect newRect(x-radius2, y-radius2, radius2*2, radius2*2);
									acceptedDartboards.push_back(newRect);
									printf("circlevi\n");
								}
								else
								{
									Rect newRect(x-radius1, y-radius1, radius1*2, radius1*2);
									acceptedDartboards.push_back(newRect);
									printf("circlevi\n");
								}
							}
							else
							{
								if(radius1 > radius2*0.5)
								{
									Rect newRect(x-radius1, y-radius1, radius1*2, radius1*2);
									acceptedDartboards.push_back(newRect);
									printf("circlevi\n");
								}
								else
								{
									Rect newRect(x-radius2, y-radius2, radius2*2, radius2*2);
									acceptedDartboards.push_back(newRect);
									printf("circlevi\n");
								}
							}
							circleCheck[j] = true;
							interCheck[k] = true;
							}
						}
					}
				}
				else
				{
					if(dartboards[i].width < radius1*2.6 || dartboards[i].width < radius2*2.6)
					{
						if(dartboards[i].width > radius1*0.8 || dartboards[i].width > radius2*0.8)
						{
							if(!circleCheck[j] && !dartCheck[i])
							{
								if((radius1 > radius2))
								{
									if(radius2 > radius1*0.5)
									{
										Rect newRect(x-radius2, y-radius2, radius2*2, radius2*2);
										acceptedDartboards.push_back(newRect);
										printf("circlevi\n");
										dartCheck[i] = true;
										circleCheck[j] = true;
									}
									else
									{
										Rect newRect(x-radius1, y-radius1, radius1*2, radius1*2);
										acceptedDartboards.push_back(newRect);
										printf("circlevi\n");
										dartCheck[i] = true;
										circleCheck[j] = true;
									}
								}
								else
								{
									if(radius1 > radius2*0.5)
									{
										Rect newRect(x-radius1, y-radius1, radius1*2, radius1*2);
										acceptedDartboards.push_back(newRect);
										printf("circlevi\n");
										dartCheck[i] = true;
										circleCheck[j] = true;
									}
									else
									{
										Rect newRect(x-radius2, y-radius2, radius2*2, radius2*2);
										acceptedDartboards.push_back(newRect);
										printf("circlevi\n");
										dartCheck[i] = true;
										circleCheck[j] = true;
									}
								}
							}
						}
					}
				}
			}
		}
		//printf("no seggy");
		for (int i = 0; i < dartboards.size(); i++)
		{
			for(int k = 0; k < intersects.size();k++)
			{
				int intX = intersects[k].x;
				int intY = intersects[k].y;
				Rect centralRegion = Rect((dartboards[i].x + dartboards[i].width/3), (dartboards[i].y + dartboards[i].height/3), (dartboards[i].width/3), (dartboards[i].height/3));
				if(intX > centralRegion.x && intX < (centralRegion.x+centralRegion.width) && intY > centralRegion.y && intY < (centralRegion.y + centralRegion.height))
				{
					if(dartboards[i].width < MAX_RAD*2.6)
					{
						if(dartboards[i].width > MIN_RAD*0.8)
						{
							if(acceptedDartboards.size() > 0)
							{
								for(int m = 0;m < acceptedDartboards.size();m++)
								{
									if(!dartCheck[i] && !interCheck[k] && !((dartboards[i] & acceptedDartboards[m]).area() > 0))
									{
										printf("intvi\n");
										acceptedDartboards.push_back(dartboards[i]);
										dartCheck[i] = true;
										interCheck[k] = true;
									}
								}
							}
							else
							{
								if(!dartCheck[i] && !interCheck[k])
								{
									printf("intvi\n");
									acceptedDartboards.push_back(dartboards[i]);
									dartCheck[i] = true;
									interCheck[k] = true;
								}
							}
						}
					}
				}
			}
		}
		return acceptedDartboards;
}

int getIndexOfLargestElement(int arr[], int size)
{
  int largestIndex = 0;
  for (int index = largestIndex; index < size; index++)
	{
  	if (arr[largestIndex] < arr[index])
		{
        largestIndex = index;
    }
  }
  return largestIndex;
}

vector<Point> houghLines(Mat &sobelMag, Mat &linesGrad, Mat &lines, Mat &houghSpaceLines, Mat &blines, Mat &temp)
{
	int highestImageVal = 0;
	vector<Point> intersects;
	temp.create(sobelMag.rows, sobelMag.cols, CV_64F);
	for(int i = 0; i < sobelMag.cols; i++)
	{
		for(int j = 0; j < sobelMag.rows; j++)
		{
			temp.at<double>(j,i) = 0;
		}
	}
	int max_length = (int)(sqrt((sobelMag.cols*sobelMag.cols) + (sobelMag.rows*sobelMag.rows))/2);
	int centre_x = (int)sobelMag.cols/2;
	int centre_y = (int)sobelMag.rows/2;
	houghSpaceLines.create(max_length, 360, CV_64F);
	float houghSpace[max_length][360];
	for(int i = 0; i < max_length; i++)
	{
		for(int j = 0; j < 360; j++)
		{
			houghSpace[i][j] = 0;
		}
	}
	int highestVote = 0;
	int offset = 10;
	pair<int, int> highestIndex;
	for(int i = 0; i < linesGrad.cols; i++)
	{
		for(int j = 0; j < linesGrad.rows; j++)
		{
			int imageVal = sobelMag.at<double>(j,i);
			float theta = linesGrad.at<double>(j,i);
			if (imageVal == 255)
			{

				if((theta >= offset && theta <= 90-offset) || (theta >= 90+offset && theta <= 180-offset) || (theta >= 180+offset && theta <= 270-offset) || (theta >= 270 + offset && theta <= 360 - offset)){
					float tolerance = 1;
					if (theta > 360)
					{
						theta = theta - 360;
					}
					float minGrad = theta - tolerance;
					float maxGrad = theta + tolerance;
					for(int k = 0; k < 360; k+=5)
					{
						if((k >= minGrad && k <= maxGrad) || (k>=360+minGrad  && minGrad < 0) || (k<=maxGrad-360 && maxGrad > 360))
						{
							float angle = k * (M_PI / 180);
							float icos = (i - centre_x)*cos(angle);
							float jsin = (j - centre_y)*sin(angle);
							int rho = icos + jsin;

							if(rho < 0)
							{
								rho = abs(rho);
							}
							houghSpace[rho][k] += 10;
							if(houghSpace[rho][k] > highestVote)
							{
								highestVote = houghSpace[rho][k];
								highestIndex.first = rho;
								highestIndex.second = k;
							}
						}
					}
				}
			}
		}
	}
	//populate hough space
	for(int i = 0; i < max_length; i++)
	{
		for(int j = 0; j < 360; j++)
		{
			int imval = houghSpace[i][j];
			if (imval > 255)
			{
				imval = 255;
			}
			houghSpaceLines.at<double>(i,j) = imval;
		}
	}

	//count up votes and draw lines
	int highestVotes = 0;
	//get highest votes
	for (int i = 0; i < max_length; i++)
	{
		for (int j = 0; j < 360; j++)
		{
			float votes = 0;
			votes += houghSpaceLines.at<double>(i,j);
			//sum up pixel and its neighbours checking for out of bounds
			if(i+1 < max_length)
			{
				votes += houghSpace[i+1][j]; //1 0
				if(j-1 >= 0)
				{
					votes += houghSpace[i+1][j-1];//1 - 1
				}
				else if(j+1 <= 360)
				{
					votes += houghSpace[i+1][j+1];// 1 1
				}
			}
			if(j+1 <= 360)
			{
				votes += houghSpace[i][j+1];	//0 1
			}
			if(i-1 >= 0)
			{
				votes += houghSpace[i-1][j]; // -1 0
				if(j-1 >= 0)
				{
					votes += houghSpace[i-1][j-1];//-1 -1
				}
				else if(j+1 <= 360)
				{
					votes += houghSpace[i-1][j+1];// -1 1
				}
			}
			if(j-1 >= 0)
			{
				votes += houghSpace[i][j-1];	// 0 -1
			}
			if (votes > highestVotes)
			{
				highestVotes = (int) votes;
			}
		}
	}

		// use votes to draw lines
		for (int i = 0; i < max_length; i++)
		{
			for (int j = 0; j < 360; j++)
			{
				float votes = 0;
				votes += houghSpaceLines.at<double>(i,j);
				if(i+1 < max_length)
				{
					votes += houghSpace[i+1][j]; //1 0
					if(j-1 >= 0)
					{
						votes += houghSpace[i+1][j-1];//1 - 1
					}
					else if(j+1 <= 360)
					{
						votes += houghSpace[i+1][j+1];// 1 1
					}
				}
				if(j+1 <= 360)
				{
					votes += houghSpace[i][j+1];	//0 1
				}
				if(i-1 >= 0)
				{
					votes += houghSpace[i-1][j]; // -1 0
					if(j-1 >= 0)
					{
						votes += houghSpace[i-1][j-1];//-1 -1
					}
					else if(j+1 <= 360)
					{
						votes += houghSpace[i-1][j+1];// -1 1
					}
				}
				if(j-1 >= 0)
				{
					votes += houghSpace[i][j-1];	// 0 -1
				}

				if (votes != 0)
				{
					votes = (votes * 100) / (highestVotes);
				}

				if (votes > 10)
				{
				int x1, y1, x2, y2, x3, y3;
          		x1 = y1 = x2 = y2 = x3 = y3 = 0;
       			float radAlt = (90-(j%90)) * (M_PI / 180);
       			float radMod =  (j%90) * (M_PI / 180);
       			if((j >= 180 && j < 270) || (j >=0 && j <90)){
       				x3 = i* cos(radMod);
       				y3 = i* sin(radMod);
   					if(j >= 180 && j <= 270){
       					x3 = -x3;
       					y3 = -y3;
       					x1 = centre_x + x3 + (max_length)*cos(radAlt);
       					y1 = centre_y + y3 - (max_length)*sin(radAlt);
       					x2 = centre_x + x3 - (max_length)*sin(radMod);
       					y2 = centre_y + y3 + (max_length)*cos(radMod);
       				}else{
       					x1 = centre_x + x3 + (max_length)*sin(radMod);
       					y1 = centre_y + y3 - (max_length)*cos(radMod);
       					x2 = centre_x + x3 - (max_length)*cos(radAlt);
       					y2 = centre_y + y3 + (max_length)*sin(radAlt);
       				}
       			}else{
       				//printf("it is voting for us\n");
       				x3 = i* sin(radMod);
       				y3 = i* cos(radMod);
       				if(j >= 90 && j <= 180){
       					x3 = -x3;
       					x1 = centre_x + x3 + (max_length)*cos(radMod);
       					y1 = centre_y + y3 + (max_length)*sin(radMod);
       					x2 = centre_x + x3 - (max_length)*sin(radAlt);
       					y2 = centre_y + y3 - (max_length)*cos(radAlt);
       				}else{
       					y3 = -y3;
       					x1 = centre_x + x3 + (max_length)*sin(radAlt);
       					y1 = centre_y + y3 + (max_length)*cos(radAlt);
       					x2 = centre_x + x3 - (max_length)*cos(radMod);
       					y2 = centre_y + y3 - (max_length)*sin(radMod);
       				}
       			}

				Point point1(x1,y1);
				Point point2(x2,y2);
				//cvtColor(frame_grayReal, blines, COLOR_GRAY2BGR);
				blines = temp.clone();
				line(blines, point1, point2, Scalar(255,255,255), 2, 8);
				addWeighted(blines,0.01,temp,0.99,0,temp);
				//printf("x1 : %d    y1 : %d  x2 : %d    y2 : %d  \n",x1,y1,x2,y2);
				}
			}
		}
		for (int i = 0; i < temp.cols; i++)
		{
			for (int j = 0; j < temp.rows; j++)
			{
				int imVal = temp.at<double>(j,i);
				if (imVal > highestImageVal)
				{
					highestImageVal = (int)imVal;
				}
			}
		}
		int threshVal2 = highestImageVal*0.85;
		bool skip = false;
		int ydiff,xdiff;
		ydiff = xdiff = 0;
		for(int i = 0;i < temp.cols;i++)
		{
			for(int j = 0;j < temp.rows;j++)
			{
				if(temp.at<double>(j,i) < threshVal2)
				{
					temp.at<double>(j,i) = 0;
				}
				else
				{
					temp.at<double>(j,i) = 255;
					if (!skip){
						Point pointI(i,j);
						intersects.push_back(pointI);
						skip = true;
					}else{
						ydiff++;
						if(ydiff > 20 && xdiff > 20){
							skip = !skip;
							ydiff = 0;
							xdiff = 0;
						}
					}
				}
				if(skip){
					xdiff++;
				}
			}
		}
		return intersects;
}

vector<myCircle> houghCircle(Mat &edges, Mat &thetas, Mat &grey, Mat &space)
{
	float x, y, dx, dy;
	int x1, y1, x2, y2;
	bool regionDoone = false;
	int regionShiftx = 0;
	int regionShifty = 0;
	space.create(edges.size(), CV_64F);
	int sizes[3] = {edges.cols, edges.rows, RADIUS_RANGE};
	Mat houghSpace (3, sizes, CV_64F, double(0));
	for(int i = 0;i < edges.cols;i++)
	{
		for(int j = 0;j < edges.rows;j++)
		{
			for(int k = 0;k < RADIUS_RANGE;k++)
			{
				int r = k+MIN_RAD;
				float imageVal = edges.at<double>(j,i);
				float theta = thetas.at<double>(j,i);
				theta = (theta / 255) * 180;
				theta = theta * (M_PI / 180);
				if(imageVal == 255)
				{
					x = (r)*cos(theta);
					y = (r)*sin(theta);
					x1 = i + x; y1 = j + y;
					x2 = i - x; y2 = j - y;
					if((x1 > 1) && (y1 > 1) && (x1 < edges.cols - 1) && (y1 < edges.rows - 1))
					{
						houghSpace.at<double>(x1,y1,r) += 1;
					}
					if((x2 > 1) && (y2 > 1) && (x2 < edges.cols - 1) && (y2 < edges.rows - 1))
					{
						houghSpace.at<double>(x2,y2,r) += 1;
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
			for(int k = 0;k < RADIUS_RANGE;k++)
			{
				int votes = 0;
				votes += houghSpace.at<double>(i,j,k);
				if(k > RADIUS_RANGE/2)
				{
					if(i+1 <= edges.cols)
					{
						votes += houghSpace.at<double>(i+1,j,k); //1 0
						if(j-1 >= 0)
						{
							votes += houghSpace.at<double>(i+1,j-1,k);//1 - 1
						}
						else if(j+1 <= edges.rows)
						{
							votes += houghSpace.at<double>(i+1,j+1,k);// 1 1
						}
					}
					if(j+1 <= edges.rows)
					{
						votes += houghSpace.at<double>(i,j+1,k);	//0 1
					}
					if(i-1 >= 0)
					{
						votes += houghSpace.at<double>(i-1,j,k); // -1 0
						if(j-1 >= 0)
						{
							votes += houghSpace.at<double>(i-1,j-1,k);//-1 -1
						}
						else if(j+1 <= edges.rows)
						{
							votes += houghSpace.at<double>(i-1,j+1,k);// -1 1
						}
					}
					if(j-1 >= 0)
					{
						votes += houghSpace.at<double>(i,j-1,k);	// 0 -1
					}
					if (votes > highestVotes)
					{
						highestVotes = votes;
					}
				}
			}
		}
	}
	int chunkRat = 5;
	int chunkY = (int)edges.rows/chunkRat;
	int chunkX = (int)edges.cols/chunkRat;
	vector<myCircle> finalCentres;

	//populate hough space
	for (int i = 0; i < edges.cols; i++)
	{
		for (int j = 0; j < edges.rows; j++)
		{
			float imVal = 0;
			for (int k = 0; k < RADIUS_RANGE; k++)
			{
				imVal += houghSpace.at<double>(i,j,k);
				if(imVal> 255)imVal = 255;
			}
			space.at<double>(j, i) = imVal;
		}
	}
	//calculate best cirles
	for(int m = 0;m < chunkRat+1;m++)
	{
		for(int n = 0;n < chunkRat+1;n++)
		{
			int bestBestCirc[5] = {0,0,0,0,0};
			for(int j = 0 + chunkY*m; j < chunkY*(m+1);j++)
			{
				if(j < edges.rows-1)
				{
					for(int i = 0 + chunkX*n; i < chunkX*(n+1);i++)
					{
						if(i < edges.cols-1)
						{
							int bestCirc[4] = {0,0,0,0};
							int concentric[2] = {0,0};
							int center[9] = {0,0,0,0,0,0,0,0,0};
							for(int k = 0;k < RADIUS_RANGE;k++)
							{
								int votes = 0;
								votes += houghSpace.at<double>(i,j,k);//0 0
								center[4] += houghSpace.at<double>(i,j,k);
								if(k > 0)
								{
									if(i+1 <= edges.cols)
									{
										votes += houghSpace.at<double>(i+1,j,k); //1 0
										center[5] += houghSpace.at<double>(i+1,j,k);
										if(j-1 >= 0)
										{
											votes += houghSpace.at<double>(i+1,j-1,k);//1 - 1
											center[2] += houghSpace.at<double>(i+1,j-1,k);
										}
										else if(j+1 <= edges.rows)
										{
											votes += houghSpace.at<double>(i+1,j+1,k);// 1 1
											center[8] += houghSpace.at<double>(i+1,j+1,k);
										}
									}
									if(j+1 <= edges.rows)
									{
										votes += houghSpace.at<double>(i,j+1,k);	//0 1
										center[7] += houghSpace.at<double>(i,j+1,k);
									}
									if(i-1 >= 0)
									{
										votes += houghSpace.at<double>(i-1,j,k); // -1 0
										center[3] += houghSpace.at<double>(i-1,j,k);
										if(j-1 >= 0)
										{
											votes += houghSpace.at<double>(i-1,j-1,k);//-1 -1
											center[0]+= houghSpace.at<double>(i-1,j-1,k);
										}
										else if(j+1 <= edges.rows)
										{
											votes += houghSpace.at<double>(i-1,j+1,k);// -1 1
											center[6] += houghSpace.at<double>(i-1,j+1,k);
										}
									}
									if(j-1 >= 0){
										votes += houghSpace.at<double>(i,j-1,k);	// 0 -1
										center[1] += houghSpace.at<double>(i,j-1,k);
									}
								}
								if (votes != 0)
								{
									votes = (votes * 100) / (highestVotes);
								}
								if(votes > 90)
								{
									if (votes > bestCirc[0])
									{
										int cx,cy;
										int index = getIndexOfLargestElement(center,9);
										switch(index)
										{
											case 0:
												cx = i-1;
												cy = j-1;
												break;
											case 1:
												cx = i;
												cy = j-1;
												break;
											case 2:
												cx = i+1;
												cy = j-1;
												break;
											case 3:
												cx = i-1;
												cy = j;
												break;
											case 4:
												cx = i;
												cy = j;
												break;
											case 5:
												cx = i+1;
												cy = j;
												break;
											case 6:
												cx = i-1;
												cy = j+1;
												break;
											case 7:
												cx = i;
												cy = j+1;
												break;
											case 8:
												cx = i+1;
												cy = j+1;
												break;
										}
										bestCirc[0] = votes;
										bestCirc[1] = cx;
										bestCirc[2] = cy;
										bestCirc[3] = k+MIN_RAD;
									}
								}
							}
							// Bigger cicle = bestcirc radius*1.5 -> max radius
							//Smaller circle = bestcirc radius*2/3 -> min radius
							if(bestCirc[0] > 0)
							{
								regionDoone = true;
								int smallerCirc = (bestCirc[3]-MIN_RAD)*4/5;
								int biggerCirc = (bestCirc[3]-MIN_RAD)*1.1;
							 	for(int k = 0;k < smallerCirc;k++)
								{
									int votes = 0;
									votes += houghSpace.at<double>(i,j,k);
									if(k > RADIUS_RANGE/2)
									{
										if(i+1 <= edges.cols)
										{
											votes += houghSpace.at<double>(i+1,j,k); //1 0
											if(j-1 >= 0)
											{
												votes += houghSpace.at<double>(i+1,j-1,k);//1 - 1
											}
											else if(j+1 <= edges.rows)
											{
												votes += houghSpace.at<double>(i+1,j+1,k);// 1 1
											}
										}
										if(j+1 <= edges.rows)
										{
											votes += houghSpace.at<double>(i,j+1,k);	//0 1
										}
										if(i-1 >= 0)
										{
											votes += houghSpace.at<double>(i-1,j,k); // -1 0
											if(j-1 >= 0)
											{
												votes += houghSpace.at<double>(i-1,j-1,k);//-1 -1
											}
											else if(j+1 <= edges.rows)
											{
												votes += houghSpace.at<double>(i-1,j+1,k);// -1 1
											}
										}
										if(j-1 >= 0)
										{
											votes += houghSpace.at<double>(i,j-1,k);	// 0 -1
										}
									}
									if (votes != 0)
									{
										votes = (votes * 100) / (highestVotes);
									}
									if(votes > 10)
									{
										if (votes > concentric[0])
										{
											concentric[0] = votes;
											concentric[1] = k+MIN_RAD;
										}
									}
								}
								for(int k = biggerCirc;k < RADIUS_RANGE;k++)
								{
									int votes = 0;
									votes += houghSpace.at<double>(i,j,k);
									if(k > RADIUS_RANGE/2)
									{
										if(i+1 <= edges.cols)
										{
											votes += houghSpace.at<double>(i+1,j,k); //1 0
											if(j-1 >= 0)
											{
												votes += houghSpace.at<double>(i+1,j-1,k);//1 - 1
											}
											else if(j+1 <= edges.rows)
											{
												votes += houghSpace.at<double>(i+1,j+1,k);// 1 1
											}
										}
										if(j+1 <= edges.rows)
										{
											votes += houghSpace.at<double>(i,j+1,k);	//0 1
										}
										if(i-1 >= 0)
										{
											votes += houghSpace.at<double>(i-1,j,k); // -1 0
											if(j-1 >= 0)
											{
												votes += houghSpace.at<double>(i-1,j-1,k);//-1 -1
											}
											else if(j+1 <= edges.rows)
											{
												votes += houghSpace.at<double>(i-1,j+1,k);// -1 1
											}
										}
										if(j-1 >= 0)
										{
											votes += houghSpace.at<double>(i,j-1,k);	// 0 -1
										}
									}
									if (votes != 0)
									{
										votes = (votes * 100) / (highestVotes);
									}
									if(votes > 10)
									{
										if (votes > concentric[0])
										{
											concentric[0] = votes;
											concentric[1] = k+MIN_RAD;
										}
									}
								}
								if(bestCirc[0] > bestBestCirc[0])
								{
									bestBestCirc[0] = bestCirc[0];
									bestBestCirc[1] = bestCirc[1];
									bestBestCirc[2] = bestCirc[2];
									bestBestCirc[3] = bestCirc[3];
									bestBestCirc[4] = concentric[1];
								}
							}
						}
					}
				}
			}
			if(bestBestCirc[0] != 0)
			{
				Point center(bestBestCirc[1],bestBestCirc[2]);
				circle(grey, center, bestBestCirc[3], Scalar(0,255,0), 2, 8);
				circle(grey, center, bestBestCirc[4], Scalar(0,255,0), 2, 8);
				//add this best circle to the list to return
				myCircle bestCircleCentreandRadius;
				bestCircleCentreandRadius.x = bestBestCirc[1];
				bestCircleCentreandRadius.y = bestBestCirc[2];
				bestCircleCentreandRadius.radius1 = bestBestCirc[3];
				bestCircleCentreandRadius.radius2 = bestBestCirc[4];
				finalCentres.push_back(bestCircleCentreandRadius);
			}
		}
	}
	return finalCentres;
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
					int imageval = (int)paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = sum;
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
				float threshold = 0.3;
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
	//printf("Detected Dartboards%i\n", detectedDartboards);
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
