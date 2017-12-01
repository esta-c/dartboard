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
#define MAX_RAD 100 //125
#define RADIUS_RANGE 78

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

struct myCircle { int x;
									int y;
									int radius1;
							 		int radius2;};

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

vector<myCircle> houghCircle(Mat &edges,
								 										Mat &thetas,
								 										Mat &grey,
								 										Mat &space);

void houghLines(Mat&sobelMag,
								Mat&sobelGrad,
								Mat&lines,
								Mat&houghSpaceLines);

vector<Rect> refineDartboards(vector<Rect> dartboards,
																vector<myCircle> circleCentres);

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

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	GaussianBlur(frame_gray, 10, frame_gray);

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	//perform canny edge Detection
	Canny(frame_gray, canny, 80, 40, 3);
	imwrite("canny.jpg", canny);

	GaussianBlur(canny, 4, cannyBlurred);

	//2.5 Perform sobel transform
	sobel(cannyBlurred,sobelX,sobelY,sobelMag,sobelGr);
	imwrite("sobelGr.jpg", sobelGr);

	//threshold image
	thresholdMag(sobelMag,70); //50


	imwrite("sobelMag.jpg", sobelMag);

	//hough transform circles
	circles.create(frame_gray.size(), frame_gray.type());
	circles = frame_gray;

	vector<myCircle> circleCentres = houghCircle(sobelMag, sobelGr, circles, houghSpaceCircle);
	printf("gets out\n");
	houghLines(sobelMag, sobelGr, lines, houghSpaceLines);
	imwrite("houghspacelines.jpg", houghSpaceLines);

	printf("writing houghspace\n");

  imwrite("houghspacecircle.jpg", houghSpaceCircle);
  imwrite("circles.jpg", circles);
  //imwrite("canny.jpg", canny);

  // 3. Print number of dartboards found
  cout << dartboards.size() << endl;

  // 4. Draw box around dartboards found
	//normal dartboards
	for( int i = 0; i < dartboards.size(); i++ )
	{
		rectangle(frame, Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height), Scalar( 0, 255, 0 ), 2);
	}
	//refined dartboards
	/*vector<Rect> acceptedDartboards = refineDartboards(dartboards, circleCentres);
	for( int i = 0; i < acceptedDartboards.size(); i++ )
	{
		rectangle(frame, Point(acceptedDartboards[i].x, acceptedDartboards[i].y), Point(acceptedDartboards[i].x + acceptedDartboards[i].width, acceptedDartboards[i].y + acceptedDartboards[i].height), Scalar( 0, 255, 0 ), 2);
	}*/
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

vector<Rect> refineDartboards(vector<Rect> dartboards, vector<myCircle> circleCentres)
{
	vector<Rect> acceptedDartboards;
	for (int i = 0; i < dartboards.size(); i++)
	{
		for(int j = 0; j < circleCentres.size(); j++)
		{
			Rect centralRegion = Rect((dartboards[i].x + dartboards[i].width/4), (dartboards[i].y + dartboards[i].height/4), (dartboards[i].width/2), (dartboards[i].height/2));
			int x = circleCentres[j].x;
			int y = circleCentres[j].y;
			int radius1 = circleCentres[j].radius1;
			int radius2 = circleCentres[j].radius2;
			if(x > centralRegion.x && x < (centralRegion.x+centralRegion.width) && y > centralRegion.y && y < (centralRegion.y + centralRegion.height))
			{
				//printf("radius1 = %i, radius2 = %i, width = %i\n", radius1, radius2, dartboards[i].width);
				if(dartboards[i].width < radius1*2.4 || dartboards[i].width < radius2*2.4)
				{
					if(dartboards[i].width > radius1 || dartboards[i].width > radius2)
					{
						acceptedDartboards.push_back(dartboards[i]);
					}
				}
			}
		}
	}
	return acceptedDartboards;
}

void houghLines(Mat &sobelMag, Mat &sobelGrad, Mat &lines, Mat &houghSpaceLines)
{
	int max_length = (int)sqrt((sobelMag.cols*sobelMag.cols) + (sobelMag.rows*sobelMag.rows));
	houghSpaceLines.create(max_length, 360, sobelMag.type());
	int houghSpace[max_length][360];
	for(int i = 0; i < max_length; i++)
	{
		for(int j = 0; j < 360; j++)
		{
			houghSpace[i][j] = 0;
		}
	}
	int highestVote = 0;
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
					gradient = -gradient;
				}
				float minGrad = gradient - tolerance;
				if (minGrad < -180)
				{
					minGrad = -minGrad;
				}
				float maxGrad = gradient + tolerance;
				if(maxGrad > 180)
				{
					maxGrad = -maxGrad;
				}
				for(int k = -180; k < 180; k++)
				{
					if(k >= minGrad && k <= maxGrad)
					{
						float angle = k * (M_PI / 180);
						int index = k+180;
						float icos = i*cos(angle);
						float jsin = j*sin(angle);
						int rho = icos + jsin;
						if(rho < 0){
							rho = abs(rho);
							index = index + 180;
						}
						//printf("%d\n",rho);
						houghSpace[rho][index] += 1;
						if(houghSpace[rho][index] > highestVote){
							highestVote = houghSpace[rho][index];
						}

					}
				}
			}
		}
	}
	for(int i = 0; i < max_length; i++)
	{
		for(int j = 0; j < 360; j++)
		{
			//int desscaledAngle = j/5;
			int imval = houghSpace[i][j/*desscaledAngle*/];
			imval = imval * 255/highestVote;
			//printf("%d",imval);
			if (imval > 255)
			{
				imval = 255;
			}
			imval = imval;
			houghSpaceLines.at<uchar>(i,j) = imval;
			}
		}
	}

int getIndexOfLargestElement(int arr[], int size)
{
    int largestIndex = 0;
    for (int index = largestIndex; index < size; index++)
		{
    	//printf("%d \n",arr[index]);
        if (arr[largestIndex] < arr[index])
				{
            largestIndex = index;
        }
    }
    return largestIndex;
}


vector<myCircle> houghCircle(Mat &edges, Mat &thetas, Mat &grey, Mat &space)
{
	float x, y, dx, dy;
	int x1, y1, x2, y2;
	bool regionDoone = false;
	int regionShiftx = 0;
	int regionShifty = 0;
	space.create(edges.size(), edges.type());
	int*** houghSpace = create3dArray(edges.cols, edges.rows, RADIUS_RANGE);
	for(int i = 0;i < edges.cols;i++)
	{
		for(int j = 0; j < edges.rows;j++)
		{
			space.at<uchar>(j,i) = 0;
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
						houghSpace[x1][y1][r] += 1;
					}
					if((x2 > 1) && (y2 > 1) && (x2 < edges.cols - 1) && (y2 < edges.rows - 1))
					{
						houghSpace[x2][y2][r] += 1;
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
			//int imval = 0;
			//imval = space.at<uchar>(j,i);
			for(int k = 0;k < RADIUS_RANGE;k++)
			{
				int votes = 0;
				//imval += houghSpace[i][j][k]*1;
				//if(imval > 255)imval = 255;
				votes += houghSpace[i][j][k];
				if(k > RADIUS_RANGE/2)
				{
					if(i+1 <= edges.cols)
					{
						votes += houghSpace[i+1][j][k]; //1 0
						if(j-1 >= 0)
						{
							votes += houghSpace[i+1][j-1][k];//1 - 1
						}
						else if(j+1 <= edges.rows)
						{
							votes += houghSpace[i+1][j+1][k];// 1 1
						}
					}
					if(j+1 <= edges.rows)
					{
						votes += houghSpace[i][j+1][k];	//0 1
					}
					if(i-1 >= 0)
					{
						votes += houghSpace[i-1][j][k]; // -1 0
						if(j-1 >= 0)
						{
							votes += houghSpace[i-1][j-1][k];//-1 -1
						}
						else if(j+1 <= edges.rows)
						{
							votes += houghSpace[i-1][j+1][k];// -1 1
						}
					}
					if(j-1 >= 0)
					{
						votes += houghSpace[i][j-1][k];	// 0 -1
					}
					if (votes > highestVotes)
					{
						highestVotes = votes;
					}
				}
			}
		}
	}
	printf("%d \n", highestVotes);
	int chunkRat = 5;
	int chunkY = edges.rows/chunkRat;
	int chunkX = edges.cols/chunkRat;
	//printf("chunked %d  %d \n",chunkX,chunkY);
	//printf("rows %d  Cols %d \n",edges.rows,edges.cols);
	vector<myCircle> finalCentres;
	for(int m = 0;m < chunkRat+1;m++)
	{
		for(int n = 0;n < chunkRat+1;n++)
		{
			//printf("current chunks: %d  %d",n, m);
			int bestBestCirc[5] = {0,0,0,0,0};
			for(int j = 0 + chunkY*m; j < chunkY*(m+1);j++)
			{
				if(j < edges.rows-1)
				{
					for(int i = 0 + chunkX*n; i < chunkX*(n+1);i++)
					{
						//printf("i  %d j  %d\n",i,j );
						if(i < edges.cols-1)
						{
							int imvalPrime = 0;
							//printf("here1?\n");
							imvalPrime = space.at<uchar>(j,i);
							//printf("here2?\n");
							for(int k = 0;k < RADIUS_RANGE;k++)
							{
								imvalPrime += houghSpace[i][j][k];
								if(imvalPrime > 255)imvalPrime = 255;
							}
							space.at<uchar>(j, i) = (uchar) imvalPrime;
							int bestCirc[4] = {0,0,0,0};
							int concentric[2] = {0,0};
							int center[9] = {0,0,0,0,0,0,0,0,0};
							for(int k = 0;k < RADIUS_RANGE;k++)
							{
								int votes = 0;
								votes += houghSpace[i][j][k];//0 0
								center[4] += houghSpace[i][j][k];
								if(k > 0/*RADIUS_RANGE/2 + 10*/)
								{
									if(i+1 <= edges.cols)
									{
										votes += houghSpace[i+1][j][k]; //1 0
										center[5] += houghSpace[i+1][j][k];
										if(j-1 >= 0)
										{
											votes += houghSpace[i+1][j-1][k];//1 - 1
											center[2] += houghSpace[i+1][j-1][k];
										}
										else if(j+1 <= edges.rows)
										{
											votes += houghSpace[i+1][j+1][k];// 1 1
											center[8] += houghSpace[i+1][j+1][k];
										}
									}
									if(j+1 <= edges.rows)
									{
										votes += houghSpace[i][j+1][k];	//0 1
										center[7] += houghSpace[i][j+1][k];
									}
									if(i-1 >= 0)
									{
										votes += houghSpace[i-1][j][k]; // -1 0
										center[3] += houghSpace[i-1][j][k];
										if(j-1 >= 0)
										{
											votes += houghSpace[i-1][j-1][k];//-1 -1
											center[0]+= houghSpace[i-1][j-1][k];
										}
										else if(j+1 <= edges.rows)
										{
											votes += houghSpace[i-1][j+1][k];// -1 1
											center[6] += houghSpace[i-1][j+1][k];
										}
									}
									if(j-1 >= 0){
										votes += houghSpace[i][j-1][k];	// 0 -1
										center[1] += houghSpace[i][j-1][k];
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
										//printf("%d  \n", index);
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
								//printf("looking for concentric\n");
								int smallerCirc = (bestCirc[3]-MIN_RAD)*4/5;
								int biggerCirc = (bestCirc[3]-MIN_RAD)*1.1;
								//printf("%d bigger circ\n",biggerCirc);
							 	for(int k = 0;k < smallerCirc;k++)
								{
									//printf("smaller \n");
									int votes = 0;
									votes += houghSpace[i][j][k];
									if(k > RADIUS_RANGE/2)
									{
										if(i+1 <= edges.cols)
										{
											votes += houghSpace[i+1][j][k]; //1 0
											if(j-1 >= 0)
											{
												votes += houghSpace[i+1][j-1][k];//1 - 1
											}
											else if(j+1 <= edges.rows)
											{
												votes += houghSpace[i+1][j+1][k];// 1 1
											}
										}
										if(j+1 <= edges.rows)
										{
											votes += houghSpace[i][j+1][k];	//0 1
										}
										if(i-1 >= 0)
										{
											votes += houghSpace[i-1][j][k]; // -1 0
											if(j-1 >= 0)
											{
												votes += houghSpace[i-1][j-1][k];//-1 -1
											}
											else if(j+1 <= edges.rows)
											{
												votes += houghSpace[i-1][j+1][k];// -1 1
											}
										}
										if(j-1 >= 0)
										{
											votes += houghSpace[i][j-1][k];	// 0 -1
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
									//printf("bigger \n");
									int votes = 0;
									votes += houghSpace[i][j][k];
									if(k > RADIUS_RANGE/2)
									{
										if(i+1 <= edges.cols)
										{
											votes += houghSpace[i+1][j][k]; //1 0
											if(j-1 >= 0)
											{
												votes += houghSpace[i+1][j-1][k];//1 - 1
											}
											else if(j+1 <= edges.rows)
											{
												votes += houghSpace[i+1][j+1][k];// 1 1
											}
										}
										if(j+1 <= edges.rows)
										{
											votes += houghSpace[i][j+1][k];	//0 1
										}
										if(i-1 >= 0)
										{
											votes += houghSpace[i-1][j][k]; // -1 0
											if(j-1 >= 0)
											{
												votes += houghSpace[i-1][j-1][k];//-1 -1
											}
											else if(j+1 <= edges.rows)
											{
												votes += houghSpace[i-1][j+1][k];// -1 1
											}
										}
										if(j-1 >= 0)
										{
											votes += houghSpace[i][j-1][k];	// 0 -1
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
				grey.at<uchar>(bestBestCirc[2],bestBestCirc[1]) = 40;
				grey.at<uchar>(bestBestCirc[2]-1,bestBestCirc[1]) = 40;
				grey.at<uchar>(bestBestCirc[2]+1,bestBestCirc[1]) = 40;
				grey.at<uchar>(bestBestCirc[2],bestBestCirc[1]-1) = 40;
				grey.at<uchar>(bestBestCirc[2],bestBestCirc[1]+1) = 40;
				for(int y = -(bestBestCirc[3]);y < (bestBestCirc[3]+1);y++)
				{
					for(int x = -(bestBestCirc[3]);x < (bestBestCirc[3]+1);x++)
					{
						if(sqrt((x*x) + (y*y)) <= (bestBestCirc[3]) && sqrt((x*x) + (y*y)) > (bestBestCirc[3]-1))
						{
							if (bestBestCirc[1]+x < 0 || bestBestCirc[1]+x > edges.cols || bestBestCirc[2]+y < 0 || bestBestCirc[2]+y > edges.rows){ }
							else
							{
								grey.at<uchar>(bestBestCirc[2] + y, bestBestCirc[1] + x) = 255;
							}
						}
					}
				}
				//if(bestCirc[3] > 0){printf("best circ %d ",bestCirc[3] );printf("concentric %d \n",concentric[1]);}
				for(int y = -(bestBestCirc[4]);y < (bestBestCirc[4]+1);y++)
				{
					for(int x = -(bestBestCirc[4]);x < (bestBestCirc[4]+1);x++)
					{
						if(sqrt((x*x) + (y*y)) <= (bestBestCirc[4]) && sqrt((x*x) + (y*y)) > (bestBestCirc[4]-1))
						{
							if (bestBestCirc[1]+x < 0 || bestBestCirc[1]+x > edges.cols || bestBestCirc[2]+y < 0 || bestBestCirc[2]+y > edges.rows){ }
							else
							{
								grey.at<uchar>(bestBestCirc[2] + y, bestBestCirc[1] + x) = 255;
							}
						}
					}
				}
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
//	delete[] houghSpace;
	return finalCentres;
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
				float threshold = 0.4;
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
	// printf("TP = %f\n", tp);
	// printf("FP = %f\n", fp);
	// printf("FN = %f\n", fn);
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
