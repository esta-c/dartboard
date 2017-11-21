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
#include <numeric>

using namespace std;
using namespace cv;

/** Function Headers */
std::vector<Rect> detectAndDisplay( Mat frame,
											 std::vector<Rect> dartboards );

std::vector<Rect> chooseGroundTruths(int imageNumber);

float f1( std::vector<Rect> dartboards,
				  std::vector<Rect> groundTruths);

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

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	//2.5 Perform circle Detection
  // 3. Print number of dartboards found
	std::cout << dartboards.size() << std::endl;

  // 4. Draw box around dartboards found
	for( int i = 0; i < dartboards.size(); i++ )
	{
		rectangle(frame, Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height), Scalar( 0, 255, 0 ), 2);
	}
	return dartboards;
}

float f1( std::vector<Rect> dartboards, std::vector<Rect> groundTruths) {
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
			{ /*doesn't intersect*/ }
			else
			{
				//what amount do they overlap by and is this enough to count
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
