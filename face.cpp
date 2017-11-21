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

using namespace std;
using namespace cv;

/** Function Headers */
std::vector<Rect> detectAndDisplay( Mat frame,
											 std::vector<Rect> faces );

std::vector<Rect> chooseGroundTruths(int imageNumber);

float f1( std::vector<Rect> faces,
				  std::vector<Rect> groundTruths);

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;
int imageNumber;

/** @function main */
int main( int argc, const char** argv )
{
	std::vector<Rect> faces;
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	//Get the image number and gTNumber
	imageNumber = atoi (argv[2]);

	std::vector<Rect> groundTruths = chooseGroundTruths(imageNumber);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	faces = detectAndDisplay( frame , faces );
	// 3.5 f1Score
	float f1Score = f1( faces, groundTruths );

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
    {Rect(444,15,153,175)},
		{Rect(196,132,195,191)},
    {Rect(101,95,91,93)},
    {Rect(323,148,67,72)},
    {Rect(185,94,213,203)},
    {Rect(433,141,110,111)},
    {Rect(210,115,63,66)},
    {Rect(254,170,150,144)},
    {Rect(842,218,117,119), Rect(67,252,60,89)},
    {Rect(203,48,231,232)},
    {Rect(92,104,95,109), Rect(585,127,56,86), Rect(916,149,37,65)},
    {Rect(174,105,59,56)},
    {Rect(156,77,60,137)},
    {Rect(272,120,131,131)},
    {Rect(120,101,125,127), Rect(989,95,122,125)},
    {Rect(154,56,129,138)},
};
	//get ground truths
	std::vector<Rect> groundTruths = allGroundTruths[imageNumber];
	return groundTruths;
}

/** @function detectAndDisplay */
std::vector<Rect> detectAndDisplay( Mat frame , std::vector<Rect> faces )
{
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}
	return faces;
}

float f1( std::vector<Rect> faces, std::vector<Rect> groundTruths) {
	float tpr = 0;
	float tp = 0;
	float fp = (float)faces.size();
	float detectedDartboards;
	for(int a = 0; a < (int)groundTruths.size(); a++) {
		for( int i = 0; i < (int)faces.size(); i++ )
		{
			//if the box intersects check the level of intersection
			if( (groundTruths[a].x + groundTruths[a].width) < faces[i].x || (faces[i].x + faces[i].width) < groundTruths[a].x
					|| (groundTruths[a].y + groundTruths[a].height) < faces[i].y || (faces[i].y + faces[i].height) < groundTruths[a].y)
			{ /*doesn't intersect*/ }
			else
			{
				//what amount do they overlap by and is this enough to count
				//intersection area
				float intersectWidth = min(faces[i].x + faces[i].width, groundTruths[a].x + groundTruths[a].width) - max(faces[i].x, groundTruths[a].x);
				float intersectHeight = min(faces[i].y + faces[i].height, groundTruths[a].y + groundTruths[a].height) - max(faces[i].y, groundTruths[a].y);
				float intersectArea = intersectWidth*intersectHeight;
				//union area
				float unionArea = faces[i].area() + groundTruths[a].area() - intersectArea;
				//intersection over union
				float jaccard = intersectArea/unionArea;
				//printf("Jaccard %f\n", jaccard);
				float threshold = 0.6;
				if (jaccard > threshold) {
				//make dartboards detected = loop number + 1
					detectedDartboards = a + 1;
				//tp + 1
				tp++;
				//reduce fpr by 1
				fp--;
			}
		}
	}
}
	float fn = (float)groundTruths.size() - detectedDartboards;
	//tpr
	tpr = detectedDartboards;
	//precision
	float precision = tp / (float)faces.size();
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
