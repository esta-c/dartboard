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
void detectAndDisplay( Mat frame,
											 std::vector<Rect> faces );

/*void f1( std::vector<Rect> faces,
					 std::vector<Rect> groundTruths,
					 int numberOfGroundTruths,
				 	 float f1); */

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;
std::vector<Rect> faces;
int imageNumber;
int groundTruthNumber;
float f1;

/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	//Get the image number and gTNumber
/*imageNumber = argv[2];
	groundTruthNumber = argv[3];*/

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame , faces );

	// 3.5 f1Score

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame , std::vector<Rect> faces )
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
}

void f1( std::vector<Rect> faces, std::Rect groundTruths, int numberOfGroundTruths, float f1) {
	float tpr = 0;
	int detectedDartboards;
	for(int a = 0; a < numberOfGroundTruths; a++) {
		for( int i = 0; i < faces.size(); i++ )
		{
		float currenttpr = 0;
			//if the box intersects check the level of intersection
			if( (groundTruths[a].x + groundTruths[a].width) < faces[i].x || (faces[i].x + faces[i].width) < groundTruths[a].x
					|| (groundTruths[a].y + groundTruths[a].height) < faces[i].y || (faces[i].y + faces[i].height) < groundTruths[a].y)
			{ //doesn't intersect }
			else
			{
				//what amount do they overlap by and is this enough to count
				// some code here for overlap outputting a float
				if (overlap greater than threshold) {
				//make dartboards detected = loop number + 1
				detectedDartboards = a + 1;
				//check if bigger than previous tpr if it is make it that
				if (currentOverlap > currenttpr) {
				currenttpr = currentOverlap;
					}
				}
			}
	}
	//add to total tpr of the imagex
	tpr += currenttpr;
}

	//tpr divided by detectedDartboards = true tpr
    tpr = tpr/detectedDartboards;
	//faces.size - detectedDartboards = fpr
	  int fpr = faces.size - detectedDartboards;
	//false negative = gTNumber - detectedDartboards
		int fnr = numberOfGroundTruths - detectedDartboards;
	//precision
		float precision = tpr / (tpr + fpr);
	//recall
		float recall = tpr / (tpr + fnr);
	//f1
	 	float f1Score = 2*(( precision * recall ) / (precision + recall));
	//add f1Score to total
		f1 += f1Score;
}
