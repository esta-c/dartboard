// header inclusion
#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/cv.hpp>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h> 

using namespace cv;


void sobel(
	cv::Mat &input, 
	cv::Mat &sobelX,
	cv::Mat &sobelY,
	cv::Mat &sobelMag,
	cv::Mat &sobelGr	);

void prep(
	cv::Mat &input,
	cv::Mat &goddamn);

void thresh(
	cv::Mat &input,
	int threshVal);

void houghAyy(
	cv::Mat &threshed,
	cv::Mat &thetas,
	cv::Mat &grey,
	cv::Mat &space);

void GaussianBlur(
	cv::Mat &input, 
	int size,
	cv::Mat &blurredOutput);

int main( int argc, char** argv )
{
 // LOADING THE IMAGE
 char* imageName = argv[1];

 Mat image;
 image = imread(imageName);

 if( argc != 2 || !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }
//printf("load image");
  // CONVERT COLOUR, BLUR AND SAVE
 Mat gray_image;
 cvtColor( image, gray_image, CV_BGR2GRAY );
 Mat goddamn;
 prep(gray_image,goddamn);
 imwrite("prep.jpg", goddamn);
 GaussianBlur(goddamn,12,goddamn);
 //printf("is it even gray imaging");

 Mat sobelX;
 Mat sobelY;
 Mat sobelMag;
 Mat sobelGr;
 Mat hough;
 hough.create(gray_image.size(),gray_image.type());
 hough = gray_image;

 Mat space;

 sobel(goddamn,sobelX,sobelY,sobelMag,sobelGr);
 thresh(sobelMag,90);
 houghAyy(sobelMag,sobelGr,hough,space);
 printf("quick to the exit\n");

 imwrite("greycoin.jpg", goddamn);
 //printf("all done\n");
 imwrite("sobelx.jpg",sobelX);
 
 imwrite("sobely.jpg",sobelY);
 
 imwrite("sobelMag.jpg",sobelMag);
 
 imwrite("sobelGr.jpg",sobelGr);
 imwrite("space.jpg",space);
 imwrite("hough.jpg",hough);
 
 return 0;
}

void thresh(cv::Mat &input,int threshVal){
	for(int i = 0;i < input.rows;i++){
		for(int j = 0;j < input.cols;j++){
			if(input.at<uchar>(i,j) > threshVal){
				input.at<uchar>(i,j) = (uchar) 255;
			}else{
				input.at<uchar>(i,j) = (uchar) 0;
			}
		}
	}
}

void prep(cv::Mat &input,cv::Mat &goddamn){
	goddamn.create(input.size(), input.type());
	for(int i = 0;i < input.rows;i++){
		for(int j = 0;j < input.cols;j++){
			int imageVal = input.at<uchar>(i,j);
			//printf("%d>>",imageVal);
			imageVal += 100;
			//printf("%d,,,,,",imageVal);
			if(imageVal > 255)imageVal = 255;
			goddamn.at<uchar>(i,j) = (uchar) imageVal;
		}
	}
}

void houghAyy(cv::Mat &threshed,cv::Mat &thetas,cv::Mat &grey,cv:: Mat &space){
	space.create(threshed.size(), threshed.type());
	short houghSpace[threshed.cols+1][threshed.rows+1][12];
	for(int i = 0;i < threshed.rows;i++){
		for(int j = 0; j < threshed.cols;j++){
			for(int k = 0;k < 12;k++){
				houghSpace[j][i][k] = 0;
			}
		}
	}
	//printf("%d rows :: %d cols\n",threshed.rows,threshed.cols);
	for(int i = 0;i < threshed.rows;i++){
		for(int j = 0;j < threshed.cols;j++){
			//printf("shit\n");
			space.at<uchar>(i,j) = 0;
			int imageVal = threshed.at<uchar>(i,j);
			int theta = thetas.at<uchar>(i,j);
			theta = (((theta - 0) * 360 / 255) + 0);
			//printf("%d:imageVal\n",imageVal);
			//printf("%d::x   %d::y\n",j,i);
			if(imageVal == 255){

				//printf("edgeboys\n");
				for(int r = 34;r < 46;r++){
					//for(int theta = 0;theta < 181;theta++){
					//for(float di = 0;di < 1;di += 0.1){
						for(int q = 0;q < 2;q++){
							int x0;
							int y0;
						//	x0 = (int) j - (r)*cos(theta);
						//	y0 = (int) i - (r)*sin(theta);
							if(q == 0){
								x0 = (int) j - /*1/2*/(r)*cos(theta);
								y0 = (int) i - /*1/2*/(r)*sin(theta);
							//}else if(q == 1){
							//	x0 = (int) j - (r+di)*cos(M_PI +thetas.at<uchar>(i,j));
							//	y0 = (int) i - (r+di)*sin(M_PI +thetas.at<uchar>(i,j));
							}else{
								x0 = (int) j + (r)*cos(theta);
								y0 = (int) i + (r)*sin(theta);	
							}
							int rad = r - 34;
							//printf("%d::x0  %d::yo  %d::rad\n",x0,y0,rad);
							if((x0 < 1) || (y0 < 1)){
								//printf("nope\n");
							}else if(x0 > (threshed.cols - 1)){
								//printf("nein\n");
							}else if(y0 > threshed.rows - 1){
								//printf("never\n");
							
							}else{
								//printf("good\n");
								//printf("%u:current vote\n",houghSpace[x0][y0][rad]);
								if(houghSpace[x0][y0][rad] == 0){
									houghSpace[x0][y0][rad] += (short) 1;
								}else{
									houghSpace[x0][y0][rad] *= (short) 4;
								}
								//printf("%u:current vote\n",houghSpace[x0][y0][rad]);
							}
						}
					//}
					//}
				}
			}
		}
	}
	printf("we oughty\n");
	for(int i = 34;i < threshed.rows-34;i++){
		for(int j = 34;j < threshed.cols-34;j++){
			int imval = 0;
			imval = space.at<uchar>(i,j);
			for(int k = 0;k < 12;k++){
				int votes = 0;
				//printf("imval laod\n");
				imval += houghSpace[j][i][k]*10;
				//printf("%d imval upd\n",imval);
				if(imval > 255)imval = 255;
				//printf("x%d y%d grey upd\n",j,i);
				votes += houghSpace[j][i][k];
				votes += houghSpace[j+1][i][k];
				votes += houghSpace[j-1][i][k];
				votes += houghSpace[j][i+1][k];
				votes += houghSpace[j][i-1][k];
				if(votes > 160){//83
					grey.at<uchar>(i,j) = 40;
					grey.at<uchar>(i,j-1) = 40;
					grey.at<uchar>(i,j+1) = 40;
					grey.at<uchar>(i-1,j) = 40;
					grey.at<uchar>(i+1,j) = 40;
					//printf("%d: x   %d: y   %d:  r  %u:  votes\n",j,i,k + 35,houghSpace[j][i][k]);
					for(int y = -(k+34);y < (k+35);y++){
						for(int x = -(k+34);x < (k + 35);x++){
							if(sqrt((x*x) + (y*y)) <= (k+34) && sqrt((x*x) + (y*y)) > (k+ 33)){
								grey.at<uchar>(i + y,j + x) = 255;
							}  
						}
					}
				}
			}
			space.at<uchar>(i,j) =(uchar) imval;
		}
	}
	/*for(int i = 0; i < threshed.cols;i++){
		grey.at<uchar>(340,i) = 255;
	
	}*/
	printf("RUN\n");

}

void sobel(cv::Mat &input, cv::Mat &sobelX, cv::Mat &sobelY, cv::Mat &sobelMag, cv::Mat &sobelGr){
	sobelX.create(input.size(), input.type());
	sobelY.create(input.size(), input.type());
	sobelMag.create(input.size(), input.type());
	sobelGr.create(input.size(), input.type());
	//printf("made mats");

	int xKernel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
	int yKernel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}}; 


	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 1, 1, 1, 1,cv::BORDER_REPLICATE);

	//printf("begin convolve");
	for (int i = 0; i < input.rows;i++){
		for (int j = 0;j < input.cols;j++){
			int sum[2] = {0,0};
			for (int m = -1;m<2;m++){
				for (int n = -1;n<2;n++){
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
			// (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
			//sum[0] = (((sum[0] - -800) * 255 / 1600) + 0);
			//sum[1] = (((sum[1] - -800) * 255 / 1600) + 0);
			double dir = 0;
			if(sum[0] < 19 && sum[0] > -19 && sum[1] < 19 && sum[1] > -19){
			}else{
				/*if(x == 0){
					dir = 90;
				}else if(y == 0){
					dir = 0;
				}else{*/
					dir = atan2(sum[1],sum[0])*360/M_PI;
				//}
				//dir = atan2(y,x) * 180/M_PI;
				
			}
			if(dir < 0){dir = 360 - (dir* -1);}
			printf("%f\n <<<",dir);
			//printf("<%d>",dir);
			if(sum[0] < 0)sum[0] = sum[0]*-1;
			if(sum[1] < 0)sum[1] = sum[1]*-1;
			
			/*if(sum[0] == 0 && sum[1] == 0){
				dir = 0;
			}else if(sum[0] == 0){
				dir = 0;
			}else if(sum[1] == 0){
				dir = 90;
			}else{
				dir = (int) atan(sum[1]/sum[0]) * 180/M_PI;
			}*/
			//printf("<%d>",dir);
			
			if(sum[0] > 255)sum[0]= 255;
			if(sum[1] > 255)sum[1] = 255;

			int mag = sqrt((sum[0]*sum[0]) + (sum[1]*sum[1]));
			if(mag > 255){mag = 255;}
			dir = (((dir - 0) * 255 / 360) + 0);


			sobelMag.at<uchar>(i,j) = (uchar) mag;
			sobelGr.at<uchar>(i,j) = (uchar) dir;

			sobelX.at<uchar>(i, j) = (uchar) sum[0];
			sobelY.at<uchar>(i, j) = (uchar) sum[1];
			

			//dir = fastAtan2(sum[0],sum[1]);
			//if(dir < 0)dir = dir *-1;
			//int deg = (int) (180 + dir/M_PI*180);
			

		}
		}
	//thresh(sobelX,250);
	//thresh(sobelY,200);
	/*for (int i = 0; i < input.rows;i++){
		for (int j = 0;j < input.cols;j++){
			int y = sobelY.at<uchar>(i,j);
			int x = sobelX.at<uchar>(i,j);
		}
	}*/
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

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

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

