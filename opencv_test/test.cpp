#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    cout << getBuildInformation() <<endl;
    Mat img = imread("lena.png");
    imshow("image", img);
    waitKey(0);
    return 0;
}
