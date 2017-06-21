#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main(){

    /*
    Mat kernel_horizontal = (Mat_<int>(3, 3) <<
        0, -1, 0,
        0, 1, 0,
        0, 0, 0);

    Mat kernel_vertical = (Mat_<int>(3, 3) <<
        0, 0, 0,
        0, -1, 1,
        0, 0, 0);

    */

    Mat output_image_hor,output_image_ver,grayscale_mat,binary_mat,final_image;
    Mat picture_input = imread("/home/exp4/catkin_ws/src/hit_and_miss/src/siyah1.jpg",CV_LOAD_IMAGE_COLOR);
    cout << "Picture width - height : " << picture_input.size().width  << " - " << picture_input.size().height << endl;
    if(picture_input.size().width > 1080 && picture_input.size().height > 1920)
        resize(picture_input,picture_input,Size(),0.3,0.2,INTER_LINEAR);


    //Gray uzayi donusum ve thresholding
    cv::cvtColor(picture_input, grayscale_mat, cv::COLOR_BGR2GRAY);
    //blur(grayscale_mat,grayscale_mat,Size(),Point(-1,-1),BORDER_DEFAULT);

    //GaussianBlur(grayscale_mat,grayscale_mat,cv::Size(7,7),0);

    //int size = 17;
    //medianBlur(grayscale_mat,grayscale_mat,19);

    int size = 17;
    medianBlur(grayscale_mat,grayscale_mat,21);

    cv::dilate(grayscale_mat,grayscale_mat,cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size)));
    cv::erode(grayscale_mat,grayscale_mat, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size)));
    cv::erode(grayscale_mat,grayscale_mat, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size)));
    cv::dilate(grayscale_mat,grayscale_mat,cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size)));

    adaptiveThreshold(grayscale_mat,binary_mat,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY_INV,13,5);
    //cv::inRange(grayscale_mat, cv::Scalar(0, 0, 0), cv::Scalar(127,255,0), binary_mat);
    //threshold(grayscaleMat, binaryMat, 127, 255, cv::THRESH_BINARY);

    /*
    vector<Vec4i> lines;
    HoughLinesP( binary_mat, lines, 1, CV_PI/180, 200, 50, 50);
    for( size_t i = 0; i < lines.size(); i++ )
    {
        line( picture_input, Point(lines[i][0], lines[i][1]),
        Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
    }

    std::cout << lines.size() << std::endl;

    */

    /*morphologyEx(binary_mat, output_image_hor, MORPH_HITMISS, kernel_horizontal);
    morphologyEx(binary_mat, output_image_ver, MORPH_HITMISS, kernel_vertical);

    double alpha = 0.5;
    double beta = ( 1.0 - alpha );
    addWeighted( output_image_hor, alpha, output_image_ver, beta, 0.0, final_image);

    bitwise_or(output_image_hor,output_image_ver,final_image);

    kernel_horizontal = (kernel_horizontal + 1) * 127;
    kernel_vertical   = (kernel_vertical + 1) * 127;
    kernel_horizontal.convertTo(kernel_horizontal, CV_8U);
    kernel_vertical.convertTo(kernel_vertical, CV_8U);

    imshow("kernel_horizontal", output_image_hor);
    imshow("kernel_vertical", output_image_ver);*/



    //Contour bulma ve koordinattan benzerlik hesabi
    vector<vector<cv::Point> > cntrs;
    vector<cv::Vec2i> result;
    vector<cv::Vec4i> hierarchy;
    vector<cv::Point> points;


    findContours(binary_mat,cntrs, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    cv::Mat drawing = cv::Mat::zeros( picture_input.size(), CV_8UC3 );

    for(int i = 0;i<cntrs.size();i++){
        double peri = arcLength(cntrs[i],true);
        approxPolyDP(cntrs[i],result,0.04*peri,true);

        if(result.capacity() >= 8 && (contourArea(result) > 5000)){
            cv::Scalar color = cv::Scalar(0, 255, 0);
            drawContours(drawing, cntrs, i, color, 1, 8, hierarchy, 0, cv::Point());

            for(int j = 0;j< result.size();j++){
                points.push_back(result.at(j));
                //cout << result.at(j) << endl;
            }
        }
    }

    int intensityCountBlack = 0;
    int intensityCountWhite = 0;

    cout << "Point Coordinates and intensities: x - y - intensity" << endl;
    for(int i = 4;i < points.size()-4; i++){
        Scalar intensity = grayscale_mat.at<uchar>(points.at(i).y, points.at(i).x);
        //cout << points.size() << endl;
        cout << points.at(i).x << " - " << points.at(i).y << " - " << intensity.val[0] <<  endl;
        if(intensity.val[0] > 100)
            intensityCountWhite++;
        else if(intensity.val[0] < 63)
            intensityCountBlack++;
    }

    double euclidDist1 = sqrt(pow((points.at(4).x - points.at(4).y),2) + pow((points.at(8).x - points.at(8).y),2));
    double euclidDist2 = sqrt(pow((points.at(9).x - points.at(9).y),2) + pow((points.at(10).x - points.at(10).y),2));
    double euclidDist3 = sqrt(pow((points.at(11).x - points.at(11).y),2) + pow((points.at(12).x - points.at(12).y),2));
    double euclidDist4 = sqrt(pow((points.at(6).x - points.at(6).y),2) + pow((points.at(7).x - points.at(7).y),2));

    double ratio1 = euclidDist1 / euclidDist2;
    double ratio2 = euclidDist3 / euclidDist4;

    double similarity = ratio1 - ratio2;

    if(similarity < 0.3 || similarity > -0.3){
        if(intensityCountBlack > 5)
            cout << "Result : Found black color" << endl;
        else if(intensityCountWhite > 5)
            cout << "Result : Found white color" << endl;
    }



    imshow("contours",drawing);

    //imshow("Original",picture_input);

    imshow("Gray",grayscale_mat);

    //imshow("Binary",binary_mat);

    //imshow("Hit or Miss", final_image);

    waitKey(0);

    return 0;
}
