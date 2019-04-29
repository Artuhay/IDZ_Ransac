#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const int features = 2000;

int match_slider = 1, max_match = 100;
int iters_slider = 1, max_iters = 2000;
int rans_tresh_slider = 1, max_rans_thresh = 10;
int confidence_slider = 1, max_confidence = 100;
Mat ideal_image, query_image, corrected_image, homography;

void CorrectImage(
    const Mat &im1,
    const Mat &im2,
    Mat &corr_image,
    Mat &h,
    const int iters,
    const int rans_thr,
    const int match_perc,
    const int confidence) {

  Mat im1Gray, im2Gray;
  cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
  cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);

  std::vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;

  Ptr<Feature2D> orb = ORB::create(features);
  orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
  orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

  std::vector<DMatch> matches;
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(cv::BFMatcher::BRUTEFORCE_HAMMING);
  matcher->match(descriptors1, descriptors2, matches, Mat());

  std::sort(matches.begin(), matches.end());

  const int numGoodMatches = matches.size() * match_perc/100.0f;
  matches.erase(matches.begin()+numGoodMatches, matches.end());

  std::vector<Point2f> points1, points2;

  for( size_t i = 0; i < matches.size(); i++ ) {
    points1.push_back(keypoints1[matches[i].queryIdx].pt);
    points2.push_back(keypoints2[matches[i].trainIdx].pt);
  }

  cv::Mat inliers;
  // Find homography
  h = findHomography(points1, points2, RANSAC, rans_thr, inliers, iters, confidence/100.0f);

  std::vector<DMatch> match_inliers;
  for (int i = 0; i < matches.size(); ++i) {
    if(inliers.at<uchar>(i)) {
      match_inliers.push_back(matches[i]);
    }
  }
  Mat imMatches;
  drawMatches(im1, keypoints1, im2, keypoints2, match_inliers, imMatches);
  imshow("EstimatedEdgesWindow", imMatches);

  warpPerspective(im1, corr_image, h, im2.size());

}

void on_trackbar1(int, void*) {

  CorrectImage(query_image, ideal_image, corrected_image, homography, iters_slider, rans_tresh_slider, match_slider, confidence_slider);

  std::cout << "Number of RANSAC iterations: " << iters_slider << std::endl;

  imshow("HomographyWindow", corrected_image);
}

void on_trackbar2(int, void*) {

  CorrectImage(query_image, ideal_image, corrected_image, homography, iters_slider, rans_tresh_slider, match_slider, confidence_slider);

  std::cout << "RANSAC threshold value: " << rans_tresh_slider << std::endl;

  imshow("HomographyWindow", corrected_image);
}

void on_trackbar3(int, void*) {

  CorrectImage(query_image, ideal_image, corrected_image, homography, iters_slider, rans_tresh_slider, match_slider, confidence_slider);

  std::cout << "Ratio of matched points in RANSAC: " << match_slider << std::endl;

  imshow("HomographyWindow", corrected_image);
}

void on_trackbar4(int, void*) {

  CorrectImage(query_image, ideal_image, corrected_image, homography, iters_slider, rans_tresh_slider, match_slider, confidence_slider);

  std::cout << "RANSAC confidence: " << confidence_slider << std::endl;

  imshow("HomographyWindow", corrected_image);
}

int main(int argc, char **argv) {

  if(argc != 3) {
    cout << "Imput <ideal image path> <query image path>" << endl;
    return -1;
  }

  string ideal_filename = argv[1];
  ideal_image = imread(ideal_filename);
  if (!ideal_image.data) {
    cout << "Fault to read ideal image!" << endl;
    return -1;
  }


  string query_filename = argv[2];
  query_image = imread(query_filename);
  if (!query_image.data) {
    cout << "Fault to read query image!" << endl;
    return -1;
  }

  iters_slider = 1;
  rans_tresh_slider = 1;
  match_slider = 1;
  confidence_slider = 1;

  char TrackbarName1[50], TrackbarName2[50], TrackbarName3[50], TrackbarName4[50];
  sprintf(TrackbarName1, "Max_iters x %d", max_iters);
  sprintf(TrackbarName2, "Rans_thre x %d", max_rans_thresh);
  sprintf(TrackbarName3, "Match_part x %d", max_match);
  sprintf(TrackbarName4, "Confidence_part x %d", max_confidence);

  namedWindow("HomographyWindow", cv::WINDOW_AUTOSIZE);
  namedWindow("EstimatedEdgesWindow", cv::WINDOW_AUTOSIZE);

  cv::createTrackbar(TrackbarName1, "HomographyWindow", &iters_slider, max_iters, on_trackbar1);
  cv::createTrackbar(TrackbarName2, "HomographyWindow", &rans_tresh_slider, max_rans_thresh, on_trackbar2);
  cv::createTrackbar(TrackbarName3, "HomographyWindow", &match_slider, max_match, on_trackbar3);
  cv::createTrackbar(TrackbarName4, "HomographyWindow", &confidence_slider, max_confidence, on_trackbar4);

  on_trackbar1(iters_slider, 0);

  on_trackbar2(rans_tresh_slider, 0);

  on_trackbar3(match_slider, 0);

  on_trackbar4(confidence_slider, 0);


  cout << "Estimated homography : \n" << homography << endl;
  cv::waitKey(0);
  return 0;
}