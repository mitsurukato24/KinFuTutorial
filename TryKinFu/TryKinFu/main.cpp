#include <cstdio>
#include <iostream>
#include <ctime>
#include <cmath>
#include <time.h>
#include <windows.h>
#include <fstream>
#include <chrono>
#include <random>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/rgbd/kinfu.hpp>
#include <opencv2/rgbd/depth.hpp>
#include <opencv2/rgbd/dynafu.hpp>

using namespace std;
using namespace cv;


bool readDepthForTsukuba(std::string filename, Mat &depth)
{
	// read depth image for tsukuba dataset
	FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		std::cout << "File can not be opened. \n";
		return false;
	}
	fs["depth"] >> depth;
	fs.release();
	return true;
}

bool readDepthForTUM(std::string filename, Mat &depth)
{
	depth = imread(filename, IMREAD_UNCHANGED);
	if (depth.empty()) return false;
	depth.convertTo(depth, CV_32FC1, 1.0 / 5000.0);
	// depth.setTo(numeric_limits<double>::quiet_NaN(), depth == 0);
	return true;
}

std::string getZeropadStr(int num, int len)
{
	// 0 padding for tsukuba dataset
	std::ostringstream oss;
	oss << std::setw(len) << std::setfill('0') << num;
	return oss.str();
}

void readTsukubaDataset(
	const string rgbDirname, const string depthDirname, const string gtFilename,
	vector<string> &rgbFilenames, vector<string> &depthFilenames,
	Mat &cameraMat, vector<Eigen::VectorXf> &gtPoses, int maxNumImg = 1000
)
{
	const int width = 640, height = 480;
	cameraMat.at<float>(0, 0) = 615.f;
	cameraMat.at<float>(1, 1) = 615.f;
	cameraMat.at<float>(0, 2) = width / 2.f;
	cameraMat.at<float>(1, 2) = height / 2.f;

	// get the situation name like flashlight
	int s = rgbDirname.rfind("illumination") + 13;
	int e = rgbDirname.find('/', s + 1);
	string situation = rgbDirname.substr(s, e - s);

	// X Y Z A B C : x, y, z, theta_x, theta_y, theta_z
	ifstream ifs(gtFilename);
	string line;
	int count = 0;
	while (std::getline(ifs, line))
	{
		stringstream ss(line);
		Eigen::VectorXf tmp = Eigen::VectorXf::Zero(6);
		for (int i = 0; i < 6; i++)
		{
			string tmps;
			ss >> tmps;
			tmp(i) = std::stof(tmps);
		}
		// I use flourescent
		rgbFilenames.push_back(rgbDirname + "tsukuba_" + situation + "_L_" + getZeropadStr(count + 1, 5) + ".png");
		depthFilenames.push_back(depthDirname + "tsukuba_depth_L_" + getZeropadStr(count + 1, 5) + ".xml");
		gtPoses.push_back(tmp);
		if (count++ > maxNumImg + 1)  break;
	}
	ifs.close();
}

void readTUMDataset(
	const string dirname, vector<string> &rgbFilenames, vector<string> &depthFilenames,
	Mat &cameraMat, vector<Eigen::VectorXf> &gtPoses, int maxNumImg = 1000
)
{
	// TODO: other dataset
	cameraMat.at<float>(0, 0) = 567.6f;
	cameraMat.at<float>(1, 1) = 570.2f;
	cameraMat.at<float>(0, 2) = 324.7f;
	cameraMat.at<float>(1, 2) = 250.1f;
	ifstream tumifs(dirname + "associations.txt");

	int count = 0;
	string line;
	while (std::getline(tumifs, line))
	{
		std::stringstream ss(line);
		Eigen::VectorXf tmp = Eigen::VectorXf::Zero(7);
		for (int i = 0; i < 12; i++)
		{
			string s;
			ss >> s;
			if (i == 1)	rgbFilenames.push_back(dirname + s);
			else if (i == 3)	depthFilenames.push_back(dirname + s);
			else if (i >= 5)
				tmp(i - 5) = std::stof(s);
		}
		Eigen::Quaternionf q(tmp(6), tmp(3), tmp(4), tmp(5));
		Eigen::VectorXf x = Eigen::VectorXf::Zero(6);
		Sophus::SE3f pose(Eigen::Quaternionf(tmp(6), tmp(3), tmp(4), tmp(5)), Eigen::Vector3f(tmp(0), tmp(1), tmp(2)));
		x = pose.log();
		gtPoses.push_back(x);
		if (count++ > maxNumImg) break;
	}
	tumifs.close();
}

int main()
{
	// --- read dataset
	Mat cameraMat = Mat::eye(3, 3, CV_32F);
	vector<Eigen::VectorXf> gtPoses;
	vector<string> rgbFilenames, depthFilenames;
	const int maxNumImg = 2000;
	bool flagTsukubaNotTUM = !false;
	if (flagTsukubaNotTUM)
	{
		std::string rgbDirname = "C:/Users/mkkat/Documents/Dataset/NewTsukubaStereoDataset/illumination/fluorescent/left/";
		// scale : cm
		std::string depthDirname = "C:/Users/mkkat/Documents/Dataset/NewTsukubaStereoDataset/groundtruth/depth_maps/left/";
		std::string gtFilename = "C:/Users/mkkat/Documents/Dataset/NewTsukubaStereoDataset/groundtruth/camera_track.txt";
		readTsukubaDataset(rgbDirname, depthDirname, gtFilename, rgbFilenames, depthFilenames, cameraMat, gtPoses, maxNumImg);
	}
	else
	{
		string tumDirName = "C:/Users/mkkat/Documents/Dataset/TUMRGB-D/rgbd_dataset_freiburg3_long_office_household/";
		readTUMDataset(tumDirName, rgbFilenames, depthFilenames, cameraMat, gtPoses, maxNumImg);
	}

	Mat prevDepth, curDepth;
	if (flagTsukubaNotTUM)
	{
		readDepthForTsukuba(depthFilenames[0], prevDepth);
	}
	else
	{
		readDepthForTUM(depthFilenames[0], prevDepth);
	}
	int width = prevDepth.cols, height = prevDepth.rows;
	Ptr<kinfu::Params> params = kinfu::Params::coarseParams();
	params->frameSize = prevDepth.size();
	params->intr = cameraMat;
	params->depthFactor = 1000.f;
	Ptr<kinfu::KinFu> kinfu = kinfu::KinFu::create(params);
	UMat pd;
	prevDepth.copyTo(pd);
	kinfu->update(pd);
	viz::Viz3d window;
	for (int idx = 1; idx < maxNumImg; idx++)
	{
		cout << "frame : " << idx << endl;
		if (flagTsukubaNotTUM)
		{
			readDepthForTsukuba(depthFilenames[idx], curDepth);
		}
		else
		{
			readDepthForTUM(depthFilenames[idx], curDepth);
		}

		UMat cd;
		curDepth.copyTo(cd);
		if (!kinfu->update(cd))
		{
			kinfu->reset();
			continue;
		}

		Mat render;
		kinfu->render(render);
		if (!render.empty())
		{
			imshow("prev", render);
			waitKey(100);
		}
		UMat pts3d, normals;
		//kinfu->getCloud(pts3d, normals);
		kinfu->getPoints(pts3d);
		if (!pts3d.empty())
		{
			viz::WCloud cloud(pts3d);
			window.showWidget("depth", cloud);
			window.spinOnce(1);
		}
	}
}