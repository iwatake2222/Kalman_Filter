/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <deque>
#include <array>
#include <algorithm>
#include <chrono>

/* for OpenCV */
#include <opencv2/opencv.hpp>

static constexpr int32_t kWidth = 400;
static constexpr int32_t kHeight = 400;
static constexpr int32_t kWidthInterval = 4;

static double GetRandom()
{
	double value = (double)rand() / RAND_MAX * 2;	// [0, 2]
	return value - 1;	// [-1, 1]
}


void CreateRandomWalk(std::vector<double>& v_list, int32_t num_steps)
{
	v_list.push_back(0);
	for (int32_t i = 1; i < num_steps; i++) {
		v_list.push_back(v_list[i -1] + GetRandom());
	}
}

void AddNoise(std::vector<double>& v_list, double deviation)
{
	v_list.push_back(0);
	for (auto&v : v_list) {
		v += GetRandom() * deviation;
	}
}

void DrawList(cv::Mat& mat, const std::vector<double>& v_list, double scale, cv::Scalar color)
{
	std::vector<int32_t> px_y_list;
	for (const auto& v : v_list) {
		px_y_list.push_back(kHeight - (v * scale + (kHeight / 2)));
	}
	for (int32_t x = 1; x < px_y_list.size(); x++) {
		//cv::circle(mat, cv::Point(x, px_y_list[x]), 1, cv::Scalar(255, 0, 0));
		cv::line(mat, cv::Point((x - 1) * kWidthInterval, px_y_list[x - 1]), cv::Point(x * kWidthInterval, px_y_list[x]), color);
	}
}

void KalmanForward(const std::vector<double>& v_list, std::vector<double>& est_list, double startposition, double start_deviation, double deviation_true, double deviation_noise)
{
	std::vector<double> x_prev;
	std::vector<double> P_prev;
	std::vector<double> K;
	std::vector<double> P;
	std::vector<double> x;

	x_prev.push_back(startposition);
	P_prev.push_back(start_deviation);
	K.push_back(P_prev[0] / (P_prev[0] + deviation_noise));
	P.push_back(deviation_noise * P_prev[0] / (P_prev[0] + deviation_noise));
	x.push_back(x_prev[0] + K[0] * (v_list[0] - x_prev[0]));

	for (int32_t t = 1; t < v_list.size(); t++) {
		x_prev.push_back(x[t - 1]);
		P_prev.push_back(P[t - 1] + deviation_true);

		K.push_back(P_prev[t] / (P_prev[t] + deviation_noise));
		x.push_back(x_prev[t] + K[t] * (v_list[t] - x_prev[t]));
		P.push_back(deviation_noise * P_prev[t] / (P_prev[t] + deviation_noise));
	}

	for (int32_t t = 0; t < v_list.size(); t++) {
		est_list.push_back(x[t]);
	}
}

double KalmanStep(double observation, double startposition, double start_deviation, double deviation_true, double deviation_noise)
{
	static double x_prev;
	static double P_prev;
	static double K;
	static double P;
	static double x;

	static bool isFirstCall = true;
	if (isFirstCall) {
		isFirstCall = false;
		x_prev = startposition;
		P_prev = start_deviation;
		K = P_prev / (P_prev + deviation_noise);
		P = deviation_noise * P_prev / (P_prev + deviation_noise);
		x = x_prev + K * (observation - x_prev);
	}

	x_prev = x;
	P_prev = P + deviation_true;
	K = P_prev / (P_prev + deviation_noise);
	x =  x_prev + K * (observation - x_prev);
	P = deviation_noise * P_prev / (P_prev + deviation_noise);

	return x;
}

int32_t main()
{
	std::srand(1234);

	std::vector<double> true_list;
	CreateRandomWalk(true_list, kWidth / kWidthInterval);
	std::vector<double> observed_list = true_list;
	AddNoise(observed_list, 10);

	std::vector<double> est_list;
	KalmanForward(observed_list, est_list, observed_list[0], 1, 1, 10);

	std::vector<double> est_by_step_list;
	for (const auto& v : observed_list) {
		est_by_step_list.push_back(KalmanStep(v, observed_list[0], 1, 1, 10));
	}
	

	/* Draw Result */
	double max_value = 1;
	max_value = (std::max)(max_value, (std::abs)(*std::max_element(true_list.begin(), true_list.end())));
	max_value = (std::max)(max_value, (std::abs)(*std::min_element(true_list.begin(), true_list.end())));
	max_value = (std::max)(max_value, (std::abs)(*std::max_element(observed_list.begin(), observed_list.end())));
	max_value = (std::max)(max_value, (std::abs)(*std::min_element(observed_list.begin(), observed_list.end())));
	double scale = (kHeight / 2) / max_value;

	cv::Mat mat = cv::Mat::zeros(kHeight, kWidth, CV_8UC3);
	DrawList(mat, true_list, scale, cv::Scalar(255, 255, 255));
	DrawList(mat, observed_list, scale, cv::Scalar(0, 0, 255));
	DrawList(mat, est_list, scale, cv::Scalar(0, 255, 0));
	DrawList(mat, est_by_step_list, scale, cv::Scalar(0, 255, 255));
	cv::imshow("mat", mat);
	cv::waitKey(-1);

	return 0;
}
