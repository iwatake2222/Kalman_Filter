#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#include <opencv2/opencv.hpp>

#include "simple_matrix.h"
#include "kalman_filter.h"

static constexpr int32_t kWidth = 1600;
static constexpr int32_t kHeight = 800;
static constexpr double real_delta_t = 0.1;
static constexpr double real_sigma_true = 1;
static constexpr double real_sigma_observe = 1;


static double GetRandom(double mean = 0.0, double sigma = 1.0)
{
	//static std::random_device seed_gen;
	static std::mt19937 seed_gen(12345);
	static std::default_random_engine engine(seed_gen());
	std::normal_distribution<> dist(mean, sigma);
	return dist(engine);
}

static void DrawList(cv::Mat& mat, const std::vector<double>& v_list, double max_v, cv::Scalar color)
{
	std::vector<int32_t> px_y_list;
	for (const auto& v : v_list) {
		px_y_list.push_back(kHeight - (v + max_v) * (kHeight / (2 * max_v)));
	}

	for (int32_t x = 1; x < px_y_list.size(); x++) {
		cv::line(mat, cv::Point((x - 1), px_y_list[x - 1]), cv::Point(x, px_y_list[x]), color);
		//cv::circle(mat, cv::Point(x, px_y_list[x]), 1, color);
	}
}


static KalmanFilter CreateKalmanFilter_UniformLinearMotion(double x0)
{
	constexpr int32_t kNumObserve = 1;	/* (x) */
	constexpr int32_t kNumStatus = 2;	/* (x, v) */
	constexpr double delta_t = 0.1;
	constexpr double sigma_true = 1;
	constexpr double sigma_observe = 1;

	/*** X(t) = F * X(t-1) + w(t) ***/
	/* Matrix to calculate X(t) from X(t-1). assume uniform motion: x(t) = x(t-1) + vt, v(t) = v(t-1) */
	const SimpleMatrix F(kNumStatus, kNumStatus, {
		1, delta_t,
		0, 1
		});

	/* Matrix to calculate delta_X from noise(=w). Assume w as accel */
	const SimpleMatrix G(kNumStatus, 1, {
		delta_t * delta_t / 2,
		delta_t
		});

	/* w(t), = noise, follows Q */
	const SimpleMatrix Q = G * G.Transpose() * (sigma_true * sigma_true);

	/*** Z(t) = H * X(t) + v(t) ***/
	/* Matrix to calculate Z(observed value) from X(internal status) */
	const SimpleMatrix H(kNumObserve, kNumStatus, {
		1, 0
		});

	/* v(t), = noise, follows R */
	const SimpleMatrix R(1, 1, { sigma_observe * sigma_observe });

	/* First internal status */
	const SimpleMatrix P0(kNumStatus, kNumStatus, {
		0, 0,
		0, 0
		});

	const SimpleMatrix X0(kNumStatus, 1, {
		x0,
		0
		});

	KalmanFilter kf;
	kf.Initialize(
		F,
		Q,
		H,
		R,
		X0,
		P0
	);

	return kf;
}

static KalmanFilter CreateKalmanFilter_UniformLinearAcceleration(double x0)
{
	constexpr int32_t kNumObserve = 1;	/* (x) */
	constexpr int32_t kNumStatus = 3;	/* (x, v, a) */
	constexpr double delta_t = 0.1;
	constexpr double sigma_true = 1;
	constexpr double sigma_observe = 1;

	/*** X(t) = F * X(t-1) + w(t) ***/
	/* Matrix to calculate X(t) from X(t-1). assume uniform motion: x(t) = x(t-1) + vt + a*t*t/2, v(t) = v(t-1) + a*t, a(t) = a(t-1) */
	const SimpleMatrix F(kNumStatus, kNumStatus, {
		1, delta_t, delta_t * delta_t / 2,
		0,       1,               delta_t,
		0,       0,                     1,
		});

	/* Matrix to calculate delta_X from noise(=w). Assume w as accel */
	const SimpleMatrix G(kNumStatus, 1, {
		delta_t * delta_t / 2,
		delta_t,
		1
		});

	/* w(t), = noise, follows Q */
	const SimpleMatrix Q = G * G.Transpose() * (sigma_true * sigma_true);

	/*** Z(t) = H * X(t) + v(t) ***/
	/* Matrix to calculate Z(observed value) from X(internal status) */
	const SimpleMatrix H(kNumObserve, kNumStatus, {
		1, 0, 0
		});

	/* v(t), = noise, follows R */
	const SimpleMatrix R(1, 1, { sigma_observe * sigma_observe });

	/* First internal status */
	const SimpleMatrix P0(kNumStatus, kNumStatus, {
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
		});

	const SimpleMatrix X0(kNumStatus, 1, {
		x0,
		0,
		0
		});

	KalmanFilter kf;
	kf.Initialize(
		F,
		Q,
		H,
		R,
		X0,
		P0
	);

	return kf;
}

static void PredictFilter(KalmanFilter& kf)
{
	kf.Predict();
}

static void UpdateKalmanFilter(KalmanFilter& kf, double z)
{
	SimpleMatrix Z(1, 1, { z });
	kf.Update(Z);
}


static double GetX_RandomAcc()
{
	static const SimpleMatrix F(2, 2, {
		1, real_delta_t,
		0, 1
		});
	static const SimpleMatrix G(2, 1, {
		real_delta_t* real_delta_t / 2,
		real_delta_t
		});

	static SimpleMatrix X(2, 1, {
		0,
		0
		});

	double w = GetRandom(0.0, real_sigma_true);
	X = F * X + G * w;

	return X(0, 0);
}

static double GetX_Sin()
{
	static double t = 0;
	double value = std::sin(4 * t * 2 * 3.14 / kWidth) * real_sigma_true * 20;
	t++;
	return value;
}

static double GetX()
{
	return GetX_RandomAcc();
	//return GetX_Sin();
	
}

static KalmanFilter CreateKalmanFilter(double x0)
{
	return CreateKalmanFilter_UniformLinearMotion(x0);
	//return CreateKalmanFilter_UniformLinearAcceleration(x0);
}

int main(int argc, char *argv[])
{
	std::vector<double> x_true_list;
	std::vector<double> z_list;
	std::vector<double> x_predict_list;
	std::vector<double> x_est_list;

	double x0 = 0;
	KalmanFilter kf = CreateKalmanFilter(x0);
	x_true_list.push_back(x0);
	z_list.push_back(x0);
	x_predict_list.push_back(x0);
	x_est_list.push_back(x0);

	for (int32_t i = 1; i < kWidth; i++) {
		/* X_true */
		double x_true = GetX();
		x_true_list.push_back(x_true);

		/* Z (observed value) */
		double v = GetRandom(0.0, real_sigma_observe);
		double z = x_true + v;
		z_list.push_back(z);

		/* X_predict */
		PredictFilter(kf);
		x_predict_list.push_back(kf.X(0, 0));

		/* X_estimate */
		UpdateKalmanFilter(kf, z);
		x_est_list.push_back(kf.X(0, 0));
	}

	/*** Display result ***/
	double max_v_pos = 0;
	for (const auto& v : x_true_list) {
		max_v_pos = (std::max)(max_v_pos, (std::abs)(v));
	}
	cv::Mat mat_pos = cv::Mat::zeros(kHeight, kWidth, CV_8UC3);
	DrawList(mat_pos, x_true_list, max_v_pos, cv::Scalar(255, 255, 255));
	DrawList(mat_pos, z_list, max_v_pos, cv::Scalar(0, 0, 255));
	DrawList(mat_pos, x_predict_list, max_v_pos, cv::Scalar(255, 0, 0));
	DrawList(mat_pos, x_est_list, max_v_pos, cv::Scalar(0, 255, 0));
	
	cv::imshow("mat_pos", mat_pos);
	cv::waitKey(-1);

	return 0;
}

