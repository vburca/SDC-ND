#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

const double UKF::EPS = 0.001;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 4;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:
  Complete the initialization. See ukf.h for other member properties.
  Hint: one or more values initialized above might be wildly off...
  */

  // Count of total measurements received so far, for lidar and radar
  radar_measurements_ = 0;
  lidar_measurements_ = 0;

  // Count of number of NIS values above the given threshold, for lidar and radar
  radar_NIS_above_thresh_ = 0;
  lidar_NIS_above_thresh_ = 0;

  // NIS thresholds for lidar and radar
  // Values took from: http://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
  // Values also present in the Udacity course notes.
  radar_NIS_thresh_ = 7.815;  // Radar has 3 degrees of freedom
  lidar_NIS_thresh_ = 5.991;  // LIDAR has 2 degrees of freedom

  // NIS Chi2 percentage threshold for the computed NIS values in raport to the thresholds
  // corresponding to the measured sensor
  chi2_threshold_percentage_ = 0.05;

  // Accepted percentage error range +/- Epsilon around the threshold percentage
  chi2_threshold_percentage_epsilon_ = 0.03;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Define measurement noise covariance matrix
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;

  // Define measurement noise covariance matrix
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++)
  {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_)
  {
    // Initialize state vector x
    x_ << 0, 0, 5, .1, .1;

    // Initialize state covariance matrix P
    P_ << .1, 0, 0, 0, 0,
          0, .1, 0, 0, 0,
          0, 0, 4, 0, 0,
          0, 0, 0, 7, 0,
          0, 0, 0, 0, 7;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      float rho = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      x_(0) = rho * cos(phi);
      x_(1) = rho * sin(phi);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
    }

    // Update the timestamp
    time_us_ = meas_package.timestamp_;

    // Done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // Prediction

  // Compute the time elapsed between the current and previous measurements
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0; // in seconds
  time_us_ = meas_package.timestamp_;

  // Make the prediction
  // TODO - Maybe only predict if dt is not too small!
  Prediction(dt);

  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
  {
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
  {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  AugmentedSigmaPoints(&Xsig_aug);
  SigmaPointPrediction(Xsig_aug, delta_t, &Xsig_pred_);
  PredictMeanAndCovariance(&x_, &P_);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_out = VectorXd(2);
  MatrixXd Zsig_out = MatrixXd(2, 2 * n_aug_ + 1);
  MatrixXd S_out = MatrixXd(2, 2);

  PredictLidarMeasurement(&z_out, &Zsig_out, &S_out);
  UpdateState(z, z_out, Zsig_out, S_out, meas_package.sensor_type_, &x_, &P_);
  CheckFilterConsistency(meas_package.sensor_type_, z, z_out, S_out);
}

void UKF::PredictLidarMeasurement(VectorXd* z_out, MatrixXd* Zsig_out, MatrixXd* S_out)
{
  // Set measurement dimension for radar (p_x, p_y)
  int n_z = z_out->size();

  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // Transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);

    // Measurement model
    Zsig(0, i) = p_x;
    Zsig(1, i) = p_y;
  }

  // Calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Calculate measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  S = S + R_lidar_;

  // Write result
  *z_out = z_pred;
  *Zsig_out = Zsig;
  *S_out = S;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_out = VectorXd(3);
  MatrixXd Zsig_out = MatrixXd(3, 2 * n_aug_ + 1);
  MatrixXd S_out = MatrixXd(3, 3);

  PredictRadarMeasurement(&z_out, &Zsig_out, &S_out);
  UpdateState(z, z_out, Zsig_out, S_out, meas_package.sensor_type_, &x_, &P_);
  CheckFilterConsistency(meas_package.sensor_type_, z, z_out, S_out);
}

void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* Zsig_out, MatrixXd* S_out)
{
  // Set measurement dimension for radar (r, phi, r_dot)
  int n_z = z_out->size();

  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // Transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    const double p_x = Xsig_pred_(0, i);
    const double p_y = Xsig_pred_(1, i);
    const double v = Xsig_pred_(2, i);
    const double yaw = Xsig_pred_(3, i);

    const double v1 = cos(yaw) * v;
    const double v2 = sin(yaw) * v;

    // Measurement model
    const double rho = sqrt(p_x * p_x + p_y * p_y);
    double phi = UKF::EPS;
    if (p_x != 0 && p_y != 0)
    {
      phi = atan2(p_y, p_x);
    }
    const double rho_dot = (p_x * v1 + p_y * v2) / std::max(UKF::EPS, rho);

    Zsig(0, i) = rho;
    Zsig(1, i) = phi;
    Zsig(2, i) = rho_dot;
  }

  // Calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Calculate measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // Normalize yaw angle
    NormalizeAngle(&z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  S = S + R_radar_;

  // Write result
  *z_out = z_pred;
  *Zsig_out = Zsig;
  *S_out = S;
}

void UKF::UpdateState(VectorXd& z, VectorXd& z_pred, MatrixXd& Zsig, MatrixXd& S,
    MeasurementPackage::SensorType sensor, VectorXd* x_out, MatrixXd* P_out)
{
  // Create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  // Create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, z_pred.size());

  // Calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(&x_diff(3));

    VectorXd z_diff = Zsig.col(i) - z_pred;

    if (sensor == MeasurementPackage::RADAR)
    {
      NormalizeAngle(&z_diff(1));
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = z - z_pred;
  if (sensor == MeasurementPackage::RADAR)
  {
    NormalizeAngle(&z_diff(1));
  }

  // Update state mean and covariance matrix
  x = x_ + K * z_diff;
  P = P_ - K * S * K.transpose();

  *x_out = x;
  *P_out = P;
}


void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out)
{
  // Create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // Create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // Create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  // Create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // Create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // Create augmented sigma points
  Xsig_aug.col(0) = x_aug;

  float sqroot = sqrt(lambda_ + n_aug_);

  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i + 1) = x_aug + sqroot * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqroot * L.col(i);
  }

  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd& Xsig_aug, double delta_t, MatrixXd* Xsig_out)
{
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    const double p_x      = Xsig_aug(0, i);
    const double p_y      = Xsig_aug(1, i);
    const double v        = Xsig_aug(2, i);
    const double yaw      = Xsig_aug(3, i);
    const double yawd     = Xsig_aug(4, i);
    const double nu_a     = Xsig_aug(5, i);
    const double nu_yawdd = Xsig_aug(6, i);

    double px_p, py_p;

    // Avoid division by zero
    if (fabs(yawd) > 0.001)
    {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else
    {
      px_p = p_x + v * cos(yaw) * delta_t;
      py_p = p_y + v * sin(yaw) * delta_t;
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // Add noise
    px_p = px_p + 0.5 * delta_t * delta_t * cos(yaw) * nu_a;
    py_p = py_p + 0.5 * delta_t * delta_t * sin(yaw) * nu_a;
    v_p = v_p + delta_t * nu_a;

    yaw_p = yaw_p + 0.5 * delta_t * delta_t * nu_yawdd;
    yawd_p = yawd_p + delta_t * nu_yawdd;

    // Write predicted sigma points into the right column
    Xsig_pred(0, i) = px_p;
    Xsig_pred(1, i) = py_p;
    Xsig_pred(2, i) = v_p;
    Xsig_pred(3, i) = yaw_p;
    Xsig_pred(4, i) = yawd_p;
  }

  *Xsig_out = Xsig_pred;
}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out)
{
  // Create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  // Create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);

  // Predict state mean
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    x = x + weights_(i) * Xsig_pred_.col(i);
  }

  // Predict state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x;

    // Normalize yaw angle
    NormalizeAngle(&x_diff(3));

    P = P + weights_(i) * x_diff * x_diff.transpose();
  }

  *x_out = x;
  *P_out = P;
}

void UKF::CheckFilterConsistency(MeasurementPackage::SensorType sensor, VectorXd& z,
  VectorXd& z_pred, MatrixXd& S)
{
  // Calculate the actual NIS value
  VectorXd z_diff = z - z_pred;
  float NIS_value = z_diff.transpose() * S.inverse() * z_diff;

  float percentage_values_above_thresh;

  // Calculate percentages of measurements within each threshold
  if (sensor == MeasurementPackage::LASER)
  {
    // Count the LIDAR measurement
    lidar_measurements_ += 1;

    // Check if this NIS value is above the threshold
    if (NIS_value > lidar_NIS_thresh_)
    {
      lidar_NIS_above_thresh_ += 1;
    }

    // Calculate the percentage of the LIDAR NIS values that were above the NIS threshold,
    // out of all the LIDAR measurements
    percentage_values_above_thresh = ((float) lidar_NIS_above_thresh_) / lidar_measurements_;
  }
  else if (sensor == MeasurementPackage::RADAR)
  {
    // Count the Radar measurement
    radar_measurements_ += 1;

    // Check if this NIS value is above the threshold
    if (NIS_value > radar_NIS_thresh_)
    {
      radar_NIS_above_thresh_ += 1;
    }

    // Calculate the percentage of the Radar NIS values that were above the NIS threshold,
    // out of all the Radar measurements
    percentage_values_above_thresh = ((float) radar_NIS_above_thresh_) / radar_measurements_;
  }

  cout <<endl << "Percentage of NIS values above the threshold: " << percentage_values_above_thresh * 100 << "%" << endl;
  cout << "Accepted percentage interval (i.e. threshold interval): ( "
    << (chi2_threshold_percentage_ - chi2_threshold_percentage_epsilon_) * 100 << "% , "
    << (chi2_threshold_percentage_ + chi2_threshold_percentage_epsilon_) * 100 << "% )" << endl;

  // Check if the filter is consistent
  // Check if the percentage of NIS values is within (chi2_thresh - epsilon, chi2_thresh + epsilon)
  if (percentage_values_above_thresh > chi2_threshold_percentage_ - chi2_threshold_percentage_epsilon_
    && percentage_values_above_thresh < chi2_threshold_percentage_ + chi2_threshold_percentage_epsilon_)
  {
    cout << "Filter is CONSISTENT" << endl;
  }
  else
  {
    cout << "Filter is NOT CONSISTENT" << endl;
  }
}

void UKF::NormalizeAngle(double* angle_inout)
{
  *angle_inout = atan2(sin(*angle_inout), cos(*angle_inout));
}
