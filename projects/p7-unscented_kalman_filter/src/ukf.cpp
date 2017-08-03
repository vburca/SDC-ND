#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

const float UKF::EPS = 0.5f;
int iteration = 0;

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
  std_a_ = 2;

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

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  double weight0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++)
  {
    double weight = 0.5 / (lambda_ + n_aug_);
    weights_(i) = weight;
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  cout << "\n\nProcessMeasurement" << endl;
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_)
  {
    // Initialize state vector x
    x_ << 1., 1., 0., 0., 0.;

    // Initialize state covariance matrix P
    P_ << 1., 0., 0., 0., 0.,
        0., 1., 0., 0., 0.,
        0., 0., 1., 0., 0.,
        0., 0., 0., 1., 0.,
        0., 0., 0., 0., 1.;

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

    // Avoid small values
    // if (fabs(x_(0)) < 0.001 && fabs(x_(1)) < 0.001)
    // {
    //   x_(0) = EPS;
    //   x_(1) = EPS;
    // }

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

  cout << "Before Update x_:\n" << x_ << endl;

  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }

  cout << "End Processing x_:\n" << x_ << endl;

  iteration += 1;
  if (iteration == 1)
  {
    exit(1);
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
  cout << "Prediction" << endl;
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
}

void UKF::PredictLidarMeasurement(VectorXd* z_out, MatrixXd* Zsig_out, MatrixXd* S_out)
{
  cout << "PredictLidarMeasurement" << endl;
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

  // Define measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;

  // Add measurement noise covariance matrix
  S = S + R;

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
}

void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* Zsig_out, MatrixXd* S_out)
{
  cout << "PredictRadarMeasurement" << endl;
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
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // Measurement model
    double rho = sqrt(p_x * p_x + p_y * p_y);
    double phi = atan2(p_y, p_x);
    double rho_dot = 0.0;

    if (rho > 0.0001)
    {
      rho_dot = (p_x * v1 + p_y * v2) / rho;
    }

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
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Define measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_ * std_radr_, 0, 0,
       0, std_radphi_ * std_radphi_, 0,
       0, 0, std_radrd_ * std_radrd_;

  // Add measurement noise covariance matrix
  S = S + R;

  // Write result
  *z_out = z_pred;
  *Zsig_out = Zsig;
  *S_out = S;
}

void UKF::UpdateState(VectorXd& z, VectorXd& z_pred, MatrixXd& Zsig, MatrixXd& S,
    MeasurementPackage::SensorType sensor, VectorXd* x_out, MatrixXd* P_out)
{
  cout << "UpdateState" << endl;
  // Create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  // Create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, z_pred.size());

  cout << "Xsig_pred_:\n" << Xsig_pred_ << endl;
  cout << "x_:\n" << x_ << endl;
  cout << "Zsig:\n" << Zsig << endl;
  cout << "z_pred:\n" << z_pred <<endl;
  cout << "weights_:\n" << weights_ << endl;

  // Calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    cout << "****** [" << i << "]" << endl;
    cout << "Tc Before:\n" << Tc << endl;
    cout << "x_diff:\n" << x_diff << endl;
    cout << "z_diff:\n" << z_diff << endl;
    cout << "weights_(" << i << "):" << weights_(i) << endl;
    cout << "w * x_diff * z_diff_T:\n" << weights_(i) * x_diff * z_diff.transpose() << endl;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();

    cout << "Tc After:\n" << Tc << endl << "******" << endl;
  }

  cout << "Tc:\n" << Tc << endl;

  // Calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = z - z_pred;
  while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

  cout << "Before updating x_" << endl;
  cout << "x_:\n" << x_ << endl;
  cout << "K:\n" << K << endl;
  cout << "z_diff:\n" << z_diff << endl;

  // Update state mean and covariance matrix
  x = x + K * z_diff;
  P = P - K * S * K.transpose();

  cout << "Updated state x_: " << endl << x << endl;
  cout << "Updated state covariance P: " << endl << P << endl;

  *x_out = x;
  *P_out = P;
}


void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out)
{
  cout << "AugmentedSigmaPoints" << endl;
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

  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd& Xsig_aug, double delta_t, MatrixXd* Xsig_out)
{
  cout << "SigmaPointPrediction" << endl;
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double p_x      = Xsig_aug(0, i);
    double p_y      = Xsig_aug(1, i);
    double v        = Xsig_aug(2, i);
    double yaw      = Xsig_aug(3, i);
    double yawd     = Xsig_aug(4, i);
    double nu_a     = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

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
  cout << "PredictMeanAndCovariance" << endl;
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
  cout << "Predicted state mean\n" << x << endl;

  // Predict state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x;

    // Normalize yaw angle
    while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    P = P + weights_(i) * x_diff * x_diff.transpose();
  }
  cout << "Predicted state covariance matrix\n" << P << endl;

  *x_out = x;
  *P_out = P;
}
