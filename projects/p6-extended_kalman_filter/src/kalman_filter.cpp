#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */

  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */

  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  UpdateState(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */

  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  // Calculate h(x') = (sqrt(px^2 + py^2), arctan(py/px), (px*vx + py*vy)/norm)
  MatrixXd hx = MatrixXd(3, 1);
  float norm = sqrt(px * px + py * py);

  // Check for division by zero - if norm is too close to zero, use Jacobian instead
  if (norm < 0.0001)
  {
    hx = H_ * x_;
  }
  else
  {
    hx << norm,
          atan2(py, px),
          (px * vx + py * vy) / norm;
  }

  // Measurement update
  VectorXd y = z - hx;
  // Normalize the angle
  y(1) = atan2(sin(y(1)), cos(y(1)));

  UpdateState(y);
}

void KalmanFilter::UpdateState(const Eigen::VectorXd &y)
{
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  // New estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
