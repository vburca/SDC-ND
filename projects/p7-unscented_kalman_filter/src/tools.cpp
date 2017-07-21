#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
  rmse.fill(0);

  // Check validity of inputs
  // * the estimation vector size should not be zero
  if (estimations.size() == 0)
  {
    cout << "Estimations size is zero!" << endl;
    return rmse;
  }

  // * the estimation vector size should equal the ground truth vector size
  if (estimations.size() != ground_truth.size())
  {
    cout << "Estimations vector and ground truth vector have different sizes!" << endl;
    return rmse;
  }

  // Accumulate squared residuals
  for (int i = 0; i < estimations.size(); i++)
  {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // Calculate the mean of the residual
  rmse = rmse / estimations.size();

  // Calculate the square root of the mean of the residual
  rmse = rmse.array().sqrt();

  return rmse;
}