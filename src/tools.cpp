#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  VectorXd residuals(4);
  residuals << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  if (estimations.size() == 0) {
    return rmse;
  }
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size()) {
    return rmse;
  }

  // accumulate squared residuals
  for (int i = 0; i < estimations.size(); ++i) {
    // ... your code here
    residuals = estimations[i] - ground_truth[i];
    rmse = rmse.array() + residuals.array() * residuals.array();
  }

  // calculate the mean
  rmse = rmse.array() / estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;
}
