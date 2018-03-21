#include "ukf.h"
#include <cmath>
#include <iostream>
#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  is_initialized_ = false;
  previous_timestamp_ = 0;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 8;

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

  // State dimension
  n_x_ = 5;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  // Augmented state dimension
  n_aug_ = 7;

  n_z_radar_ = 3;

  // set weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_[0] = lambda_ / (lambda_ + n_aug_);
  weights_.tail(2 * n_aug_) =
      0.5 / (lambda_ + n_aug_) * VectorXd::Ones(2 * n_aug_);

  // measurement laser matrix
  H_laser_ = MatrixXd(2, n_x_);
  H_laser_ << 1, 0, 0, 0, 0, 0, 1, 0, 0, 0;

  // measurement noise covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_package) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    if (measurement_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates,
      // px = rho * cos(phi), py = rho * sin(phi)
      x_ << measurement_package.raw_measurements_[0] *
                cos(measurement_package.raw_measurements_[1]),
          measurement_package.raw_measurements_[0] *
              sin(measurement_package.raw_measurements_[1]),
          0, 0, 0;
    } else if (measurement_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << measurement_package.raw_measurements_[0],
          measurement_package.raw_measurements_[1], 0, 0, 0;
    }

    previous_timestamp_ = measurement_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // compute the time elapsed between the current and previous measurements
  float dt = (measurement_package.timestamp_ - previous_timestamp_) /
             1000000.0;  // dt - expressed in seconds
  previous_timestamp_ = measurement_package.timestamp_;
  Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (measurement_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(measurement_package.raw_measurements_);
  } else if (measurement_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(measurement_package.raw_measurements_);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} dt the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double dt) {
  MatrixXd Xsig_aug = MatrixXd(n_x_, 2 * n_x_ + 1);
  GenerateAugmentedSigmaPoints(&Xsig_aug);

  SigmaPointPrediction(Xsig_aug, dt);

  PredictMeanAndCovariance();
}

void UKF::GenerateAugmentedSigmaPoints(MatrixXd* Xsig_out) {
  // create augmented mean vector
  VectorXd x_aug = VectorXd::Zero(n_aug_);

  // create augmented mean state
  x_aug.head(n_x_) = x_;

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  // calculate square root of P_aug
  MatrixXd A_aug = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  Xsig_aug.block(0, 1, n_aug_, n_aug_) =
      x_aug * MatrixXd::Ones(1, n_aug_) + sqrt(lambda_ + n_aug_) * A_aug;
  Xsig_aug.block(0, n_aug_ + 1, n_aug_, n_aug_) =
      x_aug * MatrixXd::Ones(1, n_aug_) - sqrt(lambda_ + n_aug_) * A_aug;

  // print result
  // std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  // write result
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(const MatrixXd& Xsig_aug, double dt) {
  // create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // predict sigma points
  VectorXd x = VectorXd(n_aug_);
  VectorXd nu = VectorXd(n_x_);
  VectorXd x_deterministic = VectorXd(n_x_);
  for (short i = 0; i < Xsig_aug.cols(); i++) {
    float v, ksi, ksi_dot, nu_a, nu_ksi_dd;
    x = Xsig_aug.col(i);
    v = x(2);
    ksi = x(3);
    ksi_dot = x(4);
    nu_a = x(5);
    nu_ksi_dd = x(6);

    nu(0) = 0.5 * dt * dt * cos(ksi) * nu_a;
    nu(1) = 0.5 * dt * dt * sin(ksi) * nu_a;
    nu(2) = dt * nu_a;
    nu(3) = 0.5 * dt * dt * nu_ksi_dd;
    nu(4) = dt * nu_ksi_dd;
    if (fabs(ksi_dot) < 0.001) {
      x_deterministic(0) = v * cos(ksi) * dt;
      x_deterministic(1) = v * sin(ksi) * dt;
    } else {
      x_deterministic(0) = v / ksi_dot * (sin(ksi + ksi_dot * dt) - sin(ksi));
      x_deterministic(1) = v / ksi_dot * (-cos(ksi + ksi_dot * dt) + cos(ksi));
    }
    x_deterministic(2) = 0;
    x_deterministic(3) = ksi_dot * dt;
    x_deterministic(4) = 0;
    Xsig_pred.col(i) = x.head(n_x_) + x_deterministic + nu;
  }
  // print result
  // std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  // write result
  Xsig_pred_ = Xsig_pred;
}

void UKF::PredictMeanAndCovariance() {
  // create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  // create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);

  VectorXd deviation = VectorXd(n_x_);

  // predict state mean
  x = Xsig_pred_ * weights_;
  // predict state covariance matrix
  for (short i = 0; i < 2 * n_aug_ + 1; i++) {
    deviation = Xsig_pred_.col(i) - x;
    // angle normalization
    while (deviation(3) > M_PI) deviation(3) -= 2. * M_PI;
    while (deviation(3) < -M_PI) deviation(3) += 2. * M_PI;
    P += weights_[i] * deviation * deviation.transpose();
  }

  // print result
  // std::cout << "Predicted state" << std::endl;
  // std::cout << x << std::endl;
  // std::cout << "Predicted covariance matrix" << std::endl;
  // std::cout << P << std::endl;

  // write result
  x_ = x;
  P_ = P;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(VectorXd raw_measurements) {
  VectorXd y = raw_measurements - H_laser_ * x_;

  MatrixXd S = H_laser_ * P_ * H_laser_.transpose() + R_laser_;
  MatrixXd PHt = P_ * H_laser_.transpose();
  MatrixXd K = P_ * H_laser_.transpose() * S.inverse();

  // consistency check - normalized innovation squared (NIS)
  float nis_eps = y.transpose() * S.inverse() * y;
  // chi^2 95-percentile with 2 degrees of freedom (px and py) is 5.991
  float nis_threshold = 5.991;
  cout << "NIS L " << nis_eps;
  if (nis_eps > nis_threshold) {
    cout << " - too high";
  }
  cout << endl;

  // new estimate
  x_ = x_ + (K * y);
  P_ = (MatrixXd::Identity(n_x_, n_x_) - K * H_laser_) * P_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(VectorXd raw_measurements) {
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_radar_);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);

  MatrixXd Zsig = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);

  PredictRadarMeasurement(&z_pred, &S, &Zsig);

  UpdateState(z_pred, S, Zsig, raw_measurements);

  // consistency check - normalized innovation squared (NIS)
  float nis_eps = (raw_measurements - z_pred).transpose() * S.inverse() *
                  (raw_measurements - z_pred);
  // chi^2 95-percentile with 3 degrees of freedom (px and py) is 7.815
  float nis_threshold = 7.815;
  cout << "NIS R " << nis_eps;
  if (nis_eps > nis_threshold) {
    cout << " - too high";
  }
  cout << endl;
}

void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out,
                                  MatrixXd* Zsig_out) {
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_radar_);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
  S.fill(0.0);

  // transform sigma points into measurement space
  for (short i = 0; i < 2 * n_aug_ + 1; i++) {
    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double ksi = Xsig_pred_(3, i);

    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
    Zsig(1, i) = atan2(p_y, p_x);
    Zsig(2, i) = (p_x * cos(ksi) + p_y * sin(ksi)) * v / Zsig(0, i);
  }
  // calculate mean predicted measurement
  z_pred = Zsig * weights_;
  // calculate measurement covariance matrix S
  for (short i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd deviation = Zsig.col(i) - z_pred;
    // angle normalization
    while (deviation(1) > M_PI) deviation(1) -= 2. * M_PI;
    while (deviation(1) < -M_PI) deviation(1) += 2. * M_PI;
    S += weights_[i] * deviation * deviation.transpose();
  }
  MatrixXd R = MatrixXd(n_z_radar_, n_z_radar_);
  R.fill(0.0);
  R(0, 0) = std_radr_ * std_radr_;
  R(1, 1) = std_radphi_ * std_radphi_;
  R(2, 2) = std_radrd_ * std_radrd_;
  S += R;

  // print result
  // std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  // std::cout << "S: " << std::endl << S << std::endl;

  // write result
  *z_out = z_pred;
  *S_out = S;
  *Zsig_out = Zsig;
}

void UKF::UpdateState(const VectorXd& z_pred, const MatrixXd& S,
                      const MatrixXd& Zsig, const VectorXd& z) {
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);

  MatrixXd K = MatrixXd(n_x_, n_z_radar_);

  Tc.fill(0.0);
  // calculate cross correlation matrix
  for (short i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd deviation_x = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (deviation_x(3) > M_PI) deviation_x(3) -= 2. * M_PI;
    while (deviation_x(3) < -M_PI) deviation_x(3) += 2. * M_PI;

    VectorXd deviation_z = Zsig.col(i) - z_pred;
    // angle normalization
    while (deviation_z(1) > M_PI) deviation_z(1) -= 2. * M_PI;
    while (deviation_z(1) < -M_PI) deviation_z(1) += 2. * M_PI;
    Tc += weights_[i] * deviation_x * deviation_z.transpose();
  }

  // calculate Kalman gain K;
  K = Tc * S.inverse();

  // update state mean and covariance matrix
  // residual
  VectorXd z_diff = z - z_pred;
  // angle normalization
  while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
  x_ += K * z_diff;

  P_ -= K * S * K.transpose();

  // print result
  // std::cout << "Updated state x: " << std::endl << x_ << std::endl;
  // std::cout << "Updated state covariance P: " << std::endl << P_ <<
  // std::endl;
}
