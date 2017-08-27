#pragma once

#ifndef PID_H
#define PID_H

#include <vector>

class PID {
public:
  /*
  * Constructor
  */
  PID(double Kp, double Ki, double Kd);

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  // Not going to use this for the moment, trying to use the member list initialization
  // void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();

  void UseTwiddle();
  bool ShouldTwiddle();
  void Twiddle();

private:
  void UpdateCoeffs(std::vector<double>& p, std::vector<double>& dp);
  void ResetPIDErrors();

private:
  /*
  * Initialization
  */
  bool is_initialized_;

  /*
  * Errors
  */
  double p_error_;
  double i_error_;
  double d_error_;
  double prev_cte_;

  /*
  * Coefficients
  */
  double Kp_;
  double Ki_;
  double Kd_;

  /*
  * Coefficient step
  */
  double Kp_d_ = 0.1;
  double Ki_d_ = 0.1;
  double Kd_d_ = 0.1;

  /*
  * PID error
  */
  double err_ = 0.0;
  double best_err_ = 1e+10; // Initialize with large value

  /*
  * Number of PID steps executed (updates)
  */
  int steps_ = 0;
  const int MIN_STEPS_ = 500;

  /*
  * Twiddle accuracy tolerance
  */
  double tolerance_ = 0.2;

  /*
  * Use twiddle flag
  */
  bool use_twiddle_ = false;

  /*
  * Twiddle update index
  */
  int twiddle_index_ = 0;

  /*
  * Track the twiddle state in regards to updating steps
  */
  int twiddle_state_ = 0;
};

#endif /* PID_H */
