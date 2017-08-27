#include "PID.h"
#include <iostream>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID(double Kp, double Ki, double Kd) : Kp_(Kp), Ki_(Ki), Kd_(Kd) {}

PID::~PID() {}

void PID::UpdateError(double cte)
{
    if (!is_initialized_)
    {
        prev_cte_ = cte;
        is_initialized_ = true;
    }
    p_error_ = cte;
    i_error_ += cte;
    d_error_ = cte - prev_cte_;

    prev_cte_ = cte;

    steps_ += 1;

    // Update error only after several steps have been taken, getting closer to the expected trajectory
    if (steps_ > MIN_STEPS_)
    {
        err_ += cte * cte;
    }
}

double PID::TotalError()
{
    return - Kp_ * p_error_ - Kd_ * d_error_ - Ki_ * i_error_;
}

bool PID::ShouldTwiddle()
{
    return use_twiddle_ &&
        Kp_d_ + Ki_d_ + Kd_d_ > tolerance_ &&
        steps_ > 2 * MIN_STEPS_;
}

void PID::Twiddle()
{
    cout << "Twiddling... state: " << twiddle_state_ << endl;

    cout << "Best error: " << best_err_ << endl;
    cout << "[Kp, Ki, Kd] = " << "[" << Kp_ << ", " << Ki_ << ", " << Kd_ << "]" << endl;

    vector<double> p = {Kp_, Ki_, Kd_};
    vector<double> dp = {Kp_d_, Ki_d_, Kd_d_};

    if (twiddle_state_ == 0)
    {
        p[twiddle_index_] += dp[twiddle_index_];
        twiddle_state_ = 1;

        UpdateCoeffs(p, dp);
        ResetPIDErrors();
        return;
    }

    double error = err_ / (steps_ - MIN_STEPS_);
    cout << "Calculated error: " << error << endl;

    if (twiddle_state_ == 1)
    {
        if (error < best_err_)
        {
            best_err_ = error;
            dp[twiddle_index_] *= 1.1;

            twiddle_index_ = (twiddle_index_ + 1) % p.size();
            twiddle_state_ = 0;

            UpdateCoeffs(p, dp);
            ResetPIDErrors();
            return;
        }
        else
        {
            p[twiddle_index_] -= 2 * dp[twiddle_index_];
            twiddle_state_ = 2;

            UpdateCoeffs(p, dp);
            ResetPIDErrors();
            return;
        }
    }

    if (twiddle_state_ == 2)
    {
        if (error < best_err_)
        {
            best_err_ = error;
            dp[twiddle_index_] *= 1.1;
        }
        else
        {
            p[twiddle_index_] += dp[twiddle_index_];
            dp[twiddle_index_] *= 0.9;
        }

        twiddle_index_ = (twiddle_index_ + 1) % p.size();
        twiddle_state_ = 0;
    }

    UpdateCoeffs(p, dp);
    ResetPIDErrors();
}

void PID::UseTwiddle()
{
    use_twiddle_ = true;
}

void PID::UpdateCoeffs(vector<double>& p, vector<double>& dp)
{
    Kp_ = p[0];
    Ki_ = p[1];
    Kd_ = p[2];

    Kp_d_ = dp[0];
    Ki_d_ = dp[1];
    Kd_d_ = dp[2];
}

void PID::ResetPIDErrors()
{
    err_ = 0.0;
    steps_ = 0.0;

    p_error_ = 0.0;
    i_error_ = 0.0;
    d_error_ = 0.0;
    prev_cte_ = 0.0;
}




