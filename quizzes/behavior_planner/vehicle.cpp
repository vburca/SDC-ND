#include <iostream>
#include "vehicle.h"
#include <math.h>
#include <map>
#include <string>
#include <iterator>
#include <algorithm>

/**
 * Initializes Vehicle
 */
Vehicle::Vehicle(int lane, int s, int v, int a) {

    this->lane = lane;
    this->s = s;
    this->v = v;
    this->a = a;
    state = "CS";
    max_acceleration = -1;

}

Vehicle::~Vehicle() {}

// TODO - Implement this method.
void Vehicle::update_state(map<int,vector < vector<int> > > predictions) {
    /*
    Updates the "state" of the vehicle by assigning one   of the
    following values to 'self.state':

    "KL" - Keep Lane
     - The vehicle will attempt to drive its target speed, unless there is
       traffic in front of it, in which case it will slow down.

    "LCL" or "LCR" - Lane Change Left / Right
     - The vehicle will IMMEDIATELY change lanes and then follow longitudinal
       behavior for the "KL" state in the new lane.

    "PLCL" or "PLCR" - Prepare for Lane Change Left / Right
     - The vehicle will find the nearest vehicle in the adjacent lane which is
       BEHIND itself and will adjust speed to try to get behind that vehicle.

    INPUTS
    - predictions
    A dictionary. The keys are ids of other vehicles and the values are arrays
    where each entry corresponds to the vehicle's predicted location at the
    corresponding timestep. The FIRST element in the array gives the vehicle's
    current position. Example (showing a car with id 3 moving at 2 m/s):

    {
      3 : [
        {"s" : 4, "lane": 0},
        {"s" : 6, "lane": 0},
        {"s" : 8, "lane": 0},
        {"s" : 10, "lane": 0},
      ]
    }

    */

    // Define possible states
    vector<string> states;

    if (this->state == "KL")
    {
        states = { "KL", "PLCL", "PLCR" };
    }
    else if (this->state == "PLCL")
    {
        states = { "KL", "PLCL", "LCL" };
    }
    else if (this->state == "PLCR")
    {
        states = { "KL", "PLCR", "LCR" };
    }
    else if (this->state == "LCL")
    {
        states = { "KL", "LCL" };
    }
    else if (this->state == "LCR")
    {
        states = { "KL", "LCR" };
    }

    // Erase impossible states
    if (this->lane == 0)
    {
        states.erase(std::remove(states.begin(), states.end(), "LCR"), states.end());
        states.erase(std::remove(states.begin(), states.end(), "PLCR"), states.end());
    }
    else if (this->lane == this->lanes_available - 1)
    {
        states.erase(std::remove(states.begin(), states.end(), "LCL"), states.end());
        states.erase(std::remove(states.begin(), states.end(), "PLCL"), states.end());
    }

    vector<double> costs;
    for (string state : states)
    {
        vector<Vehicle::Snapshot> trajectory = trajectory_for_state(state, predictions, 5);
        costs.push_back(calculate_cost(trajectory, predictions));
    }

    double min_cost = costs[0];
    string best_state = states[0];
    for (int i = 0; i < states.size(); i++)
    {
        if (costs[i] < min_cost)
        {
            min_cost = costs[i];
            best_state = states[i];
        }
    }

    this->state = best_state;
}

vector<Vehicle::Snapshot> Vehicle::trajectory_for_state(
                string state,
                map<int, vector<vector<int>>> predictions,
                int horizon)
{
    Snapshot snap = this->save_snapshot();
    this->state = state;

    vector<Vehicle::Snapshot> trajectory;
    trajectory.push_back(snap);

    for (int i = 0; i < horizon; i++)
    {
        this->restore_from_snapshot(snap);
        this->state = state;

        this->realize_state(predictions);
        this->increment(1);

        Snapshot new_snap = this->save_snapshot();
        trajectory.push_back(new_snap);

        map<int, vector<vector<int>>>::iterator it = predictions.begin();
        while (it != predictions.end())
        {
            it->second.erase(it->second.begin());
            it++;
        }
    }

    this->restore_from_snapshot(snap);

    return trajectory;
}

void Vehicle::configure(vector<int> road_data) {
    /*
    Called by simulator before simulation begins. Sets various
    parameters which will impact the ego vehicle.
    */
    target_speed = road_data[0];
    lanes_available = road_data[1];
    goal_s = road_data[2];
    goal_lane = road_data[3];
    max_acceleration = road_data[4];
}

string Vehicle::display() {

    ostringstream oss;

    oss << "s:    " << this->s << "\n";
    oss << "lane: " << this->lane << "\n";
    oss << "v:    " << this->v << "\n";
    oss << "a:    " << this->a << "\n";

    return oss.str();
}

void Vehicle::increment(int dt = 1) {

    this->s += this->v * dt;
    this->v += this->a * dt;
}

vector<int> Vehicle::state_at(int t) {

    /*
    Predicts state of vehicle in t seconds (assuming constant acceleration)
    */
    int s = this->s + this->v * t + this->a * t * t / 2;
    int v = this->v + this->a * t;
    return {this->lane, s, v, this->a};
}

bool Vehicle::collides_with(Vehicle other, int at_time) {

    /*
    Simple collision detection.
    */
    vector<int> check1 = state_at(at_time);
    vector<int> check2 = other.state_at(at_time);
    return (check1[0] == check2[0]) && (abs(check1[1]-check2[1]) <= L);
}

Vehicle::collider Vehicle::will_collide_with(Vehicle other, int timesteps) {

    Vehicle::collider collider_temp;
    collider_temp.collision = false;
    collider_temp.time = -1;

    for (int t = 0; t < timesteps+1; t++)
    {
          if( collides_with(other, t) )
          {
            collider_temp.collision = true;
            collider_temp.time = t;
            return collider_temp;
        }
    }

    return collider_temp;
}

void Vehicle::realize_state(map<int,vector < vector<int> > > predictions) {

    /*
    Given a state, realize it by adjusting acceleration and lane.
    Note - lane changes happen instantaneously.
    */
    string state = this->state;
    if(state.compare("CS") == 0)
    {
        realize_constant_speed();
    }
    else if(state.compare("KL") == 0)
    {
        realize_keep_lane(predictions);
    }
    else if(state.compare("LCL") == 0)
    {
        realize_lane_change(predictions, "L");
    }
    else if(state.compare("LCR") == 0)
    {
        realize_lane_change(predictions, "R");
    }
    else if(state.compare("PLCL") == 0)
    {
        realize_prep_lane_change(predictions, "L");
    }
    else if(state.compare("PLCR") == 0)
    {
        realize_prep_lane_change(predictions, "R");
    }

}

void Vehicle::realize_constant_speed() {
    a = 0;
}

int Vehicle::_max_accel_for_lane(map<int,vector<vector<int> > > predictions, int lane, int s) {

    int delta_v_til_target = target_speed - v;
    int max_acc = min(max_acceleration, delta_v_til_target);

    map<int, vector<vector<int> > >::iterator it = predictions.begin();
    vector<vector<vector<int> > > in_front;
    while(it != predictions.end())
    {

        int v_id = it->first;

        vector<vector<int> > v = it->second;

        if((v[0][0] == lane) && (v[0][1] > s))
        {
            in_front.push_back(v);

        }
        it++;
    }

    if(in_front.size() > 0)
    {
        int min_s = 1000;
        vector<vector<int>> leading = {};
        for(int i = 0; i < in_front.size(); i++)
        {
            if((in_front[i][0][1]-s) < min_s)
            {
                min_s = (in_front[i][0][1]-s);
                leading = in_front[i];
            }
        }

        int next_pos = leading[1][1];
        int my_next = s + this->v;
        int separation_next = next_pos - my_next;
        int available_room = separation_next - preferred_buffer;
        max_acc = min(max_acc, available_room);
    }

    return max_acc;

}

void Vehicle::realize_keep_lane(map<int,vector< vector<int> > > predictions) {
    this->a = _max_accel_for_lane(predictions, this->lane, this->s);
}

void Vehicle::realize_lane_change(map<int,vector< vector<int> > > predictions, string direction) {
    int delta = -1;
    if (direction.compare("L") == 0)
    {
        delta = 1;
    }
    this->lane += delta;
    int lane = this->lane;
    int s = this->s;
    this->a = _max_accel_for_lane(predictions, lane, s);
}

void Vehicle::realize_prep_lane_change(map<int,vector<vector<int> > > predictions, string direction) {
    int delta = -1;
    if (direction.compare("L") == 0)
    {
        delta = 1;
    }
    int lane = this->lane + delta;

    map<int, vector<vector<int> > >::iterator it = predictions.begin();
    vector<vector<vector<int> > > at_behind;
    while(it != predictions.end())
    {
        int v_id = it->first;
        vector<vector<int> > v = it->second;

        if((v[0][0] == lane) && (v[0][1] <= this->s))
        {
            at_behind.push_back(v);

        }
        it++;
    }
    if(at_behind.size() > 0)
    {

        int max_s = -1000;
        vector<vector<int> > nearest_behind = {};
        for(int i = 0; i < at_behind.size(); i++)
        {
            if((at_behind[i][0][1]) > max_s)
            {
                max_s = at_behind[i][0][1];
                nearest_behind = at_behind[i];
            }
        }
        int target_vel = nearest_behind[1][1] - nearest_behind[0][1];
        int delta_v = this->v - target_vel;
        int delta_s = this->s - nearest_behind[0][1];
        if(delta_v != 0)
        {

            int time = -2 * delta_s/delta_v;
            int a;
            if (time == 0)
            {
                a = this->a;
            }
            else
            {
                a = delta_v/time;
            }
            if(a > this->max_acceleration)
            {
                a = this->max_acceleration;
            }
            if(a < -this->max_acceleration)
            {
                a = -this->max_acceleration;
            }
            this->a = a;
        }
        else
        {
            int my_min_acc = max(-this->max_acceleration,-delta_s);
            this->a = my_min_acc;
        }

    }

}

vector<vector<int> > Vehicle::generate_predictions(int horizon = 10) {

    vector<vector<int> > predictions;
    for( int i = 0; i < horizon; i++)
    {
      vector<int> check1 = state_at(i);
      vector<int> lane_s = {check1[0], check1[1]};
      predictions.push_back(lane_s);
      }
    return predictions;

}

/**
 * Definitions of cost functions
 */
double Vehicle::change_lane_cost(
                vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions,
                Vehicle::TrajectoryData data)
{
    // Penalizes lane changes AWAY from the goal lane and rewards
    // lane changes TOWARDS the goal lane.
    int proposed_lanes = data.end_lanes_from_goal;
    int cur_lanes = trajectory[0].lane;

    double cost = 0.0;
    if (proposed_lanes > cur_lanes)
    {
        cost = COMFORT;
    }
    else if (proposed_lanes < cur_lanes)
    {
        cost = -COMFORT;
    }

    return cost;
}

double Vehicle::distance_from_goal_lane(
                vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions,
                Vehicle::TrajectoryData data)
{
    // Penalizes distance from goal lane as a function of time left to target
    double distance = double(abs(data.end_distance_to_goal));
    distance = max(distance, 1.0);

    double time_to_goal = distance / data.avg_speed;
    double lanes = double(data.end_lanes_from_goal);
    double multiplier = 5 * lanes / time_to_goal;
    double cost = multiplier * REACH_GOAL;

    return cost;
}

double Vehicle::inefficiency_cost(
                vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions,
                Vehicle::TrajectoryData data)
{
    // Penalizes non-optimal (non-target) speed
    double speed = data.avg_speed;
    double target_speed = double(this->target_speed);
    double diff = target_speed - speed;
    double percentage = diff / target_speed;
    double multiplier = pow(percentage, 2.0);
    double cost = multiplier * EFFICIENCY;

    return cost;
}

double Vehicle::collision_cost(
                vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions,
                Vehicle::TrajectoryData data)
{
    // Penalizes collisions with other vehicles
    double cost = 0.0;

    if (data.collides.collision)
    {
        int time_to_collision = data.collides.time;
        double exponent = pow(double(time_to_collision), 2.0);
        double multiplier = exp(-exponent);

        cost = multiplier * COLLISION;
    }

    return cost;
}

double Vehicle::buffer_cost(
                vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions,
                Vehicle::TrajectoryData data)
{
    // Penalize small buffer time distance to other vehicles
    double cost = 0.0;
    double closest = data.closest_approach;

    if (closest = 0.0)
    {
        cost = 10 * DANGER;
    }
    else
    {
        double timesteps_away = closest / data.avg_speed;
        if (timesteps_away < DESIRED_BUFFER)
        {
            double multiplier = 1.0 - pow(timesteps_away / DESIRED_BUFFER, 2);
            cost = multiplier * DANGER;
        }
    }

    return cost;
}

double rms_acc_cost(
                vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions,
                Vehicle::TrajectoryData data)
{
    // Penalize accelerations that are too large.
    double cost = data.rms_acc * COMFORT;

    return cost;
}

double Vehicle::calculate_cost(
                vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions)
{
    Vehicle::TrajectoryData data = get_helper_data(trajectory, predictions);

    double cost = 0.0;
    cost += change_lane_cost(trajectory, predictions, data);
    cost += distance_from_goal_lane(trajectory, predictions, data);
    cost += inefficiency_cost(trajectory, predictions, data);
    cost += collision_cost(trajectory, predictions, data);
    cost += buffer_cost(trajectory, predictions, data);

    return cost;
}

Vehicle::TrajectoryData Vehicle::get_helper_data(
                vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions)
{
    Vehicle::Snapshot cur_snapshot = trajectory[0];
    Vehicle::Snapshot first = trajectory[1];
    Vehicle::Snapshot last = trajectory.back();

    int end_distance_to_goal = this->goal_s - last.s;
    int end_lanes_from_goal = abs(this->goal_lane - last.lane);

    double dt = double(trajectory.size());

    int proposed_lane = first.lane;
    double avg_speed = (last.s - cur_snapshot.s) / dt;

    Vehicle::collider collides;
    collides.collision = false;

    // Interested only in the predictions for the vehicles in the ego's lane
    map<int, vector<vector<int>>> filtered = filter_predictions_by_lane(predictions, proposed_lane);

    vector<int> accels;
    int closest_approach = 999999;
    Vehicle::Snapshot last_snap = trajectory[0];
    for (int i = 1; i <= PLANNING_HORIZON; i++)
    {
        Vehicle::Snapshot snap = trajectory[i];
        accels.push_back(snap.a);

        // Iterate over all the vehicles in the ego's lane
        map<int, vector<vector<int>>>::iterator it = filtered.begin();
        while (it != filtered.end())
        {
            int vehicle_id = it->first;
            vector<vector<int>> predicted_trajectory = it->second;

            vector<int> vehicle_state = predicted_trajectory[i];
            vector<int> last_vehicle_state = predicted_trajectory[i - 1];

            bool vehicle_collides = check_collision(snap, last_vehicle_state[1], vehicle_state[1]);

            if (vehicle_collides)
            {
                collides.collision = true;
                collides.time = i;
            }

            int dist = abs(vehicle_state[1] - snap.s);

            if (dist < closest_approach)
            {
                closest_approach = dist;
            }

            it++;
        }

        last_snap = snap;
    }

    int max_acc = 0;
    double rms_acc = 0.0;
    for (int acc : accels)
    {
        rms_acc += pow(acc, 2);

        if (abs(acc) >= max_acc)
        {
            max_acc = abs(acc);
        }
    }

    rms_acc = rms_acc / accels.size();

    Vehicle::TrajectoryData data;

    data.proposed_lane = proposed_lane;
    data.avg_speed = avg_speed;
    data.max_acc = max_acc;
    data.rms_acc = rms_acc;
    data.closest_approach = closest_approach;
    data.end_distance_to_goal = end_distance_to_goal;
    data.end_lanes_from_goal = end_lanes_from_goal;
    data.collides = collides;

    return data;
}

bool Vehicle::check_collision(Vehicle::Snapshot snapshot, int s_prev, int s_now)
{
    int s = snapshot.s;
    int v = snapshot.v;
    int v_target = s_now - s_prev;

    if (s_prev < s)
    {
        if (s_now >= s)
        {
            return true;
        }

        return false;
    }

    if (s_prev > s)
    {
        if (s_now <= s)
        {
            return true;
        }

        return false;
    }

    if (s_prev == s)
    {
        if (v_target > v)
        {
            return false;
        }

        return true;
    }
}

map<int, vector<vector<int>>> Vehicle::filter_predictions_by_lane(
                map<int, vector<vector<int>>> predictions,
                int lane)
{
    map<int, vector<vector<int>>> filtered_predictions;
    map<int, vector<vector<int>>>::iterator it = predictions.begin();

    // For each of the other vehicles, select the ones that are on the same lane as
    // the one the ego vehicle is on (i.e. the lane that is passed in).
    // Filter out the vehicles that have id -1 (i.e. are the ego vehicle)
    while (it != predictions.end())
    {
        int vehicle_id = it->first;
        vector<vector<int>> predicted_trajectory = it->second;

        if (predicted_trajectory[0][0] == lane && vehicle_id != -1)
        {
            filtered_predictions[vehicle_id] = predicted_trajectory;
        }

        it++;
    }

    return filtered_predictions;
}
