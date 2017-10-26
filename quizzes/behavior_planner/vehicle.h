#ifndef VEHICLE_H
#define VEHICLE_H

#include <iostream>
#include <random>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <iterator>

using namespace std;

class Vehicle {
public:

  /**
   * Cost weights
   */
  #define COLLISION     10e6
  #define DANGER        10e5
  #define REACH_GOAL    10e5
  #define COMFORT       10e4
  #define EFFICIENCY    10e2

  #define DESIRED_BUFFER    1.5
  #define PLANNING_HORIZON  2

  struct collider{

    bool collision ; // is there a collision?
    int  time; // time collision happens

  };

  int L = 1;

  int preferred_buffer = 6; // impacts "keep lane" behavior.

  int lane;

  int s;

  int v;

  int a;

  int target_speed;

  int lanes_available;

  int max_acceleration;

  int goal_lane;

  int goal_s;

  string state;

  /**
  * Constructor
  */
  Vehicle(int lane, int s, int v, int a);

  /**
  * Destructor
  */
  virtual ~Vehicle();

  void update_state(map<int, vector <vector<int> > > predictions);

  void configure(vector<int> road_data);

  string display();

  void increment(int dt);

  vector<int> state_at(int t);

  bool collides_with(Vehicle other, int at_time);

  collider will_collide_with(Vehicle other, int timesteps);

  void realize_state(map<int, vector < vector<int> > > predictions);

  void realize_constant_speed();

  int _max_accel_for_lane(map<int,vector<vector<int> > > predictions, int lane, int s);

  void realize_keep_lane(map<int, vector< vector<int> > > predictions);

  void realize_lane_change(map<int,vector< vector<int> > > predictions, string direction);

  void realize_prep_lane_change(map<int,vector< vector<int> > > predictions, string direction);

  vector<vector<int> > generate_predictions(int horizon);

  /**
   * Helper structs
   */
  struct Snapshot
  {
      int lane;
      int s;
      int v;
      int a;
      string state;
  };

  inline Snapshot save_snapshot()
  {
      return {lane, s, v, a, state};
  }

  void restore_from_snapshot(Snapshot snap)
  {
      this->lane = snap.lane;
      this->s = snap.s;
      this->v = snap.v;
      this->a = snap.a;
      this->state = snap.state;
  }

  struct TrajectoryData
  {
      int proposed_lane;
      double avg_speed;
      double max_acc;
      double rms_acc;
      double closest_approach;
      double end_distance_to_goal;
      double end_lanes_from_goal;
      Vehicle::collider collides;
  };

  vector<Snapshot> trajectory_for_state(string state,
                map<int, vector<vector<int>>> predictions,
                int horizon);

  /**
   * Cost functions
   */

   double change_lane_cost(vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions,
                Vehicle::TrajectoryData data);

   double distance_from_goal_lane(vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions,
                Vehicle::TrajectoryData data);

   double inefficiency_cost(vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions,
                Vehicle::TrajectoryData data);

   double collision_cost(vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions,
                Vehicle::TrajectoryData data);

   double buffer_cost(vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions,
                Vehicle::TrajectoryData data);

    double rms_acc_cost(vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions,
                Vehicle::TrajectoryData data);

   double calculate_cost(vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions);

   TrajectoryData get_helper_data(vector<Vehicle::Snapshot> trajectory,
                map<int, vector<vector<int>>> predictions);

   bool check_collision(Vehicle::Snapshot snapshot, int s_prev, int s_now);

   map<int, vector<vector<int>>> filter_predictions_by_lane(map<int, vector<vector<int>>> predictions, int lane);

};

#endif