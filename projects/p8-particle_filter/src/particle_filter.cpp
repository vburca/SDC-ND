/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#define _USE_MATH_DEFINES
#include <math.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    default_random_engine gen;

    num_particles = 10;

    normal_distribution<double> x_dist(x, std[0]);
    normal_distribution<double> y_dist(y, std[1]);
    normal_distribution<double> theta_dist(theta, std[2]);

    for (int i = 0; i < num_particles; i++)
    {
        Particle p;
        p.id = i;
        p.x = x_dist(gen);
        p.y = y_dist(gen);
        p.theta = theta_dist(gen);
        p.weight = 1.0;

        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;

    normal_distribution<double> x_dist(0, std_pos[0]);
    normal_distribution<double> y_dist(0, std_pos[1]);
    normal_distribution<double> theta_dist(0, std_pos[2]);

    for (Particle &p : particles)
    {
        // If moving in straight line, i.e. yaw rate close to 0
        if (yaw_rate < 1e-4)
        {
            // Use straight line motion model
            p.x += velocity * cos(p.theta) * delta_t + x_dist(gen);
            p.y += velocity * sin(p.theta) * delta_t + y_dist(gen);
        }
        else
        {
            // Bicycle motion model for turning vehicle
            p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + x_dist(gen);
            p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + y_dist(gen);
        }

        p.theta += yaw_rate * delta_t + theta_dist(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    for (LandmarkObs &obs : observations)
    {
        double min_d = INFINITY;

        for (LandmarkObs &prediction : predicted)
        {
            const double d = dist(obs.x, obs.y, prediction.x, prediction.y);

            if (d < min_d)
            {
                min_d = d;
                obs.id = prediction.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
        std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    weights.clear();

    for (Particle &p : particles)
    {
        // Transform vehicle observations to map observations
        vector<LandmarkObs> map_observations;
        transformObservations(p, observations, map_observations);

        // Filter out un-detectable landmarks
        vector<LandmarkObs> reachable_landmarks;
        filterLandmarks(p, map_landmarks, sensor_range, reachable_landmarks);

        // Associate landmarks
        dataAssociation(reachable_landmarks, map_observations);

        // Calculate the weight of the particle
        double weight = 1.0;
        const double gauss_norm = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);

        vector<int> assoc_ids;
        vector<double> sense_xs;
        vector<double> sense_ys;

        for (LandmarkObs &obs : map_observations)
        {
            Map::single_landmark_s associated_landmark = map_landmarks.landmark_list[obs.id - 1];
            const double term1 = pow(obs.x - associated_landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
            const double term2 = pow(obs.y - associated_landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
            const double obs_weight = gauss_norm * exp(-(term1 + term2));
            weight *= obs_weight;

            assoc_ids.push_back(associated_landmark.id_i);
            sense_xs.push_back(obs.x);
            sense_ys.push_back(obs.y);
        }

        p.weight = weight;
        weights.push_back(weight);

        p = SetAssociations(p, assoc_ids, sense_xs, sense_ys);
    }

    // Normalization of weights
    // - we don't need to do this due to how we do the resampling, using the std::discrete_distribution,
    // which already does normalization internally
}

void ParticleFilter::transformObservations(Particle &particle, vector<LandmarkObs> &observations, vector<LandmarkObs> &map_observations)
{
    // Transform each observation to map coordinates observation, using coordinate system equations.
    for (LandmarkObs &obs : observations)
    {
        LandmarkObs map_obs;
        map_obs.id = obs.id;
        map_obs.x = particle.x + cos(particle.theta) * obs.x - sin(particle.theta) * obs.y;
        map_obs.y = particle.y + sin(particle.theta) * obs.x + cos(particle.theta) * obs.y;

        map_observations.push_back(map_obs);
    }
}

void ParticleFilter::filterLandmarks(Particle &particle, Map map_landmarks, double sensor_range, std::vector<LandmarkObs> &reachable_landmarks)
{
    // Filter out landmarks that are beyond the sensor range limit, i.e. that could not have been detected from this particle's position
    for (Map::single_landmark_s landmark : map_landmarks.landmark_list)
    {
        const double d = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);
        if (d <= sensor_range)
        {
            LandmarkObs landmark_obs;
            landmark_obs.id = landmark.id_i;
            landmark_obs.x = landmark.x_f;
            landmark_obs.y = landmark.y_f;
            reachable_landmarks.push_back(landmark_obs);
        }
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;

    discrete_distribution<> weight_dist(weights.begin(), weights.end());

    vector<Particle> resampled_particles;
    for (int i = 0; i < num_particles; i++)
    {
        const int sampled_id = weight_dist(gen);
        resampled_particles.push_back(particles[sampled_id]);
    }

    particles.clear();
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

     return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
