## Coefficient Effects
I started playing initially with the coefficients manually, starting from the values provided in the quizzes. I noticed that the P and D coefficients had a bigger influence over the trajectory, compared to the I coefficient. I think this might have been because there was not much steering drift, maybe.  

The influence of the P coefficient was visible right from the start, seeing how fast the car was approaching the target trajectory (i.e. making a larger arc, going towards the sides of the lane, or having a more tight arc). The D coefficient was also fairly visible when trying to keep a straight line - as the lectures presented, this coefficient helped a lot in avoiding overshooting the target trajectory.

## Choosing the final hyperparameters
As mentioned before, I started with initial values from lectures, and then tweaked them manually, according to the reasoning and observations described above, until I got to a set of parameters that were keeping the car on the track for a full circuit.

Once I achieved this, the basic "challenge", I then implemented the "twiddle" behavior, with the goal of observing how I could improve these parameters. I left twiddle to run for a long time until the tolerance was met by the parameter steps, and the total error (in regards to the target trajectory) became very small. The caveat of this approach was that I was basically training only for the first N steps of the track (I used N = 500, and was only recording errors after the 500th step; so overall I was training over 2 * 500 steps). After the twiddling algorithm stopped, I used the parameters given as a result for the full circuit, and I saw that these were over-trained parameters that did not perform well on the full circuit.

I am still not sure how twiddling could avoid such overtraining or if I implemented something badly...

In the end, I used some of the twiddled parameters and the rest from my manual configuration, and the car was able to run around the track, following the target trajectory.
