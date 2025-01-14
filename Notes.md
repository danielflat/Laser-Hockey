# Notes


## Pendulum-v1
- Observation space: (3,) 
    - x-y: cartesian coordinates of the pendulum’s end in meters.
    - theta : angle in radians. 
    - tau: torque in N m. Defined as positive counter-clockwise.
- Action space: (1,)
  - The action is a ndarray with shape (1,) representing the torque applied to free end of the pendulum.
  - min: -2.0
  - max: 2.0

- Reward: 
  - r = -(theta2 + 0.1 * theta_dt2 + 0.001 * torque2)
  - Best reward possible: 0


- Episode Truncation 
  - The episode truncates at 200 time steps.



## CartPole-v1




## Hockey-01

## TODOS

- [ ] Add Winning, Draw and Lose rates for each agent and in the end (training and testing)
- [ ] fremde Algroithmen testen zum Debuggen
- [ ] Add checkpointing start for the agents