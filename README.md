<h1 style="text-align: center">Project_RL</h1>

---

<div style="display: flex; justify-content: space-around;">
    <h5>Daniel Flat</h5>
    <h5>Erik Ayari</h5>
    <h5>André Pfrommer</h5>
</div>

---

<div style="display: flex; justify-content: space-around;">
    <h5>Lecture: Reinforcement Learning</h5>
    <h5>Semester: WS25</h5>
    <h5>University: University of Tübingen</h5>
    <h5>Lecturer: Georg Martius</h5>
    </h5>
</div>

---

This is our project submission for the Reinforcement Learning lecture.



### Instructions (see [Original Repository](https://github.com/martius-lab/hockey-env))
#### hockey-env

This repository contains a hockey-like game environment for RL

##### Install

``python3 -m pip install git+https://github.com/martius-lab/hockey-env.git``

or add the following line to your Pipfile

``hockey = {editable = true, git = "https://git@github.com/martius-lab/hockey-env.git"}``


##### HockeyEnv

![Screenshot](assets/hockeyenv1.png)

``hockey.hockey_env.HockeyEnv``

A two-player (one per team) hockey environment.
For our Reinforcment Learning Lecture @ Uni-Tuebingen.
See Hockey-Env.ipynb notebook on how to run the environment.

The environment can be generated directly as an object or via the gym registry:

``env = gym.envs.make("Hockey-v0")``

There is also a version against the basic opponent (with options)

``env = gym.envs.make("Hockey-One-v0", mode=0, weak_opponent=True)``

