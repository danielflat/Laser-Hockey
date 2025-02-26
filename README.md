<h1 style="text-align: center">The Bests Around</h1>

---

<div style="display: flex; justify-content: space-around;">
    <h5>Daniel Flat</h5>
    <h5>Erik Ayari</h5>
    <h5>André Pfrommer</h5>
</div>

---

<div style="display: flex; justify-content: space-around;">
    <h5>Lecture: Reinforcement Learning</h5>
    <h5>Semester: WS24/25</h5>
    <h5>University of Tübingen</h5>
    <h5>Lecturer: Georg Martius</h5>
</div>

---

This is the Code for the Project Report "The Bests Around – Hockey Tournament RL 24/25".
<img src="hockey/assets/hockeyenv1.png">

### Instructions

To install the dependencies, run:

```bash
poetry install
```

### Training

To train an agent, run:

```bash
python main.py
```

The settings of the training can be adjusted in the `src/settings.py` file.
You can either train TD-MPC2, MPO or DDPG agents. The default is TD-MPC2.

### Testing

To play as a human against the trained agent, run:

```bash
python test.py
```

Our best TD-MPC2 agent is set as the default opponent. But you can change it in the `test.py`.

### Reproducing the Plots

Some of the plots in the report can be reproduced by the following scripts:

To reproduce Table 1, run:

```bash
python plots/eval_tdmpc2_hockey.py
```

To reproduce Table 2, run:

```bash
python plots/eval_mini_tournament.py
```

To reproduce Figure 6, run:

```bash
python plots/eval_tdmc2_hockey_other_env_recover_plots.py
```

where the first pendulum plot is from

```bash
python plots/eval_tdmpc2_other_env.py
```

and the hockey plots from

```bash
python plots/eval_tdmpc2_hockey_training.py
```

### Note:

The final checkpoints of the agents can be found in `final_checkpoints`.
For the plots of the paper, `final_checkpoints/tdmpc2-v2-all-i6 25-02-20 17_44_47_000061500.pth` was used.

During tournament, we used `final_checkpoints/tdmpc2-v2-all-i6 25-02-20 17_44_47_000061500.pth` in the beginning,
but changed to the better version `final_checkpoints/tdmpc2-v2-all-i7 25-02-20 17_44_47_000067500.pth`.

Now the default is `final_checkpoints/tdmpc2-v2-all-i7 25-02-20 17_44_47_000067500.pth` for all the scripts.






