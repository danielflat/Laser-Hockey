#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=RL_Hockey

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day  #test    #day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=1-00:00:00  #10:00 #1-00:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"


#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out


# Do the task
echo STARTING CLUSTER!

# Set paths
CONTAINER=/common/singularityImages/rl_lecture_23.simg
OVERLAY=~/my_overlay7
#CONTAINER = ~/rl_lecture_23.simg

# Temporarily create an overlay
singularity overlay create --size 500 $OVERLAY

# Install dependencies
singularity exec --nv --overlay $OVERLAY $CONTAINER pip3 install --user --upgrade pip
singularity exec --nv --overlay $OVERLAY $CONTAINER pip3 install --user pyyaml
echo $OVERLAY

# Train the model
#singularity exec --nv /common/singularityImages/rl_lecture_23.simg python3 ~/Project_RL/main.py
singularity exec --overlay $OVERLAY $CONTAINER python3 ~/Project_RL/main.py

# Delete the overlay
rm $OVERLAY

echo FINISHED CLUSTER!

