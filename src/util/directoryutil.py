import os
import subprocess
import torch


def get_repo_path():
    """
    Gets the local absolute path of the repository main directory.
    """
    try:
        repo_path = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
        return repo_path
    except subprocess.CalledProcessError:
        return None


def get_path(directory):
    """
    Here you can construct the path for a resource with respect to the repository main directory.
    """

    # df: Lazy function to not change the path on your own manually, just to make it work on the cluster.
    # Because on the cluster `get_repo_path` does not work. Therefore, we have to get hacky.
    # Easiest way: I only have a cpu, the cluster uses a gpu. Simple solution! 🙈
    if torch.cuda.is_available():
        repo_path = "/home/stud204/Project_RL"  # if you want to use the cluster
    else:
        repo_path = get_repo_path()  # if you want to use it locally
    return os.path.join(repo_path, directory)
