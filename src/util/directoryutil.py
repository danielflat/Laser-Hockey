import os
import subprocess


def get_repo_path():
    """
    Gets the local absolute path of the repository main directory.
    """
    try:
        repo_path = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
        return repo_path
    except subprocess.CalledProcessError:
        return "/mnt/beegfs/home/stud204/Project_RL"  # not a good style but it has to work on the cluster
        # return None


def get_path(directory):
    """
    Here you can construct the path for a resource with respect to the repository main directory.
    """
    repo_path = get_repo_path()
    return os.path.join(repo_path, directory)
