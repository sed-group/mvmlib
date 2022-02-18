""" Useful utility functions """

import os


def check_folder(folder='render/'):
    """
    check if folder exists, make if not present

    Parameters
    ----------
    folder : str, optional
        name of directory to check, by default 'render/'
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
