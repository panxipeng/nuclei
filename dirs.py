import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_dir(relative_path):
    dirs = relative_path.split('/')
    top_dir = ROOT_DIR
    for dir in dirs:
        top_dir = os.path.join(top_dir, dir)
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)

    return top_dir