import os
from shutil import copytree, ignore_patterns


def copy_src(wdir, current_file=None, dst_dir="src", patterns=("*outputs*", "*__pycache__*", "*.pth", "*.pkl", "*.out", "*results*", "*.pdf")):
    if current_file is None:
        cwd_root = wdir
    else:
        current_file_fullname = os.path.join(wdir, current_file)
        cwd_root = os.path.dirname(current_file_fullname)
    print(f'Copy code from {cwd_root} to {dst_dir}')
    print(f'Excluded patterns: {patterns}')
    copytree(cwd_root, dst_dir, ignore=ignore_patterns(*patterns))
    # copytree(cwd_root, dst_dir, ignore=ignore_patterns(*patterns), dirs_exist_ok=True) # python 3.8
    print('Done!')
