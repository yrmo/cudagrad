import os
import shutil
import subprocess

import torch

build_dir = 'build'
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)

os.mkdir(build_dir)
os.chdir(build_dir)

cmake_prefix_path = torch.utils.cmake_prefix_path
subprocess.run(['cmake', f'-DCMAKE_PREFIX_PATH={cmake_prefix_path}', '..'])
subprocess.run(['cmake', '--build', '.'])

for file in os.listdir('.'):
    if file.startswith('aten_'):
        subprocess.run(f'./{file}', shell=True)
