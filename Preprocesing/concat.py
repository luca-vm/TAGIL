import os
import tarfile
from io import BytesIO
import numpy as np
import re

def concatenate_tar_files(tar_files, output_tar):
    with tarfile.open(output_tar, 'w:bz2') as out_tar:
        # Create a directory in the output tar file
        combined_data_dir = 'combined_data/'
        
        for tar_file in tar_files:
            try:
                with tarfile.open(tar_file, 'r:bz2') as tar:
                    for member in tar.getmembers():
                        if member.isfile():  # Check if the member is a regular file
                            file_content = tar.extractfile(member).read()
                            # Modify the member name to place all files in the same directory
                            member.name = os.path.join(combined_data_dir, os.path.basename(member.name))
                            info = tarfile.TarInfo(name=member.name)
                            info.size = len(file_content)
                            out_tar.addfile(tarinfo=info, fileobj=BytesIO(file_content))
            except (EOFError, tarfile.ReadError) as e:
                print(f"Warning: Skipping corrupted file {tar_file}: {e}")

def concatenate_txt_files(txt_files, output_txt):
    with open(output_txt, 'w') as out_txt:
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r') as in_txt:
                    out_txt.write(in_txt.read())
            except IOError as e:
                print(f"Warning: Skipping file {txt_file}: {e}")



def concatenate_files_in_directory(parent_dir):
    tar_files = sorted([os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if re.search(r'\d+\.tar\.bz2$', f)])
    # opt_files = sorted([os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith('_opt.tar.bz2')])
    # sal_files = sorted([os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith('_sal.tar.bz2')])
    txt_files = sorted([os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith('.txt')])

    output_tar = os.path.join(parent_dir, "train.tar.bz2")
    # output_opt = os.path.join(parent_dir, "val_opt.tar.bz2")
    # output_sal = os.path.join(parent_dir, "val_sal.tar.bz2")
    output_txt = os.path.join(parent_dir, "train.txt")
    
    # output_tar = os.path.join(parent_dir, "train.tar.bz2")
    # output_opt = os.path.join(parent_dir, "val_opt.tar.bz2")
    # output_sal = os.path.join(parent_dir, "val_sal.tar.bz2")
    # output_txt = os.path.join(parent_dir, "train.txt")
    
    # output_tar = os.path.join(parent_dir, "train.tar.bz2")
    # output_opt = os.path.join(parent_dir, "test_opt.tar.bz2")
    # output_sal = os.path.join(parent_dir, "test_sal.tar.bz2")
    # output_txt = os.path.join(parent_dir, "train.txt")

    concatenate_tar_files(tar_files, output_tar)
    # concatenate_tar_files(opt_files, output_opt)
    # concatenate_tar_files(sal_files, output_sal)
    concatenate_txt_files(txt_files, output_txt)

    print("Concatenation completed. Files saved in:", parent_dir)

if __name__ == "__main__":
    parent_directory = "/datasets/lvmayer/berzerk/train"
    # parent_directory = "./ms_pacman/val_data"
    # parent_directory = "./ms_pacman/test_data"
    concatenate_files_in_directory(parent_directory)
