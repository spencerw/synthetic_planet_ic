import os
import shutil
from genIC import genICfile

num_sims = 10
start_idx = 11
template_dir = 'template_mox'

for num in range(start_idx, start_idx + num_sims + 1):
    num_str = "{:02d}".format(num)
    dirpath = 'gen' + num_str
    print('Building ' + dirpath)

    # Duplicate the template directory
    shutil.copytree(template_dir, dirpath)

    # Generate the IC file
    icname = 'ic.dat'
    genICfile(icname)

    # Move the IC file into it
    os.rename(icname, dirpath + '/' + icname)

    # Update the jobfile name
    with open(dirpath + '/jobfile', 'r') as f:
      filedata = f.read()
    filedata = filedata.replace('{name}', dirpath)
    with open(dirpath + '/jobfile', 'w') as f:
      f.write(filedata)
