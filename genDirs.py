import os
import shutil
from genIC import genICfile

num_sims = 10

for num in range(1, num_sims+1):
    num_str = "{:02d}".format(num)
    dirpath = 'gen' + num_str
    print('Building ' + dirpath)

    # Duplicate the template directory
    shutil.copytree('template', dirpath)

    # Generate the IC file
    icname = 'ic.dat'
    genICfile(icname)

    # Move the IC file into it
    os.rename(icname, dirpath + '/' + icname)

    # Update the jobfile name
    with open(dirpath + '/jobfileNVME', 'r') as f:
      filedata = f.read()
    filedata = filedata.replace('{name}', dirpath)
    with open(dirpath + '/jobfileNVME', 'w') as f:
      f.write(filedata)
