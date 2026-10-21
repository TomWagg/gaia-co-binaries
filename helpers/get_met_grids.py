import re
from pathlib import Path
import numpy as np

grid_dir = Path("/mnt/ceph/users/twagg/metisse-grids/standard/hydrogen")

# pull the 7-digit code from each input file and convert to Z
metallicities = np.sort([
    int(re.match(r"Z(\d{7})_metallicity\.in", f.name).group(1)) * 1e-6
    for f in grid_dir.glob("Z*_metallicity.in")
]).round(6)

print(repr(metallicities))

print(repr(np.logspace(-4, np.log10(0.03), 50).round(5)))