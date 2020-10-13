#   ______   __    __  __    __  __       __  __       __ 
#  /      \ /  |  /  |/  |  /  |/  \     /  |/  \     /  |
# /$$$$$$  |$$ |  $$ |$$ |  $$ |$$  \   /$$ |$$  \   /$$ |
# $$ |  $$/ $$ |__$$ |$$ |  $$ |$$$  \ /$$$ |$$$  \ /$$$ |
# $$ |      $$    $$ |$$ |  $$ |$$$$  /$$$$ |$$$$  /$$$$ |
# $$ |   __ $$$$$$$$ |$$ |  $$ |$$ $$ $$/$$ |$$ $$ $$/$$ |
# $$ \__/  |$$ |  $$ |$$ \__$$ |$$ |$$$/ $$ |$$ |$$$/ $$ |
# $$    $$/ $$ |  $$ |$$    $$/ $$ | $/  $$ |$$ | $/  $$ |
#  $$$$$$/  $$/   $$/  $$$$$$/  $$/      $$/ $$/      $$/

#    _____          _         __             _    _       _                        _    _ __  __       _       _   _                      __   __  __               
#   / ____|        | |       / _|           | |  | |     | |                      | |  | |  \/  |     | |     | | (_)                    / _| |  \/  |              
#  | |     ___   __| | ___  | |_ ___  _ __  | |__| | __ _| | ___     __ _  ___ ___| |  | | \  / |_   _| | __ _| |_ _  ___  _ __     ___ | |_  | \  / | __ _ ___ ___ 
#  | |    / _ \ / _` |/ _ \ |  _/ _ \| '__| |  __  |/ _` | |/ _ \   / _` |/ __/ __| |  | | |\/| | | | | |/ _` | __| |/ _ \| '_ \   / _ \|  _| | |\/| |/ _` / __/ __|
#  | |___| (_) | (_| |  __/ | || (_) | |    | |  | | (_| | | (_) | | (_| | (_| (__| |__| | |  | | |_| | | (_| | |_| | (_) | | | | | (_) | |   | |  | | (_| \__ \__ \
#   \_____\___/ \__,_|\___| |_| \___/|_|    |_|  |_|\__,_|_|\___/   \__,_|\___\___|\____/|_|  |_|\__,_|_|\__,_|\__|_|\___/|_| |_|  \___/|_|   |_|  |_|\__,_|___/___/

# Author: RUBY WRIGHT 
                                                                                                                                                                                                                                                                                                     
# GenData-AveProps.py - Generation script to run gen_averaged_accretion_data function.

# Preamble
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import h5py
import time 
import os
import sys
import argparse

sys.path.append('/Users/ruby/Documents/GitHub/CHUMM/') # may need to specify
sys.path.append('/home/rwright/Software/CHUMM/') # may need to specify
from STFTools import *
from AccretionTools import *
from ParticleTools import *
from VRPythonTools import *
from GenPythonTools import *
from ProcessingTools import *
from multiprocessing import Process, cpu_count


############ 0. ARGUMENT PROCESSING ############
# Parse the arguments for accretion calculation
if True:
    parser=argparse.ArgumentParser()
    parser.add_argument('-path',type=str, default=None,
                        help='Folder to analyse')
    parser.add_argument('-recycling',type=int, default=0,
                        help='Whether to include recycling/transfer breakdown')

path=parser.parse_args().path
recyc=parser.parse_args().recycling

gen_base_accretion_catalogue(path=path,recycling=bool(recyc))
append_accretion_catalogue(path=path)
