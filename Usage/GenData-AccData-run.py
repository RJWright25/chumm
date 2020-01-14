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
                                                                                                                                                                  
                                                                                                                                                                  
# GenData-AccData-run.py - Script to run the accretion data algorithm. 
# Author: RUBY WRIGHT 

# Preamble
import warnings
warnings.filterwarnings("ignore")
import os
import sys

sys.path.append('/Users/ruby/Documents/GitHub/CHUMM/')
sys.path.append('/home/rwright/CHUMM/')
from GenPythonTools import *

# Run script
if 'Users' in os.listdir('/'):
    chummdir='/Users/ruby/Documents/GitHub/CHUMM/'
else:
    chummdir='/home/rwright/CHUMM/'
sys.path.append(chummdir)
run_script=chummdir+'Usage/GenData-AccData.py'

# Job details
slurm=False
email=True
wall_time="0-04:00:00"
num_processes_calc=1
total_mem_perprocess=8#GB

# Algorithm Details
partdata=0
snaps=[27]
pre=1
post=1
r200_facs_in=[1]; r200_facs_in=list_to_string(r200_facs_in)
r200_facs_out=[1]; r200_facs_out=list_to_string(r200_facs_out)
vmax_facs_in=[0,0.125,0.25,0.375,0.5,0.75,1]; vmax_facs_in=list_to_string(vmax_facs_in)
vmax_facs_out=[0,0.125,0.25,0.375,0.5,0.75,1]; vmax_facs_out=list_to_string(vmax_facs_out)
gen_ad=1
col_ad=1
hil_lo=-1
hil_hi=-1

# Submit/ run
if hil_lo==-1:
    test=False
else:
    test=True

filename=sys.argv[0]
runcwd=os.getcwd()
runname=runcwd.split('-')[-1]
if not os.path.exists('job_logs'):
    os.mkdir('job_logs')

if slurm:
    for snap in snaps:
        if test:
            jobname=(filename.split('-')[1]).split('_')[0]+'-'+runname+f'_pre{str(pre).zfill(2)}_post{str(post).zfill(2)}_snap{str(snap).zfill(3)}_np{str(num_processes_calc).zfill(3)}_test'
        else:
            jobname=(filename.split('-')[1]).split('_')[0]+'-'+runname+f'_pre{str(pre).zfill(2)}_post{str(post).zfill(2)}_snap{str(snap).zfill(3)}_np{str(num_processes_calc).zfill(3)}'
        jobscriptfilepath=f'job_logs/submit-{jobname}.slurm'
        if os.path.exists(jobscriptfilepath):
            os.remove(jobscriptfilepath)
        with open(jobscriptfilepath,"w") as jobfile:
            jobfile.writelines(f"#!/bin/sh\n")
            jobfile.writelines(f"#SBATCH --job-name={jobname}\n")
            jobfile.writelines(f"#SBATCH --nodes=1\n")
            jobfile.writelines(f"#SBATCH --ntasks-per-node={num_processes_calc}\n")
            jobfile.writelines(f"#SBATCH --mem={total_mem_perprocess*num_processes_calc}GB\n")
            jobfile.writelines(f"#SBATCH --time={wall_time}\n")
            jobfile.writelines(f"#SBATCH --output=job_logs/{jobname}.out\n")
            jobfile.writelines(f"#SBATCH --error=job_logs/{jobname}.err\n")
            if email:
                jobfile.writelines(f"#SBATCH --mail-type=ALL\n")
                jobfile.writelines(f"#SBATCH --mail-user=21486778@student.uwa.edu.au\n")
            jobfile.writelines(f" \n")
            jobfile.writelines(f"echo JOB START TIME\n")
            jobfile.writelines(f"date\n")
            jobfile.writelines(f"echo CPU DETAILS\n")
            jobfile.writelines(f"lscpu\n")
            jobfile.writelines(f"python {run_script}  -partdata {partdata} -r200_facs_in {r200_facs_in} -r200_facs_out {r200_facs_out} -vmax_facs_in {vmax_facs_in} -vmax_facs_out {vmax_facs_out} -np_calc {num_processes_calc} -snap {snap} -pre {pre} -post {post} -gen_ad {gen_ad} -col_ad {col_ad} -hil_lo {hil_lo} -hil_hi {hil_hi}\n")
            jobfile.writelines(f"echo JOB END TIME\n")
            jobfile.writelines(f"date\n")
        jobfile.close()
        os.system(f"sbatch {jobscriptfilepath}")

else:
    # Loop through desired calcs and submit
    for snap in snaps:
        os.system(f"python {run_script}  -partdata {partdata} -r200_facs_in {r200_facs_in} -r200_facs_out {r200_facs_out} -vmax_facs_in {vmax_facs_in} -vmax_facs_out {vmax_facs_out} -np_calc {num_processes_calc} -snap {snap} -pre {pre} -post {post} -gen_ad {gen_ad} -col_ad {col_ad} -hil_lo {hil_lo} -hil_hi {hil_hi}\n")