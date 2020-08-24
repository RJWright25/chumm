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

# GenData-Recycling-run.py - Generation script to run particle recycling tracker. Edit runtime parameters as required. 
# Requirements: base halo data, particle histories, accretion data (with write_partdata) have been generated.

# Preamble
import warnings
warnings.filterwarnings("ignore")
import os
import sys

####################################################################################################
########################################## RUNTIME PARS ############################################

# Job details
slurm=False # whether or not to use slurm submit
email=False # email results y/n (if slurm)
address='21486778@student.uwa.edu.au' #email address (if slurm)
wall_time="0-04:00:00" #job time limit (if slurm)
total_mem=8 # total memory required (if slurm)

# Algorithm Details
snaps=[27] # snaps to run calculation for
mcut=10 # mass cut for adding properties to halo (in log10 M/Msun)
rundir=os.getcwd()
basepath=f'{rundir}/acc_data/pre01_post01_np04_FOFonly/' # path with generated accretion data

####################################################################################################
####################################################################################################

# Run Script
if 'Users' in os.listdir('/'):
    chummdir='/Users/ruby/Documents/GitHub/CHUMM/'
else:
    chummdir='/home/rwright/Software/CHUMM/'
    
run_script=chummdir+'Usage/GenData-Recycling.py'
sys.path.append('/home/rwright/software/read_eagle/build/lib/python3.7/site-packages/')

# Submit/ run
filename=sys.argv[0]
runcwd=os.getcwd()
runname=runcwd.split('-')[-1]
if not os.path.exists('job_logs'):
    os.mkdir('job_logs')

if slurm:
    for snap in snaps:
        path_full=basepath+f'snap_{str(snap).zfill(3)}/'
        jobname=f'{runname}-recycdata_snap{str(snap).zfill(3)}'
        jobscriptfilepath=f'job_logs/submit-{jobname}.slurm'
        if os.path.exists(jobscriptfilepath):
            os.remove(jobscriptfilepath)
        with open(jobscriptfilepath,"w") as jobfile:
            jobfile.writelines(f"#!/bin/sh\n")
            jobfile.writelines(f"#SBATCH --job-name={jobname}\n")
            jobfile.writelines(f"#SBATCH --nodes=1\n")
            jobfile.writelines(f"#SBATCH --ntasks-per-node={1}\n")
            jobfile.writelines(f"#SBATCH --mem={total_mem*1}GB\n")
            jobfile.writelines(f"#SBATCH --time={wall_time}\n")
            jobfile.writelines(f"#SBATCH --output=job_logs/{jobname}.out\n")
            jobfile.writelines(f"#SBATCH --error=job_logs/{jobname}.err\n")
            if email:
                jobfile.writelines(f"#SBATCH --mail-type=ALL\n")
                jobfile.writelines(f"#SBATCH --mail-user={address}\n")
            jobfile.writelines(f" \n")
            jobfile.writelines(f"echo JOB START TIME\n")
            jobfile.writelines(f"date\n")
            jobfile.writelines(f"echo CPU DETAILS\n")
            jobfile.writelines(f"lscpu\n")
            jobfile.writelines(f"python {run_script} -mcut {mcut} -path {path_full} \n")
            jobfile.writelines(f"echo JOB END TIME\n")
            jobfile.writelines(f"date\n")
        jobfile.close()
        os.system(f"sbatch {jobscriptfilepath}")

else:
    for snap in snaps:
        path_full=basepath+f'snap_{str(snap).zfill(3)}/'
        os.system(f"python {run_script} -mcut {mcut} -path {path_full} ")

