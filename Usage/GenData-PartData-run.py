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

# GenData-PartData-run.py - Generation script to run particle data functions. Edit runtime parameters as required. 
# Requirements: base halo data is generated. 

# Preamble
import warnings
warnings.filterwarnings("ignore")
import os
import sys

####################################################################################################
########################################## RUNTIME PARS ############################################

# Job details
num_processes=1 # number of processed (from multiprocessing) to use (only for gen_particle_history_serial)
slurm=False # whether or not to use slurm submit
email=False # email results y/n (if slurm)
address='21486778@student.uwa.edu.au' # email address (if slurm)
wall_time="0-04:00:00" # job time limit (if slurm)
total_mem_perprocess=8 # total memory required per process (if slurm)

# Algorithm Details
gen_bph=1 # generate base particle histories (host for each particle, run in parallel)
sum_bph=1 # sum base particle histories for integrated processing history of each particle

####################################################################################################
####################################################################################################

# Run Script
if 'Users' in os.listdir('/'):
    chummdir='/Users/ruby/Documents/GitHub/CHUMM/'
else:
    chummdir='/home/rwright/CHUMM/'

run_script=chummdir+'Usage/GenData-PartData.py'

# Submit/ run
filename=sys.argv[0]
runcwd=os.getcwd()
runname=runcwd.split('-')[-1]
if not os.path.exists('job_logs'):
    os.mkdir('job_logs')

if slurm:
    jobname=f'{runname}-partdata'
    jobscriptfilepath=f'job_logs/submit-{jobname}.slurm'
    if os.path.exists(jobscriptfilepath):
        os.remove(jobscriptfilepath)
    with open(jobscriptfilepath,"w") as jobfile:
        jobfile.writelines(f"#!/bin/sh\n")
        jobfile.writelines(f"#SBATCH --job-name={jobname}\n")
        jobfile.writelines(f"#SBATCH --nodes=1\n")
        jobfile.writelines(f"#SBATCH --ntasks-per-node={num_processes}\n")
        jobfile.writelines(f"#SBATCH --mem={total_mem_perprocess*num_processes}GB\n")
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
        jobfile.writelines(f"python {run_script} -np {num_processes} -gen_bph {gen_bph} -sum_bph {sum_bph}\n")
        jobfile.writelines(f"echo JOB END TIME\n")
        jobfile.writelines(f"date\n")
    jobfile.close()
    os.system(f"sbatch {jobscriptfilepath}")

else:
    os.system(f"python {run_script} -np {num_processes} -gen_bph {gen_bph} - sum_bph {sum_bph}\n")

