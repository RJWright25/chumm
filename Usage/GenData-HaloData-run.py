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

# GenData-HaloData-run.py - Generation script to run halo data functions. Edit runtime parameters as required. 
# Requirements: GenData-HaloData.py is edited as required. 

# Preamble
import warnings
warnings.filterwarnings("ignore")
import os
import sys

####################################################################################################
########################################## RUNTIME PARS ############################################

# Job details
num_processes=1 # number of processed (from multiprocessing) to use (only for gen_detailed_halo_data)
slurm=False # whether or not to use slurm submit
email=False # email results y/n (if slurm)
address='21486778@student.uwa.edu.au' # email address (if slurm)
wall_time="0-04:00:00" # job time limit (if slurm)
total_mem=8 # total memory required (if slurm)

# Algorithm details
gen_bhd=1 # generate base halo data (with TreeFrog)
gen_dhd=1 # add all VR fields and generate B3 halo data for each snap
sum_dhd=1 # collate all B3 halo data
com_dhd=1 # compress B3 halo data with specific fields to get B4 halo data
add_progen=0 # compress B3 halo data with specific fields to get B4 halo data

####################################################################################################
####################################################################################################

# Run script
sys.path.append('/home/rwright/software/read_eagle/build/lib/python3.7/site-packages/')
run_script='GenData-HaloData.py'

# Submit/ run
filename=sys.argv[0]
runcwd=os.getcwd()
runname=runcwd.split('-')[-1]
if not os.path.exists('job_logs'):
    os.mkdir('job_logs')

if slurm:
    jobname=f'{runname}-halodata'
    jobscriptfilepath=f'job_logs/submit-{jobname}.slurm'
    if os.path.exists(jobscriptfilepath):
        os.remove(jobscriptfilepath)
    with open(jobscriptfilepath,"w") as jobfile:
        jobfile.writelines(f"#!/bin/sh\n")
        jobfile.writelines(f"#SBATCH --job-name={jobname}\n")
        jobfile.writelines(f"#SBATCH --nodes=1\n")
        jobfile.writelines(f"#SBATCH --ntasks-per-node={num_processes}\n")
        jobfile.writelines(f"#SBATCH --mem={total_mem}GB\n")
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
        jobfile.writelines(f"python {run_script} -np {num_processes} -gen_bhd {gen_bhd} -gen_dhd {gen_dhd} -sum_dhd {sum_dhd} -com_dhd {com_dhd} -add_progen {add_progen}\n")
        jobfile.writelines(f"echo JOB END TIME\n")
        jobfile.writelines(f"date\n")
    jobfile.close()
    os.system(f"sbatch {jobscriptfilepath}")

else:
    os.system(f"python {run_script} -np {num_processes} -gen_bhd {gen_bhd} -gen_dhd {gen_dhd} -sum_dhd {sum_dhd} -com_dhd {com_dhd} -add_progen {add_progen} \n")
