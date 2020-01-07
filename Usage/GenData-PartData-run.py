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
                                                                                                                                                                  
                                                                                                                                                                  
# GenData-PartData.py - Script to run the particle histories algorithm. 
# Author: RUBY WRIGHT 

# PREAMBLE
import warnings
warnings.filterwarnings("ignore")
import os
import sys

#finding system
if 'Users' in os.listdir('/'):
    chummdir='/Users/ruby/Documents/GitHub/CHUMM/'
else:
    chummdir='/home/rwright/CHUMM/'

sys.path.append(chummdir)
run_script=chummdir+'Usage/GenData-PartData.py'

##### CUSTOMIZE #####
#job
slurm=False
email=True
wall_time="0-04:00:00"

#multiprocessing
num_processes=8
total_mem_perprocess=8#GB

#run
gen_bph=1
sum_bph=1

# Get run info for outputs
filename=sys.argv[0]
runcwd=os.getcwd()
runname=runcwd.split('-')[-1]
if not os.path.exists('job_logs'):
    os.mkdir('job_logs')

if slurm:
    for snap in snaps:
        if test:
            jobname=f'{runname}-partdata'
            jobname=(filename.split('-')[1]).split('_')[0]+'-'+runname+f'_pre{str(pre).zfill(2)}_post{str(post).zfill(2)}_snap{str(snap).zfill(3)}_np{str(num_processes_calc).zfill(3)}'
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
                jobfile.writelines(f"#SBATCH --mail-user=21486778@student.uwa.edu.au\n")
            jobfile.writelines(f" \n")
            jobfile.writelines(f"echo JOB START TIME\n")
            jobfile.writelines(f"date\n")
            jobfile.writelines(f"echo CPU DETAILS\n")
            jobfile.writelines(f"lscpu\n")
            jobfile.writelines(f"python {run_script} -np {num_processes} -gen_bph {gen_bph} - sum_bph {sum_bph}\n")
            jobfile.writelines(f"echo JOB END TIME\n")
            jobfile.writelines(f"date\n")
        jobfile.close()
        os.system(f"sbatch {jobscriptfilepath}")

else:
    # Loop through desired calcs and submit
    for snap in snaps:
        os.system(f"python {run_script} -np {num_processes} -gen_bph {gen_bph} - sum_bph {sum_bph}\n")

