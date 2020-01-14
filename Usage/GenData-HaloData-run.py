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
                                                                                                                                                                  
                                                                                                                                                                  
# GenData-HaloData-run.py - Script to run the generate halo data script.
# Author: RUBY WRIGHT 

# Preamble
import warnings
warnings.filterwarnings("ignore")
import os
import sys

# Run script
run_script='GenData-HaloData.py'

# Job details
slurm=True
email=True
wall_time="0-04:00:00"
num_processes=1
total_mem=25#GB

# Algorithm details
gen_bhd=1
gen_dhd=1
sum_dhd=1
com_dhd=1

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
            jobfile.writelines(f"#SBATCH --mail-user=21486778@student.uwa.edu.au\n")
        jobfile.writelines(f" \n")
        jobfile.writelines(f"echo JOB START TIME\n")
        jobfile.writelines(f"date\n")
        jobfile.writelines(f"echo CPU DETAILS\n")
        jobfile.writelines(f"lscpu\n")
        jobfile.writelines(f"python {run_script} -np {num_processes} -gen_bhd {gen_bhd} -gen_dhd {gen_dhd} -sum_dhd {sum_dhd} -com_dhd {com_dhd} \n")
        jobfile.writelines(f"echo JOB END TIME\n")
        jobfile.writelines(f"date\n")
    jobfile.close()
    os.system(f"sbatch {jobscriptfilepath}")

else:
    os.system(f"python {run_script} -np {num_processes} -gen_bhd {gen_bhd} -gen_dhd {gen_dhd} -sum_dhd {sum_dhd} -com_dhd {com_dhd}  \n")
