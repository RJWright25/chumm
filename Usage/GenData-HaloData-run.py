# Submit script for acc data
import warnings
warnings.filterwarnings("ignore")
import os
import sys

#finding system
if 'Users' in os.listdir('/'):
    chummdir='/Users/ruby/Documents/GitHub/CHUMM/'
else:
    chummdir='/home/rwright/'

sys.path.append(chummdir)
run_script=chummdir+'Usage/GenData-HaloData.py'

#job
slurm=False
email=True
total_mem=40#GB
wall_time="0-04:00:00"

#calc
num_processes=0
gen_bhd=1
gen_dhd=0
sum_dhd=0
com_dhd=0
add_hpd=0

####################

# Get run info for outputs
filename=sys.argv[0]
runcwd=os.getcwd()
runname=runcwd.split('-')[-1]
if not os.path.exists('job_logs'):
    os.mkdir('job_logs')


if slurm:
    jobname=(filename.split('-')[1]).split('_')[0]+'-'+runname+f'_pre{str(pre).zfill(2)}_post{str(post).zfill(2)}_snap{str(snap).zfill(3)}_np{str(num_processes).zfill(3)}'
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
        jobfile.writelines(f"python {run_script} -np {num_processes} -gen_bhd {gen_bhd} -gen_dhd {gen_dhd} -sum_dhd {sum_dhd} -com_dhd {com_dhd} -add_hpd {add_hpd}")
        jobfile.writelines(f"echo JOB END TIME\n")
        jobfile.writelines(f"date\n")
    jobfile.close()
    os.system(f"sbatch {jobscriptfilepath}")

else:
    os.system(f"python {run_script} -np {num_processes} -gen_bhd {gen_bhd} -gen_dhd {gen_dhd} -sum_dhd {sum_dhd} -com_dhd {com_dhd} -add_hpd {add_hpd}")
