# Submit script for acc data
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
run_script=chummdir+'Usage/GenData-AccData.py'

##### CUSTOMIZE #####
#job
slurm=False
email=True
wall_time="0-04:00:00"

#multiprocessing
num_processes_calc=16
num_processes_use=1
total_mem_perprocess=8#GB

#calc
detailed=1
compression=1
snaps=[27]
pre=1
post=1
gen_ad=0
sum_ad=1
hil_lo=-1
hil_hi=-1

if hil_lo==-1:
    test=False
else:
    test=True

####################

# Get run info for outputs
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
            jobfile.writelines(f"#SBATCH --ntasks-per-node={num_processes_use}\n")
            jobfile.writelines(f"#SBATCH --mem={total_mem_perprocess*num_processes_use}GB\n")
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
            jobfile.writelines(f"python {run_script} -detailed {detailed} -compression {compression} -np_use {num_processes_use} -np_calc {num_processes_calc} -snap {snap} -pre {pre} -post {post} -gen_ad {gen_ad} -sum_ad {sum_ad} -hil_lo {hil_lo} -hil_hi {hil_hi}\n")
            jobfile.writelines(f"echo JOB END TIME\n")
            jobfile.writelines(f"date\n")
        jobfile.close()
        os.system(f"sbatch {jobscriptfilepath}")

else:
    # Loop through desired calcs and submit
    for snap in snaps:
        os.system(f"python {run_script} -detailed {detailed} -compression {compression} -np_use {num_processes_use} -np_calc {num_processes_calc} -snap {snap} -pre {pre} -post {post} -gen_ad {gen_ad} -sum_ad {sum_ad} -hil_lo {hil_lo} -hil_hi {hil_hi}\n")
