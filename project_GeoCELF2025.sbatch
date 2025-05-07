#!/bin/bash

#SBATCH --partition main                ### Specify partition name where to run a job.
#SBATCH --time 0-30:00:00               ### Job running time limit. Make sure it is not exceeding the partition time li>
#SBATCH --job-name 'project_GeoCELF2025'             ### Name of the job. replace my_job with your desired job name
#SBATCH --output job-%J.out             ### Output log for running job - %J is the job number variable
#SBATCH --mail-user=shlomiasi1@gmail.com        ### User's email for sending job status
#SBATCH --mail-type=ALL                ### Conditions when to send the email. ALL,BEGIN,END,FAIL,REQUEU,NONE

#SBATCH --gpus=1                ### number of GPUs, ask for more than 1 only if you can parallelize your code for multitask>
##SBATCH --mem=64G              ### ammount of RAM memory
##SBATCH --cpus-per-task=6     ### number of CPU cores

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start you code below ####
##module load anaconda             ### load anaconda module (must present when working with conda environments)                 
source $(conda info --base)/etc/profile.d/conda.sh
conda activate project_GeoCELF2025           ### activating environment, environment must be configured before running the j>
conda install -y -c conda-forge gdal pytorch torchvision pandas matplotlib pyzipper
which python
python -c "import sys; print('PYTHONPATH:', sys.executable); from osgeo import gdal; print('GDAL VERSION:', gdal.__version__)"
python /home/shlomias/project_GeoCELF2025/src/main.py     ### running the python script