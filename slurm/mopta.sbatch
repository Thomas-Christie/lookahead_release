#!/bin/bash
#SBATCH --job-name=mopta08
#SBATCH --chdir=/home/tu/tu_tu/tu_iiitc01/lookahead_release
#SBATCH --partition=cpu-single
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=48:00:00
#SBATCH --array=0-19
#SBATCH -o /home/tu/tu_tu/tu_iiitc01/lookahead_release/logs/mopta08_%j_%x_%a.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Load Singularity module
source /etc/profile.d/modules.sh
module load system/singularity/3.11.3

# Pick the seed for this array task from a fixed list
SEEDS=(
    3757552657
    673228719
    3241444873
    3685993406
    1216546553
    2078861726
    2471122328
    23012616
    3031610183
    2976135721
    556745865
    4118645239
    2090914255
    1802160804
    3957305396
    2416193258
    2322685169
    3860999861
    3521360814
    2710367621
)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Start bencher as a background sidecar
singularity run images/bencher.sif &
BENCHER_PID=$!
trap "kill $BENCHER_PID" EXIT TERM

# Wait for Bencher services to be ready
echo "Starting up bencher server"
for port in 50051 50053 50054 50055 50056 50057 50058 50059 50060; do
    until nc -z 127.0.0.1 $port; do sleep 1; done
done
echo "Bencher server started"

echo "Starting job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Using seed: ${SEED}"
echo "Running on node: $(hostname)"

source .venv/bin/activate
python scripts/experiment_runner.py --benchmark mopta08 --seed ${SEED}