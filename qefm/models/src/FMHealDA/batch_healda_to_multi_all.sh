#!/bin/bash

# Default SLURM parameters
DEFAULT_PARTITION="compute"
DEFAULT_TIME="01:00:00"
DEFAULT_CPUS=4
DEFAULT_MEM="60G"
DEFAULT_OUTPUT_DIR="/discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMHealDA/test/output"
DEFAULT_LOG_DIR="/discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMHealDA/test/logs"

# Parse command-line arguments
PARTITION="${DEFAULT_PARTITION}"
TIME="${DEFAULT_TIME}"
CPUS="${DEFAULT_CPUS}"
MEM="${DEFAULT_MEM}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
LOG_DIR="${DEFAULT_LOG_DIR}"
ADDITIONAL_SBATCH_ARGS=()

# Function to parse command-line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --partition)
                PARTITION="$2"
                shift 2
                ;;
            --time)
                TIME="$2"
                shift 2
                ;;
            --cpus)
                CPUS="$2"
                shift 2
                ;;
            --mem)
                MEM="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --log-dir)
                LOG_DIR="$2"
                shift 2
                ;;
            --sbatch)
                # Capture additional SBATCH arguments
                ADDITIONAL_SBATCH_ARGS+=("$2")
                shift 2
                ;;
            *)
                # Pass through arguments to Python script
                PYTHON_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

# Parse arguments
parse_args "$@"

# Create SBATCH script
SBATCH_SCRIPT=$(mktemp)

cat > "$SBATCH_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=healda_multi_mc
#SBATCH --output=${LOG_DIR}/healda_multi_mc_%A_%a.out
#SBATCH --error=${LOG_DIR}/healda_multi_mc_%A_%a.err
#SBATCH --time=${TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --partition=${PARTITION}
#SBATCH --array=0-62  # 31 days, 2 sources (UFS and MERRA-21c)

# Additional SBATCH arguments (if any)
$(printf "#SBATCH %s\n" "${ADDITIONAL_SBATCH_ARGS[@]}")

# Specify the base Python script location
SCRIPT="/discover/nobackup/projects/QEFM/qefm-core/qefm/models/src/FMHealDA/healda_to_multi_all_batch.py"

# Define the base output directory
BASE_OUTPUT_DIR="${OUTPUT_DIR}"

# Array of dates in December 2024
_DATES=(
     "2024-12-30" 
)

DATES=(
    "2024-12-01" "2024-12-02" "2024-12-03" "2024-12-04" "2024-12-05" "2024-12-06" "2024-12-07"
    "2024-12-08" "2024-12-09" "2024-12-10" "2024-12-11" "2024-12-12" "2024-12-13" "2024-12-14"
    "2024-12-15" "2024-12-16" "2024-12-17" "2024-12-18" "2024-12-19" "2024-12-20" "2024-12-21"
    "2024-12-22" "2024-12-23" "2024-12-24" "2024-12-25" "2024-12-26" "2024-12-27" "2024-12-28"
    "2024-12-29" "2024-12-30" "2024-12-31"
)

# Time steps within each day
TIME_STEPS=("00:00" "06:00" "12:00" "18:00")

# Calculate array index
DAY_INDEX=\$((SLURM_ARRAY_TASK_ID / 2))
SOURCE_INDEX=\$((SLURM_ARRAY_TASK_ID % 2))

# Select date and source
DATE=\${DATES[\$DAY_INDEX]}
if [ \$SOURCE_INDEX -eq 0 ]; then
    SOURCE_FLAG=""
    SOURCE_NAME="ufs"
else
    SOURCE_FLAG="--use-merra21c"
    SOURCE_NAME="merra21c"
fi

# Iterate through time steps
for TIME_STEP in "\${TIME_STEPS[@]}"; do
    # Construct full datetime
    DATETIME="\${DATE}T\${TIME_STEP}"
    
    # Create output directory with hierarchical structure
    OUTPUT_DIR="\${BASE_OUTPUT_DIR}/\${DATE}/\${SOURCE_NAME}/\${TIME_STEP}"
    mkdir -p "\${OUTPUT_DIR}"
    
    # Construct full command
    CMD="python \${SCRIPT} \
        --analysis-time \${DATETIME} \
        --output-dir \${OUTPUT_DIR} \
        --forecast-hours 120 \
        --obs-window-start -3 \
        --obs-window-end 3 \
        --device cpu \
        \${SOURCE_FLAG} \
        ${PYTHON_ARGS[*]}"
    
    # Print and execute command
    echo "Running: \$CMD"
    eval \$CMD
done
EOF

# Submit the job
sbatch "$SBATCH_SCRIPT"

# Clean up temporary script
#rm "$SBATCH_SCRIPT"
