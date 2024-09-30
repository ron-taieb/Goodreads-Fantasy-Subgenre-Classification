
#!/bin/bash

# Path to the Feather file
input_feather="/path/to/your/fantasy_reviews.feather"
output_dir="/path/to/your/processed_subsets"
mkdir -p $output_dir

# Define batch size
batch_size=100000
total_rows=3424641
num_batches=$((total_rows / batch_size))
remainder=$((total_rows % batch_size))

# Load virtual environment
source /path/to/your/venv/bin/activate

for i in $(seq 0 $num_batches); do
    start_index=$((i * batch_size))
    if [ $i -eq $num_batches ]; then
        end_index=$((start_index + remainder))
    else
        end_index=$((start_index + batch_size))
    fi
    output_feather="$output_dir/fantasy_reviews_subset_$i.feather"
    log_file="/path/to/your/logs/job_$i.log"
    err_file="/path/to/your/logs/job_$i.err"

    sbatch <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH --time=3:00:00
#SBATCH --mem=8G
#SBATCH -o $log_file
#SBATCH -e $err_file

source /path/to/your/venv/bin/activate
python /path/to/your/process_reviews_subset.py $start_index $end_index $input_feather $output_feather
EOF

done
