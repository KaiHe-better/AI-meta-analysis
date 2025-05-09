#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medical_paper



root_dir="/raid/hpc/hekai/WorkShop/My_project/Medical_paper_agent/agent3/data/OCR_results"
output_dir="/raid/hpc/hekai/WorkShop/My_project/Medical_paper_agent/agent3/data/OCR_extracted"
 

PYTHON_SCRIPT="step2/preprocess/1_LLM_extract.py"


for file in "$root_dir"/*; do
    filename=$(basename "$file")
    python "$PYTHON_SCRIPT" --root_dir "$root_dir" --output_dir "$output_dir" --need_list "$filename" &
done

wait
echo "All tasks completed!"





###    nohup bash 3_step2_exaction.sh > logfile_3_step2_exaction.log 2>&1 &


