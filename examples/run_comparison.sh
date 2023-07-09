run_command() {
    cmd=${1}

    echo `date '+%y/%m/%d %H:%M:%S'`
    echo $cmd
    $cmd
}

cfg="--dataset_id 0 --bench_name hpolib --n_workers 4"

for seed in `seq 0 9`
do
    cmd="python -m examples.bohb --seed ${seed} ${cfg}"
    run_command "${cmd}"

    cmd="python -m examples.dehb --seed ${seed} ${cfg}"
    run_command "${cmd}"

    cmd="./examples/neps.sh --seed ${seed} ${cfg}"
    run_command "${cmd}"

    # You need another environment for SMAC3.
    # cmd="python -m examples.smac --seed ${seed} ${cfg}"
    # run_command "${cmd}"
done
