dim=3
dataset_id=0

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --seed)
            seed="$2"
            shift
            shift
            ;;
        --dataset_id)
            dataset_id="$2"
            shift
            shift
            ;;
        --bench_name)
            bench_name="$2"
            shift
            shift
            ;;
        --n_workers)
            n_workers="$2"
            shift
            shift
            ;;
        --dim)
            n_workers="$2"
            shift
            shift
            ;;
        *)
            shift
            ;;
    esac
done

rm -r neps-log  # Here must be adapted to the exact log directory name.

for num in `seq 1 ${n_workers}`; do
    python -m examples.neps --seed $seed --dataset_id $dataset_id --bench_name $bench_name --n_workers $n_workers --worker_index $((num - 1)) &
    pids[${num}]=$!
    echo "Start Proc. $num"
done

for pid in ${pids[*]}; do
    wait $pid
    echo "Finish Proc. $pid"
done
