n_workers=4

for num in `seq 1 ${n_workers}`; do
    python -m optimizers.neps --seed 0 --dataset_id 0 --bench_name hpolib --n_workers $n_workers &
    pids[${num}]=$!
    echo "Start Proc. $num"
done

for pid in ${pids[*]}; do
    wait $pid
    echo "Finish Proc. $pid"
done
