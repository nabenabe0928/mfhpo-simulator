while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --n_workers)
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
    python -m examples.minimal.neps --n_workers $n_workers --worker_index $((num - 1)) &
    pids[${num}]=$!
    echo "Start Proc. $num"
done

for pid in ${pids[*]}; do
    wait $pid
    echo "Finish Proc. $pid"
done
