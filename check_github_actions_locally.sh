run_check() {
    proc_name=${1}
    cmd=${2}
    echo "### Start $proc_name ###"
    echo $cmd
    $cmd
    echo "### Finish $proc_name ###"
    printf "\n\n"
}

target="benchmark_simulator"  # target must be modified accordingly
export MFHPO_SIMULATOR_TEST="True"
run_check "pre-commit" "pre-commit run --all-files"
run_check "pytest" "python -m pytest -W ignore --cov-report term-missing --cov=$target --cov-config=.coveragerc"
run_check "black" "black tests/ $target/ examples"
rm .coverage.*
