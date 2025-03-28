import json
import os
import sys
import time
import socket

import subprocess


result = subprocess.run(["rm","-r", "outputs", "run_dir"], capture_output=True, text=True)
print(result.stdout)


##first create input for ensemble launcher
##the config.json launches 1 task gpu tile
fname = os.getenv("PBS_NODEFILE")
with open(fname,"r") as f:
    nnodes = len(f.readlines())
ntasks = nnodes*12

config = {
            "poll_interval":1,
            "sys_info":{
                "name":"aurora",
                "ncores_per_node":104,
                "ngpus_per_node":12
            },
            "ensembles":{
                        "inference":{
                                "num_nodes":1,
                                "num_processes_per_node":1,
                                "num_gpus_per_process":1,
                                "launcher":"mpi",
                                "relation":"one-to-one",
                                "cmd_template":"python3 run_inference.py --nps " + f"{ntasks}" + " --pid {opts}",
                                "opts":list(range(ntasks)),
                                "run_dir":[f"./run_dir/task_{i}" for i in range(ntasks)]
                            }
                    }
        }
fname = "./el_config.json"
with open(fname,"w") as f:
    json.dump(config, f, indent=4)

##launch the tasks
if __name__ == "__main__":
    sys.path.append('./ensemble_launcher')
    from ensemble_launcher import ensemble_launcher
    el = ensemble_launcher("el_config.json",ncores_per_node=12)
    start_time = time.perf_counter()
    print(f'Launching node is {socket.gethostname()}')
    total_poll_time = el.run_tasks()
    end_time = time.perf_counter()
    total_run_time = end_time - start_time
    print(f"{total_run_time=}")