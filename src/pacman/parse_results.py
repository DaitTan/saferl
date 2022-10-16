#!/usr/bin/env python3
import re
import sys
import json

from pathlib import Path
from collections import defaultdict

def get_num_from_str(orig_str, pattern, cast):
    return cast(orig_str.split(pattern)[1].splitlines()[0])

def main(dname: str):
    results_dir = Path(dname)
    train_results = defaultdict(dict)
    test_results = defaultdict(dict)

    for log_file in results_dir.glob('*/*.log'):
        with open(log_file) as f:
            log = f.read()
        
        exp_name = log_file.stem
        env_name = log_file.parent.stem

        train_results[env_name][exp_name] = []

        train_test_split = log.split('Training Done')
        train_results_str = train_test_split[0]
        test_results_str = train_test_split[1]

        for epoch in train_results_str.split('Reinforcement Learning Status:')[1:]:
            train_results[env_name][exp_name].append({
                'episode': get_num_from_str(epoch, "Episode: ", int),
                'avg_reward': get_num_from_str(epoch, "Avg train reward: ", float),
                'avg_last_100_reward': get_num_from_str(epoch, "Avg last 100 reward: ", float),
                'time': get_num_from_str(epoch, "Avg last 100 reward: ", float)})

        test_section = re.split("Scores:\W+", test_results_str)[1].splitlines()
        test_results[env_name][exp_name] = {
            'test_scores' : list(map(float, test_section[0].split(", "))),
            'test_win': list(map(
                lambda x: True if x == 'Win' else False,
                test_section[-1].strip("Record: ").split(", ")))}
    
    with open(results_dir / 'train_results.json', 'w') as f:
        json.dump(train_results, f, indent=4)

    with open(results_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=4)

if __name__ == '__main__':
    main(sys.argv[1])