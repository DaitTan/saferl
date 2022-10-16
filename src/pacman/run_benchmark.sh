#!/usr/bin/env bash
new_dir=$(date +"%m_%d_%Y/%H_%M_%S")
mkdir -p "outputs/${new_dir}"

constants="-n 10250 -x 10000 -q"

declare -A EXP_ARGS
EXP_ARGS["q_agent_baseline"]="-p qlearningAgents.PacmanQAgent"
EXP_ARGS["approx_q_agent_baseline"]="-p qlearningAgents.ApproximateQAgent"
EXP_ARGS["approx_q_agent_with_extractor_baseline"]="-p qlearningAgents.ApproximateQAgent -a extractor=SimpleExtractor"
EXP_ARGS["lnn_q_agent_test"]="-p lnn_q_learning_agents.LNNQAgent"
EXP_ARGS["lnn_approx_q_agent_test"]="-p lnn_q_learning_agents.LNNApproximateQAgent"

declare -A ENV_ARGS
#ENV_ARGS["small_grid_1_agent"]="-l smallGrid -k 1"
#ENV_ARGS["small_grid_2_agent"]="-l smallGrid -k 2"
#ENV_ARGS["small_classic_2_agent"]="-l smallClassic -k 2"
#ENV_ARGS["small_classic_5_agent"]="-l smallClassic -k 5"
#ENV_ARGS["medium_grid_1_agent"]="-l mediumGrid -k 1"
#ENV_ARGS["medium_grid_2_agent"]="-l mediumGrid -k 2"
#ENV_ARGS["medium_classic_2_agent"]="-l mediumClassic -k 2"
#ENV_ARGS["medium_classic_3_agent"]="-l mediumClassic -k 3"
#ENV_ARGS["medium_classic_5_agent"]="-l mediumClassic -k 5"
ENV_ARGS["original_classic_2_agent"]="-l originalClassic -k 2"
ENV_ARGS["original_classic_4_agent"]="-l originalClassic -k 4"
ENV_ARGS["original_classic_6_agent"]="-l originalClassic -k 6"
ENV_ARGS["original_classic_10_agent"]="-l originalClassic -k 10"
ENV_ARGS["open_classic_1_agent"]="-l originalClassic -k 1"
ENV_ARGS["open_classic_3_agent"]="-l originalClassic -k 3"
#ENV_ARGS["open_classic_5_agent"]="-l originalClassic -k 5"
ENV_ARGS["tricky_classic_1_agent"]="-l trickyClassic -k 1"
ENV_ARGS["tricky_classic_3_agent"]="-l trickyClassic -k 3"
#ENV_ARGS["tricky_classic_5_agent"]="-l trickyClassic -k 5"

pushd src > /dev/null

for env in ${!ENV_ARGS[@]}; do
    env_args=${ENV_ARGS[${env}]}
    out_path="../outputs/${new_dir}/${env}"
    mkdir ${out_path}

    for exp in ${!EXP_ARGS[@]}; do
        exp_args=${EXP_ARGS[${exp}]}

        python pacman.py ${constants} ${env_args} ${exp_args} | tee ${out_path}/${exp}.log
    done
done

popd > /dev/null

python parse_results.py "outputs/${new_dir}"
