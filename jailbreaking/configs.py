def config_loader(opt_mode, mode, beta_str):

    assert opt_mode in ['trad'], "Mode Not Implemented Yet"

    if opt_mode == "trad":

        assert mode in ['baseline'], "Current Mode not Supported"

        ### Momentum: Baseline

        if mode == 'baseline':

            MAX_SIM_DIR_ID = {
                "Qwen/Qwen2.5-3B-Instruct": {"file_suffix": "max_sim_26_post"},
                "Qwen/Qwen2.5-7B-Instruct": {"file_suffix": "max_sim_22_post"},
                "Qwen/Qwen2.5-14B-Instruct": {"file_suffix": "max_sim_36_post"},
                "Qwen/Qwen2.5-32B-Instruct": {"file_suffix": "max_sim_50_post"},
                "meta-llama/Llama-3.1-8B-Instruct": {"file_suffix": "max_sim_21_post"},
                "meta-llama/Llama-3.2-3B-Instruct": {"file_suffix": "max_sim_18_post"},
                "google/gemma-2-9b-it": {"file_suffix": "max_sim_31_post",
                                         "relocate_file_suffix": "max_sim_24_post",
                                         "relocate_mode": False},
                "google/gemma-2-27b-it": {"file_suffix": "max_sim_32_post",
                                         "relocate_file_suffix": "max_sim_15_post",
                                         "relocate_mode": False},
                "Unispac/Gemma-2-9B-IT-With-Deeper-Safety-Alignment": {"file_suffix": "max_sim_28_post",
                                         "relocate_file_suffix": "max_sim_17_post",
                                         "relocate_mode": False},
            }

    return MAX_SIM_DIR_ID



