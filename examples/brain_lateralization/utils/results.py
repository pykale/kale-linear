import os

import numpy as np
import pandas as pd


def fetch_weights(base_dir, group, lambda_, dataset, sessions, test_size="00", seed_=2023):
    sub_dir = os.path.join(base_dir, "lambda%s" % lambda_)
    if not os.path.exists(sub_dir):
        return None
    if lambda_ == "0_group_mix":
        lambda_ = 0
        group = "mix"

    weight = []
    num_repeat = 5
    halfs = [0, 1]
    for session_i in sessions:
        for half_i in halfs:
            for i_split in range(num_repeat):
                for seed in range(52):
                    model_file_name = "%s_L%s_test_size%s_%s%s_%s_group_%s_%s.pt" % (
                        dataset,
                        lambda_,
                        test_size,
                        session_i,
                        i_split,
                        half_i,
                        group,
                        seed_ - seed,
                    )
                    # if os.path.exists(os.path.join(sub_dir, model_file_name)):
                    #     weight.append(get_coef(model_file_name, sub_dir).reshape((1, -1)))
    if not weight:
        return None
    return np.concatenate(weight, axis=0)


def get_coef(file_name, file_dir):
    import torch

    file_path = os.path.join(file_dir, file_name)
    model = torch.load(file_path)
    return model.theta


def save_results(res_dict, out_filename, output_dir, mix_group=False):
    res_df = pd.DataFrame.from_dict(res_dict)

    if mix_group:
        out_filename = out_filename + "_mix_group"
    out_file = os.path.join(output_dir, "%s.csv" % out_filename)
    res_df.to_csv(out_file, index=False)
