import argparse
import glob
import os
import train
import test  # Assuming you have a test module
from time import time
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Run the script for training or testing a model.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Specify whether to run training or testing.")
    parser.add_argument("--model_type", type=str, help="Specify the model type (e.g., 'a', 'b', 'c').")
    parser.add_argument("extra", nargs="?", default="", help="Extra argument after mode (used for test results path).")

    args = parser.parse_args()

    # If no model_type argument is passed, use the default list ['a', 'b', 'c', 'd']
    model_types = [args.model_type] if args.model_type else ["nn-1", 'mix', 'max',"mul","sum"]

    # Generate results path
    if args.mode == "train":
        main_results_path = f"results/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    else:  # mode == "test"
        if not args.extra:
            raise ValueError("Test mode requires an extra argument for the results path.")
        main_results_path = f"results/{args.extra}"

    modality_types = ["all"]
    sizes = ["small"]
    scene_complexities = [3]
    actions_complexities = ["x"]
    time_start = time()
    mainpath2datasets = "object_detection"
    for size in sizes:
        for modality_type in modality_types:
            for scene_complexity in scene_complexities:
                for action_complexity in actions_complexities:
                    for model_type in model_types:
                        
                        print(f"{args.mode.capitalize()}: "
                              f"\n model type merger:{model_type} "
                              f"\n dataset size: {size} "
                              f"\n modality type: {modality_type} "
                              f"\n dataset type: scene complexity - {scene_complexity}"
                              f", action complexity - {action_complexity}")
                        results_path_tmp = (f"results/{mainpath2datasets}/{size}_m_{model_type}"
                                            f"_mod_{modality_type}_"
                                            f"sc_{scene_complexity}_"
                                            f"ac_{action_complexity}")
                        path2dataset = f"data/{mainpath2datasets}/{size}_sc_{scene_complexity}_{action_complexity}"
                        if args.mode == "train":
                            training_module  = train.Train(model_type=model_type,
                                        path2train=f"{path2dataset}/train",
                                        path2val=f"{path2dataset}/val",
                                        modality_type=modality_type,
                                        path2results=results_path_tmp,
                                        actions_complexity= action_complexity)
                            training_module.train_aod()
                            #training_module.train_ad()
                            #training_module.train_od()
    
                        elif args.mode == "test":
                            testing_module  = test.Test(model_type=model_type,
                                      path2test=f"{path2dataset}/test",
                                      modality_type=modality_type,
                                      path2results=results_path_tmp,
                                      actions_complexity= action_complexity)
                            #testing_module.test_aod()
                            #testing_module.test_ad()
                            testing_module.test_od()

    time_end = time()
    print(f"{args.mode.capitalize()} completed in {time_end - time_start:.2f} seconds.")


if __name__ == '__main__':
    main()

