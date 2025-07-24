import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import data.multimodal_dataset as data_loader
import pandas as pd
import models.model as module_model
from sklearn.metrics import f1_score

BATCH_SIZE = 500

class Test:
    def __init__(self, model_type, path2test, path2results="results/unknown", modality_type="all", actions_complexity=0):
        self.model_type = model_type
        self.path2test = path2test
        self.path2results = path2results
        self.modality_type = modality_type
        self.actions_complexity = actions_complexity

    def select_model(self, detection_type, config):
        return module_model.MultiModalityMerger(method=self.model_type, detection=detection_type, config=config)

    def compute_f1_score(self, y_true, y_pred):
        correct_labels_flat = torch.cat(y_true).cpu().numpy()
        predicted_labels_flat = torch.cat(y_pred).cpu().numpy()
        return f1_score(correct_labels_flat, predicted_labels_flat, average='macro')

    def test_aod(self, batch_size=BATCH_SIZE):
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        torch.manual_seed(42)
        test_dataset = data_loader.MultiModalDataset(path2dataset=self.path2test,
                                                     modality_type=self.modality_type,
                                                     detection_type="aod",
                                                     actions_complexity=self.actions_complexity)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Initialize model
        model = self.select_model(detection_type="aod", config=test_dataset.config)
        model_path = os.path.join(self.path2results, f"model_weights_aod.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        results_data = []
        correct_label = []
        pred_label = []
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, targets, cmd_idx in test_loader:
                outputs = model(inputs)

                pred_label.append(torch.argmax(outputs[0].flatten(1), dim=1))
                correct_label.append(torch.argmax(targets[0].flatten(1), dim=1))

                aod_new_score = self.get_aod_metrics_top1(outputs[0], targets[0])
                command = self.get_command(test_dataset, cmd_idx)
                scene = self.get_scene(test_dataset, cmd_idx)
                od_new_score = self.get_od_metrics(outputs[1], targets[1], scene)
                ad_new_score = self.get_ad_metrics(outputs[2], targets[2])

                data_point = {
                    "scene_id": scene.id,
                    "sctructure_type": command.type_id,
                    "action_type": command.action_id,
                    "aod_od_score": aod_new_score[1],
                    "aod_ad_score": aod_new_score[2],
                    "od_target_score": od_new_score[0],
                    "od_spec_score": od_new_score[2],
                    "od_feasible_score": od_new_score[1],
                    "aod_target_score": aod_new_score[0],
                    "ad_target_score": ad_new_score,
                    "max_probability": torch.max(outputs[0].flatten()).item(),
                    "target_action": aod_new_score[4][0],
                    "pred_action": aod_new_score[4][1],
                    "target_object": aod_new_score[4][2],

                    "pred_object":aod_new_score[4][3],
                }
                results_data.append(data_point)
                correct += aod_new_score[0]
                total += 1

        f1 = self.compute_f1_score(correct_label, pred_label)
        accuracy = correct / total * 100
        print(f"Test AOD - Top-1 Accuracy: {accuracy:.2f}%, F1 Score: {f1:.3f}")
        self.save_results(results_data, "test_data_aod")

    def test_ad(self, batch_size=BATCH_SIZE):
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        torch.manual_seed(42)
        test_dataset = data_loader.MultiModalDataset(path2dataset=self.path2test,
                                                     modality_type=self.modality_type,
                                                     detection_type="ad",
                                                     actions_complexity=self.actions_complexity)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Initialize model
        model = self.select_model(detection_type="ad", config=test_dataset.config)
        model_path = os.path.join(self.path2results, f"model_weights_ad.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        results_data = []
        correct_label = []
        pred_label = []
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, targets, cmd_idx in test_loader:
                outputs = model(inputs)
                scene = self.get_scene(test_dataset, cmd_idx)
                command = self.get_command(test_dataset, cmd_idx)
                new_score = self.get_ad_metrics(outputs, targets)

                aod_ad_target = torch.tensor(command.preprocess_target_actions(), dtype=torch.long).squeeze()
                aod_ad_target_score = self.get_ad_metrics(outputs, aod_ad_target)
                pred_label.append(torch.argmax(outputs.flatten(1), dim=1))
                correct_label.append(targets)

                data_point = {
                    "scene_id": scene.id,
                    "sctructure_type": command.type_id,
                    "action_type": command.action_id,
                    "ad_target_score": new_score,
                    "aod_ad_target_score" : aod_ad_target_score,
                }
                results_data.append(data_point)
                correct += new_score
                total += 1

        f1 = self.compute_f1_score(correct_label, pred_label)
        accuracy = correct / total * 100
        print(f"Test AD - Top-1 Object Accuracy: {accuracy:.2f}%, F1 Score: {f1:.3f}")
        self.save_results(results_data, "test_data_ad")

    def test_od(self, batch_size=BATCH_SIZE):
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        torch.manual_seed(42)
        test_dataset = data_loader.MultiModalDataset(path2dataset=self.path2test,
                                                     modality_type=self.modality_type,
                                                     detection_type="od",
                                                     actions_complexity=self.actions_complexity)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Initialize model
        model = self.select_model(detection_type="od", config=test_dataset.config)
        model_path = os.path.join(self.path2results, f"model_weights_od.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        results_data = []
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, targets, cmd_idx in test_loader:
                outputs = model(inputs)
                command = self.get_command(test_dataset, cmd_idx)
                scene = self.get_scene(test_dataset, cmd_idx)
                new_score = self.get_od_metrics(outputs, targets, scene)

                data_point = {
                    "scene_id": scene.id,
                    "sctructure_type": command.type_id,
                    "action_type": command.action_id,
                    "od_target_score": new_score[0],
                    "od_spec_score": new_score[2],
                    "od_feasible_score": new_score[1],
                }
                results_data.append(data_point)
                correct += new_score[0]
                total += 1

        accuracy = correct / total * 100
        print(f"Test OD - Target Object Accuracy: {accuracy:.2f}%")
        self.save_results(results_data, "od_test")

    def get_ad_metrics(self, prediction, target):
        prediction = prediction.squeeze()
        pred_object_idx = (prediction == torch.max(prediction)).nonzero(as_tuple=True)[0]
        target_ad_score = 0
        random_idx = torch.randint(low=0, high=pred_object_idx.size(0), size=(1,)).item()
        solo_pred = pred_object_idx[random_idx]
        if solo_pred == target:
            target_ad_score = 1
        return target_ad_score

    def get_aod_metrics_top1(self, prediction, target):
        batch_size = prediction.shape[0]
        height = prediction.shape[1]
        width = prediction.shape[2]

        # After flattening
        prediction_flattened = prediction.flatten()
        target_flattened = target.flatten()

        # Get max indices
        max_num = torch.max(prediction_flattened)
        prediction_flattened_idx = torch.where(prediction_flattened == max_num)[0]
        if prediction_flattened_idx.shape[0] > 1:
            random_idx = torch.randint(low=0, high=prediction_flattened_idx.size(0), size=(1,)).item()
            prediction_flattened_idx = prediction_flattened_idx[random_idx]
        target_flattened_idx = torch.where(target_flattened == torch.max(target_flattened))[0]

        # Recover original 2D position (row, col)
        prediction_row = prediction_flattened_idx // width
        prediction_col = prediction_flattened_idx % width


        target_row = target_flattened_idx // width
        target_col = target_flattened_idx % width

        if prediction_flattened_idx.shape == target_flattened_idx.shape:
            correct_predictions = torch.sum(prediction_flattened_idx == target_flattened_idx).item()
        else:
            correct_predictions = 0

        state_art_comparison = 0
        if target_col == prediction_col:
            state_art_comparison = 1
        else:
            state_art_comparison = correct_predictions * 100

        ad_score = 0
        od_score = 0
        if prediction_col == target_col:
            od_score = 1
        if prediction_row == target_row:
            ad_score = 1
        info = [target_row.item(),prediction_row.item(),target_col.item(),prediction_col.item()]
        return correct_predictions,od_score,ad_score,None,info

    def save_results(self, data, filename):
        df = pd.DataFrame(data)
        save_path = os.path.join(self.path2results, f"{filename}.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved test results to {save_path}")

    def get_command(self, dataset, idx):
        return dataset.data_commands[idx]

    def get_scene(self, dataset, idx):
        return dataset.data_scenes[dataset.sceneid2idx[dataset.data_commands[idx].scene_id]]

    def get_od_metrics(self, pred_object, target_object, scene):
        target_obj_score = 0
        feasible_score = 0
        specification_score = 0

        target_object_idx = target_object
        pred_object = pred_object.squeeze()
        pred_object_idx = (pred_object == torch.max(pred_object)).nonzero(as_tuple=True)[0]
        possible_objects = scene.get_possible_targets_idx(target_object)

        random_idx = torch.randint(low=0, high=pred_object_idx.size(0), size=(1,)).item()
        solo_pred = pred_object_idx[random_idx]
        if solo_pred == target_object_idx:
            target_obj_score = 1

        if all(elem in possible_objects for elem in pred_object_idx):
            feasible_score = 1

        if target_object_idx in pred_object_idx:
            if all(elem in possible_objects for elem in pred_object_idx):
                specification_score = 1 / len(pred_object_idx)
       # if specification_score != feasible_score:
        #    print(target_obj_score, feasible_score, specification_score)
        return [target_obj_score, feasible_score, specification_score]

if __name__ == '__main__':
    # Example usage: Replace with your actual paths and configurations
    path2train_example = "path/to/your/train_data.pkl"
    path2val_example = "path/to/your/val_data.pkl"
    path2test_example = "path/to/your/test_data.pkl"
    results_dir_example = "results/my_experiment"
    model_type_example = "concat"  # Or "late_fusion", "early_fusion"
    modality_type_example = "all"  # Or "rgb", "depth", "audio"
    actions_complexity_example = 0 # Or your desired complexity level

    # Create results directory if it doesn't exist
    os.makedirs(results_dir_example, exist_ok=True)

    # Train the models
    trainer = Train(model_type=model_type_example,
                    path2train=path2train_example,
                    path2val=path2val_example,
                    path2results=results_dir_example,
                    modality_type=modality_type_example,
                    actions_complexity=actions_complexity_example)

    print("Training AOD model...")
    trainer.train_aod()

    print("\nTraining AD model...")
    trainer.train_ad()

    print("\nTraining OD model...")
    trainer.train_od()

    # Test the trained models
    tester = Test(model_type=model_type_example,
                   path2test=path2test_example,
                   path2results=results_dir_example,
                   modality_type=modality_type_example,
                   actions_complexity=actions_complexity_example)

    print("\nTesting AOD model...")
    tester.test_aod()

    print("\nTesting AD model...")
    tester.test_ad()

    print("\nTesting OD model...")
    tester.test_od()