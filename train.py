import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import data.multimodal_dataset as data_loader
import matplotlib.pyplot as plt
import csv
import pandas as pd
import models.model as module_model
from sklearn.metrics import f1_score


VAL_EACH_EPOCH = 2
BATCH_SIZE = 500
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
OD_LOSS_ROUND = 3

class Train:
    def __init__(self, model_type, path2train, path2val, path2results="results/unknown", modality_type="all",actions_complexity=0):
        self.model_type = model_type
        self.path2train = path2train
        self.path2val = path2val
        self.path2results = path2results
        self.modality_type = modality_type
        self.actions_complexity = actions_complexity

    def select_model(self, detection_type,config):
        return module_model.MultiModalityMerger(method=self.model_type, detection=detection_type,config=config)

    def compute_f1_score(self, y_true, y_pred):
        correct_labels_flat = torch.cat(y_true).cpu().numpy()
        predicted_labels_flat = torch.cat(y_pred).cpu().numpy()

        return f1_score(correct_labels_flat, predicted_labels_flat, average='macro')
    def train_aod(self, batch_size=BATCH_SIZE,
                  learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS, patience=5):
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        torch.manual_seed(42)
        train_dataset = data_loader.MultiModalDataset(path2dataset=self.path2train,
                                                      modality_type=self.modality_type, 
                                                      detection_type="aod",
                                                      actions_complexity=self.actions_complexity)
        val_dataset = data_loader.MultiModalDataset(path2dataset=self.path2val,
                                                     modality_type=self.modality_type, 
                                                    detection_type="aod",
                                                    actions_complexity=self.actions_complexity)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        
        


        # Initialize model, loss function, and optimizer
        model = self.select_model(detection_type="aod",config = train_dataset.config)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loss_values, val_accuracies = [], []

        # Create a directory to save the figures
        figs_dir = self.path2results + "/figs"
        data_dir = self.path2results + "/data"
        model_save_path = os.path.join(self.path2results, f"model_weights_aod.pth")
        os.makedirs(figs_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        best_val_loss = float('inf')
        best_f1_score = float('-inf')  # Track the best validation loss
        val_data = []
        for epoch in range(num_epochs):

            model.train()
            running_loss = 0.0
            for i, (inputs, targets, _) in enumerate(train_loader):
                optimizer.zero_grad()
                obj_target = targets[0]
                outputs = model(inputs)
                loss = criterion(outputs[0], obj_target)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)  # Compute before resetting
            print(f"Epoch {epoch + 1} Avg Loss Train: {avg_loss:.3f}")
            loss_values.append(avg_loss)

            if (epoch + 1) % VAL_EACH_EPOCH == 0:
                model.eval()
                running_loss_val = 0.0
                correct, total = 0, 0

                correct_label = []
                pred_label = []
                with torch.no_grad():
                    for inputs, targets, cmd_idx in val_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs[0], targets[0])
                        running_loss_val += loss.item()

                        pred_label.append(torch.argmax(outputs[0].flatten(1), dim=1))
                        correct_label.append(torch.argmax(targets[0].flatten(1),dim=1))



                        new_score = self.get_ao_metrics_top1(outputs[0], targets[0])
                        command = self.get_command(val_dataset, cmd_idx)
                        scene = self.get_scene(val_dataset, cmd_idx)
                        od_new_score = self.get_od_metrics(outputs[1], targets[1], scene)
                        ad_new_score = self.get_ad_metrics(outputs[2], targets[2])
                        data_point = {
                            "epoch": epoch + 1,
                            "scene_id": scene.id,
                            "sctructure_type": command.type_id,
                            "action_type": command.action_id,
                            "od_target_score": od_new_score[0],
                            "od_spec_score": od_new_score[1],
                            "od_feasible_score": od_new_score[2],
                            "aod_target_score": ad_new_score,
                        }

                        val_data.append(data_point)
                        correct += new_score
                        total += 1
                avg_loss_val = running_loss_val / len(val_loader)
                f1_score = self.compute_f1_score(correct_label, pred_label)
                print(f"Epoch {epoch + 1} Avg Loss val: {avg_loss_val:.3f} F1 score: {f1_score:.3f}")
                print(f"Epoch {epoch + 1} Top-1 score {correct / total * 100} %")

                if round(best_f1_score, 3) < round(f1_score, 3):
                    best_f1_score = f1_score  # Update best validation loss
                    wait = 0  # Reset patience counter
                    torch.save(model.state_dict(), model_save_path)  # Save the best model
                else:
                    wait += 1  # Increase patience counter

                if wait >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break  # Stop training
        self.save_data(val_data, "aod")

    def get_ad_metrics(self, prediction, target):
        prediction = prediction.squeeze()
        pred_object_idx = (prediction == torch.max(prediction)).nonzero(as_tuple=True)[0]
        target_ad_score = 0
        if len(pred_object_idx) < 2 and pred_object_idx == target:
            target_ad_score = 1
        return target_ad_score

    def get_ao_metrics_top1(self, prediction, target):
        batch_size = prediction.shape[0]
        height = prediction.shape[1]
        width = prediction.shape[2]

        # After flattening
        prediction_flattened = prediction.flatten(1)
        target_flattened = target.flatten(1)

        # Get max indices
        prediction_flattened_idx = torch.argmax(prediction_flattened, dim=1)
        target_flattened_idx = torch.argmax(target_flattened, dim=1)

        # Recover original 2D position (row, col)
        prediction_row = prediction_flattened_idx // width
        prediction_col = prediction_flattened_idx % width

        target_row = target_flattened_idx // width
        target_col = target_flattened_idx % width

        correct_predictions = torch.sum(prediction_flattened_idx == target_flattened_idx)

        return correct_predictions

    def train_ad(self, batch_size=BATCH_SIZE,
                 learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS, patience=5):
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        torch.manual_seed(42)
        train_dataset = data_loader.MultiModalDataset(path2dataset=self.path2train, 
                                                      modality_type=self.modality_type,
                                                      detection_type="ad",
                                                      actions_complexity=self.actions_complexity)
        val_dataset = data_loader.MultiModalDataset(path2dataset=self.path2val,
                                                    modality_type=self.modality_type,
                                                     detection_type="ad",
                                                    actions_complexity=self.actions_complexity)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Initialize model, loss function, and optimizer
        model = self.select_model(detection_type="ad",config = train_dataset.config)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loss_values, val_accuracies = [], []

        # Create a directory to save the figures
        figs_dir = self.path2results + "/figs"
        data_dir = self.path2results + "/data"
        model_save_path = os.path.join(self.path2results, f"model_weights_ad.pth")
        os.makedirs(figs_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        best_val_loss = float('inf')
        best_f1_score = float('-inf')  # Track the best validation loss
        val_data = []
        for epoch in range(num_epochs):

            model.train()
            running_loss = 0.0
            for i, (inputs, targets, _) in enumerate(train_loader):
                obj_target = targets
                outputs = model(inputs)
                loss = criterion(outputs, obj_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)  # Compute before resetting
            print(f"Epoch {epoch + 1} Avg Loss Train: {avg_loss:.3f}")
            loss_values.append(avg_loss)

            if (epoch + 1) % VAL_EACH_EPOCH == 0:
                model.eval()
                running_loss_val = 0.0
                correct, total = 0, 0

                correct_label = []
                pred_label = []
                with torch.no_grad():
                    for inputs, targets, cmd_idx in val_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        running_loss_val += loss.item()
                        scene = self.get_scene(val_dataset, cmd_idx)
                        command = self.get_command(val_dataset, cmd_idx)
                        new_score = self.get_ad_metrics(outputs, targets)

                        pred_label.append(torch.argmax(outputs.flatten(1), dim=1))
                        correct_label.append(targets)

                        data_point = {
                            "epoch": epoch + 1,
                            "scene_id": scene.id,
                            "sctructure_type": command.type_id,
                            "action_type": command.action_id,
                            "ad_target_score": new_score
                        }
                        val_data.append(data_point)
                        correct += new_score
                        total += 1
                avg_loss_val = running_loss_val / len(val_loader)
                f1_score = self.compute_f1_score(correct_label, pred_label)
                print(f"Epoch {epoch + 1} Avg Loss val: {avg_loss_val:.3f} F1 score: {f1_score:.3f}")
                print(f"Epoch {epoch + 1} Top-1 object score {correct / total * 100} %")

                if round(best_f1_score, 3) < round(f1_score, 3):
                    best_f1_score = f1_score  # Update best validation loss
                    wait = 0  # Reset patience counter
                    torch.save(model.state_dict(), model_save_path)  # Save the best model
                else:
                    wait += 1  # Increase patience counter

                if wait >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break  # Stop training
        self.save_data(val_data, "ad")

    def train_od(self, batch_size=BATCH_SIZE,
                 learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS, patience=5):
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        torch.manual_seed(42)
        train_dataset = data_loader.MultiModalDataset(path2dataset=self.path2train, 
                                                      modality_type=self.modality_type,
                                                      detection_type="od",
                                                      actions_complexity=self.actions_complexity)
        val_dataset = data_loader.MultiModalDataset(path2dataset=self.path2val, 
                                                    modality_type=self.modality_type,
                                                     detection_type="od",
                                                    actions_complexity=self.actions_complexity)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Initialize model, loss function, and optimizer
        model = self.select_model(detection_type="od",config = train_dataset.config)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loss_values, val_accuracies = [], []

        # Create a directory to save the figures
        figs_dir = self.path2results + "/figs"
        data_dir = self.path2results + "/data"
        model_save_path = os.path.join(self.path2results, f"model_weights_od.pth")
        os.makedirs(figs_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        best_val_loss = float('inf')
        best_f1_score = float('-inf')# Track the best validation loss
        val_data = []
        for epoch in range(num_epochs):

            model.train()
            running_loss = 0.0
            for i, (inputs, targets, _) in enumerate(train_loader):
                obj_target = targets
                outputs = model(inputs)
                loss = criterion(outputs, obj_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
               # if i % 50 == 0:
                #    print(f"Epoch {epoch + 1:4}, Batch {i + 1 :4}, Loss: {loss.item():.3f}")
            avg_loss = running_loss / len(train_loader)  # Compute before resetting
            print(f"Epoch {epoch + 1} Avg Loss Train: {avg_loss:.3f}")
            loss_values.append(avg_loss)

            if (epoch + 1) % VAL_EACH_EPOCH == 0:
                model.eval()
                running_loss_val = 0.0
                correct, total = 0, 0

                pred_label = []
                correct_label=  []
                with torch.no_grad():
                    for inputs, targets, cmd_idx in val_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        running_loss_val += loss.item()
                        command = self.get_command(val_dataset, cmd_idx)
                        scene = self.get_scene(val_dataset, cmd_idx)
                        new_score = self.get_od_metrics(outputs, targets, scene)

                        data_point = {
                            "epoch": epoch + 1,
                            "scene_id": scene.id,
                            "sctructure_type": command.type_id,
                            "action_type": command.action_id,
                            "od_target_score": new_score[0],
                            "od_spec_score": new_score[1],
                            "od_feasible_score": new_score[2],
                        }
                        pred_label.append(torch.argmax(outputs.flatten(1),dim=1))
                        correct_label.append(targets)

                        val_data.append(data_point)
                        correct += new_score[0]

                        total += 1
                avg_loss_val = running_loss_val / len(val_loader)
                f1_score = self.compute_f1_score(correct_label, pred_label)
                print(f"Epoch {epoch + 1} Avg Loss val: {avg_loss_val:.3f} F1 score: {f1_score:.3f}")
                print(f"Epoch {epoch + 1} Val Target object score {correct / total} %")

                if round(best_f1_score, 3) < round(f1_score, 3):
                    best_f1_score = f1_score  # Update best validation loss
                    wait = 0  # Reset patience counter
                    torch.save(model.state_dict(), model_save_path)  # Save the best model
                else:
                    wait += 1  # Increase patience counter

                if wait >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break  # Stop training

        self.save_data(val_data, "od")

    def save_data(self, data, detection_type):
        df = pd.DataFrame(data)
        save_path = os.path.join(self.path2results, f"val_data_{detection_type}.csv")
        df.to_csv(save_path, index=False)
        print("Saved val Data!")

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

        if len(pred_object_idx) == 1 and pred_object_idx == target_object_idx:
            target_obj_score = 1

        if all(elem in possible_objects for elem in pred_object_idx):
            feasible_score = 1

        if target_object_idx in pred_object_idx:
            if all(elem in possible_objects for elem in pred_object_idx):
                specification_score = 1 - (len(pred_object_idx) - 1) / len(possible_objects)
        return [target_obj_score, feasible_score, specification_score]
