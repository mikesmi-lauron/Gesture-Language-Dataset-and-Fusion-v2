import numpy as np
import json
import ast

class Command:
    def __init__(self, cmd_id, cmd_info, config):
        self.dataset_config = config
        self.afm = self.create_afm()
        self.from_series(json.loads(cmd_info.to_json()))
        self.id = cmd_id
        self.preprocess_gpointing()
        self.gpointing_noise = self.count_noise_gpointing()

        self.ls_op = self.preprocess_language_words_op()
        self.l_act_desc_vec = self.preprocess_language_action_desc()
        self.g_act_desc_vec = self.preprocess_gesture_action_desc()
        self.scene = None

    def print_info(self):
        print(f"Command id: {self.id}")
        print("Gpointing noise: ", self.gpointing_noise)
        print("Meaning: ", self.meaning)
        print("Target actions: ", self.target_actions)
    def create_afm(self):
        actions_info = self.dataset_config["actions"]
        actions_tags = actions_info.keys()
        actions_features = set()
        for tag in actions_tags:
            actions_features.update(actions_info[tag]["conditions"].keys())
        afm = np.zeros((len(actions_tags), len(actions_features)))
        for i, tag in enumerate(actions_tags):
            action_conditions = actions_info[tag]["conditions"].keys()
            for j, feature in enumerate(action_conditions):
                if actions_info[tag]["conditions"][feature]:
                    afm[i, j] = 1 / len(action_conditions)
                if not actions_info[tag]["conditions"][feature]:
                    afm[i, j] = -1 / len(action_conditions)
        return afm
        
    def preprocess_language_words_op(self):
        s_op = []
        for j,l_obj_props_j in enumerate(self.l_obj_props):
            properties_count = len(self.dataset_config["object_properties"])
            words_limit = self.dataset_config["words_limit"]
            s_op_j = np.zeros((properties_count, words_limit))
            for i, word in enumerate(l_obj_props_j):
                # Get features
                property_idx = self.dataset_config["op2i"][word]
    
                s_op_j[property_idx, i] = 1
    
            if len(l_obj_props_j) < words_limit:
                s_op_j[:, len(l_obj_props_j):] = None
            s_op.append(s_op_j)

        return s_op

    def preprocess_language_action_desc(self):
        language_action_desc_vec =[]
        actions_info = self.dataset_config["actions"]
        actions_tags = actions_info.keys()
        for l_act_desc in self.l_act_desc:
            language_action_desc_vec_i = np.zeros((len(actions_tags)))
            for i, l_act_desc_i in enumerate(l_act_desc):
                for j, tag in enumerate(actions_tags):
                    if l_act_desc_i in actions_info[tag]["language_description"]:
                        language_action_desc_vec_i[j] = 1

            language_action_desc_vec.append(language_action_desc_vec_i)
        return language_action_desc_vec

    def preprocess_gesture_action_desc(self):
        gesture_action_desc_vec = []
        
        actions_info = self.dataset_config["actions"]
        actions_tags = actions_info.keys()

        for g_act_desc in self.g_act_desc:
            gesture_action_desc_vec_i = np.zeros((len(actions_tags)))
            for i, l_act_desc_i in enumerate(g_act_desc):
                for j, tag in enumerate(actions_tags):
                    if l_act_desc_i in actions_info[tag]["gesture_description"]:
                        gesture_action_desc_vec_i[j] = 1
                        
            gesture_action_desc_vec.append(gesture_action_desc_vec_i)

        return gesture_action_desc_vec

    def preprocess_gpointing(self):
        
        for i,pointing_i in enumerate(self.g_pointing_obj_probs):
            gpointing_obj_probs_updated = np.full(self.dataset_config["objects_limit"], None, dtype=float)
            gpointing_obj_probs_updated[:len(pointing_i )] = pointing_i
            self.g_pointing_obj_probs[i]  = gpointing_obj_probs_updated

    def preprocess_target_actions(self,ad = False):
        actions_info = self.dataset_config["actions"]
        actions_tags = actions_info.keys()
        target_idx = []
        if ad:
            target_actions = [self.meaning.split(' ')[0]]
        else:
            target_actions = self.target_actions
        for target_action in target_actions:
            for i, tag in enumerate(actions_tags):
                if target_action == tag:
                    target_idx.append(i)
                    break
        return target_idx 
            
    def preprocess_target_objects(self):
        return [int(target_id.split('-')[2]) for target_id in self.target_objs]
    def add_scene(self,scene):
        self.scene = scene
    def count_noise_gpointing(self):
        noise = []
        for i,pointing_i in enumerate(self.g_pointing_obj_probs):
            max_value = np.nanmax(pointing_i)
            mask = (pointing_i== max_value)
            noise.append(np.count_nonzero(mask))
        return noise

    def from_series(self, row):
        self.dataset_cmd_id = row.get("id")
        self.type_id = row.get("type_id")
        self.action_id = row.get("action_id")
        self.scene_id = row.get("scene_id")  # Assuming you have a setter method
        self.target_objs = ast.literal_eval(row.get("target_objs_id", ""))
        self.target_actions = ast.literal_eval(row.get("target_actions", ""))
        self.meaning = row.get("meaning", "")
        self.l_obj_props = ast.literal_eval(row.get("language_words_op", ""))
        self.l_act_desc = ast.literal_eval(row.get("language_words_ad", ""))
        self.g_act_desc = ast.literal_eval(row.get("gesture_words_ad", ""))
        self.g_pointing_obj_probs = ast.literal_eval(row.get("g_pointing_obj_probs", ""))
