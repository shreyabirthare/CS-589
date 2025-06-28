import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter
import sys
import random

def shuffle_dt(df):
  shuffled_df = shuffle(df)
  shuffled_df.reset_index(drop=True, inplace=True)
  return shuffled_df

def split(shuffled_df):
  train_df, test_df = train_test_split(shuffled_df, test_size=0.2)
  train_df.reset_index(drop=True, inplace=True)
  test_df.reset_index(drop=True, inplace=True)
  return train_df, test_df

def normalise(og_df, columns_to_normalise):
    df = og_df.copy()
    for col in columns_to_normalise:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

def encode_one_hot(df, columns_to_encode):
    encoded_df = df.copy()
    encoded_columns = []
    for col in columns_to_encode:
        encoded_cols = pd.get_dummies(df[col], prefix=col, dtype=float)
        encoded_columns.append(encoded_cols)
        encoded_df = encoded_df.drop(col, axis=1)
    encoded_df = pd.concat([encoded_df] + encoded_columns, axis=1)
    return encoded_df

def correct_encoding(encoded_df, needed_cols):
    for col in needed_cols:
        if col not in encoded_df.columns:
            encoded_df[col] = 0

    encoded_df = encoded_df[needed_cols]
    return encoded_df

def entropy_info(arr): # get entropy for current data arr - info gain
  gain = 0
  total = len(arr)
  #print(arr)
  category_counter = Counter(arr)
  for category, count in category_counter.items():
    gain -= (count/total) * math.log(count/total, 2)
  return gain

def entropy_gini(arr): # get entropy for current data arr
  gain = 1
  total = len(arr)
  category_counter = Counter(arr)
  for category, count in category_counter.items():
    gain -= (count/total) **2
  return gain

def gini_cri(df):
  avg = 0
  first_column = df.iloc[:, 0]
  attr = Counter(first_column)
  #print(attr)
  for category, count in attr.items():
    target_values = []
    for index, row in df.iterrows():
      if row[0] == category:
          target_values.append(row[1])
    entr = entropy_gini(target_values)
    #print(category, ": ", target_values, " entropy: ", entr)
    #print(len(target_values),df.shape[0])
    avg += entr * (len(target_values)/ df.shape[0])
  return avg

def information_gain(df): # get info gain for current col (df: 2x2 attribute and final label col?)
  avg = 0
  first_column = df.iloc[:, 0]
  #print(first_column)
  attr = Counter(first_column)
  #print(attr)
  for category, count in attr.items():
    target_values = []
    for index, row in df.iterrows():
      if row[0] == category:
          target_values.append(row[1])
    entr = entropy_info(target_values)
    #print(category, ": ", target_values, " entropy: ", entr)
    #print(len(target_values),df.shape[0])
    avg += entr * (len(target_values)/ df.shape[0])
  col2_values = df['target'].values
  #print(col2_values)
  entr_of_ds = entropy_info(col2_values)
  return entr_of_ds - avg

def gini_cri_num(df): 
  arr = df.iloc[:, 0].tolist()
  arr = sorted(arr)
  mean_values = []
  mini = 1e10000
  mv  = 0
  for i in range(1, len(arr)):
     mean_values.append(float((arr[i]+arr[i-1])/2))
  for mean_value in mean_values:
    df_left = df[(df.iloc[:, 0]) <= mean_value]
    df_right = df[(df.iloc[:, 0]) > mean_value]

    entropy_left = entropy_gini(df_left['target'])
    entropy_right = entropy_gini(df_right['target'])

    total_instances = len(df)
    total_instances_left = len(df_left)
    total_instances_right = len(df_right)

    res = ((total_instances_left / total_instances) * entropy_left) + ((total_instances_right / total_instances) * entropy_right)
    if(res < mini):
      mini = res
      mv = mean_value
  return mini, mv

def information_gain_num(df): 
  arr = df.iloc[:, 0].tolist()
  arr = sorted(arr)
  mean_values = []
  maxi = -1
  mv  = 0
  for i in range(1, len(arr)):
     mean_values.append(float((arr[i]+arr[i-1])/2))
  for mean_value in mean_values:
    df_left = df[(df.iloc[:, 0]) <= mean_value]
    df_right = df[(df.iloc[:, 0]) > mean_value]

    entropy_left = entropy_info(df_left['target'])
    entropy_right = entropy_info(df_right['target'])

    total_entropy = entropy_info(df['target'])
    total_instances = len(df)
    total_instances_left = len(df_left)
    total_instances_right = len(df_right)

    information_gain = total_entropy - ((total_instances_left / total_instances) * entropy_left) - ((total_instances_right / total_instances) * entropy_right)
    if(maxi < information_gain):
       maxi = information_gain
       mv = mean_value
  return maxi, mv

def select_attr_col(df, used_attributes):
    m = int(math.sqrt(df.shape[1] - 1))
    all_columns = df.columns.tolist()
    all_columns.remove('target')
    for attr in used_attributes:
        if attr in all_columns:
            all_columns.remove(attr)
    selected_columns = random.choices(all_columns, k=min(len(all_columns), m))
    return selected_columns

def find_winner(df, criteria, type_of_col, used_attributes=None):

    if used_attributes is None:
        used_attributes = set()

    attribute = None
    maxi = -1
    mini = sys.maxsize
    columns_to_include = select_attr_col(df, used_attributes)
    
    for column in columns_to_include:
        #print(column)
        df_subset = df[[column, 'target']]
        
        if criteria == 'info':
            if type_of_col[column]:
                ig, attr = information_gain_num(df_subset)
                #print(ig, attr)
            else:
                ig = information_gain(df_subset)
                
            if maxi < ig:
                maxi = ig
                if type_of_col[column]:
                    #print(column, attr)
                    attribute = [column,  attr]
                else:
                    attribute = [column, 0]
        else:
            if(type_of_col[column]):
                ig,attr = gini_cri_num(df_subset)
            else:
                ig = gini_cri(df_subset)
            if mini > ig:
                mini = ig
                if type_of_col[column]:
                    attribute = [column,  attr]
                else:
                    attribute = [column, 0]
    
    return attribute

def get_subtable(data, node,value):
  return data[data[node] == value].reset_index(drop=True)

def buildTree(data, criteria, type_of_col, threshold, min_instances, used_attributes = None): # only threshold
    target = data.keys()[-1]

    # stopping criteria -  all remaining nodes belong to the same class
    if len(data[target].unique()) == 1: 
        return data[target].iloc[0]
    
    # stopping criteria - only one attribute to split into
    if len(data.columns) == 1:  
        return data[target].mode()[0]
    
    # stopping criteria - no same attributes again 
    if used_attributes is None:
        used_attributes = set()
    
    # stopping criteria - 85%
    category_percentage = data['target'].value_counts(normalize=True) * 100
    categories_above_threshold = category_percentage[category_percentage >= threshold]
    cat = categories_above_threshold.index.tolist()
    
    if(len(cat) > 0):  # if >= threshold of some category exists then return that category
      return cat[0]

    pair_node = find_winner(data, criteria, type_of_col, used_attributes)
    if pair_node is None:
        return data[target].mode()[0]

    column, mean_val = pair_node
    tree = {column: {}}

    if type_of_col[column]:
        attr_val = np.array([f'> {mean_val}', f'<= {mean_val}'])
    else:
        attr_val = np.unique(data[column])

    

    for val in attr_val:
        if type_of_col[column] and val.startswith('> '):
            subtable = data[data[column] > mean_val].reset_index(drop=True)
        elif type_of_col[column] and val.startswith('<= '):
            subtable = data[data[column] <= mean_val].reset_index(drop=True)
        else:
            subtable = get_subtable(data, column, val)

        used_attributes.add(column)

        clValue, counts = np.unique(subtable[target], return_counts=True)
        if(len(counts) == 0):
            continue
        elif len(counts) == 1:  # Only 1 class left
            tree[column][val] = clValue[0]  # Leaf node
        else:
            # Recursively build the subtree with the updated set of used attributes
            subtree = buildTree(subtable, criteria, type_of_col,  threshold, min_instances, used_attributes.copy())
            tree[column][val] = subtree

    return tree

def traverse_tree(tree, instance, type_of_col):
    current_node = tree
    
    while isinstance(current_node, dict):
        attribute, condition_dict = list(current_node.items())[0]
        attribute_value = instance.get(attribute)
        #print(attribute, (attribute_value))
        if type_of_col[attribute] == 0:  # Check if the attribute is categorical
            next_node = condition_dict.get(attribute_value)
            #print(next_node)
        else:  # The attribute is numeric
            next_node = None
            for condition, subtree in condition_dict.items():
                #print(condition)
                if condition.startswith('<='):
                    threshold = float(condition[2:])
                    #print(threshold)
                    if attribute_value <= threshold:
                        next_node = subtree
                        break
                elif condition.startswith('> '):
                    threshold = float(condition[2:])
                    if attribute_value > threshold:
                        #print(threshold)
                        next_node = subtree
                        break
            #print(next_node)
        
        current_node = next_node
    
    return current_node

def distribute_records(df, k=10):
    class_indices = {}

    classes = df['target'].unique()

    for class_label in classes:
        indices = np.where(df['target'] == class_label)[0]
        class_indices[class_label] = indices

    category_counts = {category: group for category, group in (df['target'].value_counts()).items()}

    required_counts = {category: math.floor(count/k) for category, count in category_counts.items()}

    smaller_dfs = []

    # Distribute records based on required counts for each category
    for i in range(10):
        temp_df = pd.DataFrame(columns=df.columns)
        for category, count_list in required_counts.items():
            selected_indices = np.random.choice(class_indices[category], size=required_counts[category], replace=False)
            
            for val in selected_indices:
                class_indices[category] = class_indices[category][class_indices[category] != val]
            selected_records = df.iloc[selected_indices]
            temp_df = pd.concat([temp_df, selected_records])
   
        smaller_dfs.append(temp_df)

    # for i in range(10):
    #     print(smaller_dfs[i]['target'].value_counts())

    groups = [0,1,2,3,4,5,6,7,8,9]
    for category, indices in class_indices.items():
        select_idx = np.random.choice(groups, size=len(indices), replace=False)
        #print(category, indices, select_idx) 
        for id in range(len(indices)):
            index = indices[id]
            idx = select_idx[id]
            selected_df = df.iloc[index].to_dict() 
            curr_df = smaller_dfs[idx]
            new_row_df = pd.DataFrame([selected_df])
            curr_df = pd.concat([curr_df, new_row_df], ignore_index=True)
            smaller_dfs[idx] = curr_df

    smaller_dfs = [shuffle_dt(df) for df in smaller_dfs]

    return smaller_dfs

def bootstrap(df, num):
    bootstrap_datasets = []
    n = len(df)
    for _ in range(num):
        bootstrap_indices = np.random.choice(n, size=n, replace=True)
        bootstrap_sample = df.iloc[bootstrap_indices]
        bootstrap_sample.reset_index(drop=True, inplace=True)
        bootstrap_datasets.append(bootstrap_sample)
    return bootstrap_datasets

def majority_vote(arr):
    class_counts = Counter(arr)
    max_count = max(class_counts.values())
    majority_classes = [k for k, v in class_counts.items() if v == max_count]
    
    if len(majority_classes) > 1:
        import random
        final_prediction = random.choice(majority_classes)
    else:
        final_prediction = majority_classes[0]
    
    return final_prediction

def true_positives(df, category):
    return len(df[(df['target'] == category) & (df['predicted_target'] == category)])

def false_positives(df, category):
    return len(df[(df['target'] != category) & (df['predicted_target'] == category)])

def true_negatives(df, category):
    return len(df[(df['target'] != category) & (df['predicted_target'] != category)])

def false_negatives(df, category):
    return len(df[(df['target'] == category) & (df['predicted_target'] != category)])

def perform(smaller_dfs, categories, type_of_col, threshold, min_instances, criteria):
    results = []
    for fold in range(0,10):
        print('Fold: ', fold)
        test_df = smaller_dfs[fold]
        train_df = pd.concat(smaller_dfs[:fold] + smaller_dfs[fold+1:])
        #print('Fold: ', fold, 'Length of train and test: ', len(train_df), len(test_df))
        test_df.reset_index(inplace=True, drop=True)
        train_df.reset_index(inplace=True, drop=True)
        num_of_trees_arr = [1, 5, 10, 20, 30, 40, 50]
        for num_of_trees in num_of_trees_arr:
            datasets = bootstrap(train_df, num_of_trees)
            n_trees = []
            for i in range(len(datasets)):
                finalTree  = buildTree(datasets[i], criteria, type_of_col, threshold, min_instances)
                n_trees.append(finalTree)
            predicted_val_arr = []
            for i in test_df.index:
                instance = test_df.iloc[i]
                predicted_val_test = []
                for idx in range(len(n_trees)):
                    predicted = traverse_tree(n_trees[idx], instance, type_of_col)
                    predicted_val_test.append(predicted)
                predicted_val = majority_vote(predicted_val_test)
                predicted_val_arr.append(predicted_val)
            test_df['predicted_target'] = predicted_val_arr
            num_instances = (test_df['target'] == test_df['predicted_target']).sum()
            accuracy = num_instances/test_df.shape[0] * 100
            precision = 0
            recall = 0
            f1_score = 0
            for cat in categories:
                tp = true_positives(test_df, cat)
                fp = false_positives(test_df, cat)
                tn = true_negatives(test_df, cat)
                fn = false_negatives(test_df, cat)
                if(tp == 0):
                    precision = 0
                else:
                    precision += (tp)/(tp+fp)
                if(tp == 0):
                    recall = 0
                else:
                    recall += (tp)/(tp+fn)
                if(tp == 0):
                    f1_score = 0
                else:
                    f1_score += tp/(tp + 0.5 * (fp+fn))
            precision /= len(categories)
            recall /= len(categories)
            f1_score /= len(categories)
            results.append({'Fold': fold, 'Trees': num_of_trees, 'Accuracy':accuracy, 'Recall': recall, 'Precision': precision, 'F1 Score': f1_score})
            test_df.drop('predicted_target', axis=1, inplace=True)
            #print(num_of_trees, accuracy, precision, recall, f1_score)

    results_df = pd.DataFrame(results,columns=['Fold', 'Trees', 'Accuracy', 'Recall', 'Precision', 'F1 Score'])
    average_performance_df = results_df.groupby('Trees').agg({'Accuracy': 'mean', 'Recall': 'mean', 'Precision': 'mean', 'F1 Score': 'mean'}).reset_index()
    return results_df, average_performance_df

def cost(y_true, y_pred):
    m = len(y_true)  # Number of training examples
    epsilon = 1e-15  # Avoid numerical instability with log(0)
    cross_entropy = -1 / m * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    return cross_entropy

def initialize_weights(architecture):
    weights = []
    for i in range(1, len(architecture)):
        theta_i = np.random.rand(architecture[i], architecture[i-1] + 1) 
        weights.append(theta_i)
    return weights

def initialize_weights_minustoplus(architecture):
    weights = []
    for i in range(1, len(architecture)):
        theta_i = np.random.uniform(-1, 1, size=(architecture[i], architecture[i-1] + 1))
        weights.append(theta_i)
    return weights

def initialize_weights_as_one(architecture):
    weights = []
    for i in range(1, len(architecture)):
        theta_i = np.ones((architecture[i], architecture[i-1] + 1)) 
        weights.append(theta_i)
    return weights

def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    
    return 1 / (1 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    return x * (1 - x)

def compute_S(weights):
    S = 0
    for layer_weights in weights:
        layer_weights = layer_weights[:, 1:]
        S += np.sum(layer_weights ** 2)
    return S

def calculate_cost(x, y, weights, regularization_param, printing=False):
    j = 0
    for tr_inst in range(len(x)):
        input_array = x[tr_inst]
        target = y[tr_inst]
        y_pred = forward_pass(input_array, weights, printing)
        #print(target, y_pred)
        cost_j = cost(target, y_pred)
        if(printing):
            print('Cost ', tr_inst,' : ',cost_j)
        #print('COst: ', tr_inst, ' ', cost_j)
        j = j + cost_j
    j /= len(x)

    s = compute_S(weights)
    s = (regularization_param/(2*len(x))) * s
    if(printing):
        print('Regularized cost: ', j+s)
    return j+s

def forward_pass(input_array, weights, printing=False):
    outputs = [input_array.reshape(-1, 1)]
    for w in weights:
        input_with_bias = np.vstack((np.ones((1, outputs[-1].shape[1])), outputs[-1]))
        if(printing):
            print('a', input_with_bias)
            print('z: ', np.dot(w, input_with_bias))
        
        output = sigmoid(np.dot(w, input_with_bias))
        outputs.append(output)
        if(printing):
            print('final: ', output)
    return outputs[-1]

def backpropagation(x, y, architecture, weights, learning_rate, regularization_param, printing=False):
    gradients = [np.zeros_like(theta) for theta in weights]
    deltas = [np.zeros((size,1)) for size in architecture[1:]]
    for tr_inst in range(len(x)):
        input_array = x[tr_inst]
        target = y[tr_inst]

        outputs = [input_array.reshape(-1, 1)]
        for w in weights:
            input_with_bias = np.vstack((np.ones((1, outputs[-1].shape[1])), outputs[-1]))
            output = sigmoid(np.dot(w, input_with_bias))
            outputs.append(output)
        error = outputs[-1] - target.reshape(-1,1)
        deltas[-1] = error
        for i in range(len(weights) - 1, 0, -1):
         
            error = np.dot(weights[i].T[1:], deltas[i])
            
         
            delta_hidden = error * sigmoid_derivative(outputs[i])

            deltas[i - 1] = delta_hidden
        if(printing):
            print('deltas: ', deltas)
        for i in range(len(weights)):
            inputs = np.vstack((np.ones((1, outputs[i].shape[1])), outputs[i]))
            #print(deltas[i], inputs.T)
            gradient = np.dot(deltas[i], inputs.T) 
            #print('gradients: ', gradient)
            if(printing):
                print('gradient ',i,' : ',gradient)
            gradients[i] += gradient

    regularized = [regularization_param * w for w in weights]
   
    for i in range(len(regularized)):
        regularized[i][:, 0] = 0
   
    for i in range(len(gradients)):
        gradients[i] += regularized[i]
    gradients = [gradient / len(x) for gradient in gradients] 
    if(printing):
        print('Regularized gradients: ', gradients)
    for i in range(len(weights)):
        weights[i] = weights[i] - learning_rate * gradients[i]

    return weights

def calculate_performance(results):
    categories = (results['target'].unique()).tolist()
    num_instances = (results['target'] == results['predicted_target']).sum()
    #print(num_instances, results.shape[0])
    accuracy = (num_instances/results.shape[0])* 100
    f1_score = 0
    for cat in categories:
        tp = true_positives(results, cat)
        fp = false_positives(results, cat)
        tn = true_negatives(results, cat)
        fn = false_negatives(results, cat)
        if(tp == 0):
            f1_score += 0
        else:
            f1_score += tp/(tp + 0.5 * (fp+fn))
        #print(tp,fp,tn,fn, f1_score)
    f1_score /= len(categories)
    return accuracy, f1_score

def knn(data, instance, k):
  print('new knn')
  dataset = data.copy()
  dist = []
  distance = 0
  for i in dataset.index: # euclidean distance
    for col in data.columns:
        if col == 'target':
           continue
        distance += (float(dataset.iloc[i][col] - instance[col]))**2
    distance = math.sqrt(float(distance))
    dist.append(distance)
  dataset['euclidean_dist'] = dist
  dataset.sort_values(by='euclidean_dist', ascending=True, inplace=True)
  dataset.reset_index(drop=True, inplace=True)
  top_k_labels = dataset['target'].head(k)
  majority_value = top_k_labels.value_counts().idxmax()
  return majority_value

def accuracy_on_train(ds, k):
  data = ds.copy()
  predicted = []
  for i in data.index:
    instance = data.iloc[i]
    majority_value = knn(data, instance, k)
    #print(majority_value)
    predicted.append(majority_value)
  data['predicted_target'] = predicted
  return calculate_performance(data)

def accuracy_on_test(ds, data, k):
  train_data = ds.copy()
  test_data = data.copy()
  predicted = []
  for i in test_data.index:
    instance = test_data.iloc[i]
    majority_value = knn(train_data, instance, k)
    predicted.append(majority_value)
  test_data['predicted_target'] = predicted
  
  return calculate_performance(test_data)



