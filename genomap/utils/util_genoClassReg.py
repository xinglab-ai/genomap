import random
import numpy as np

def select_random_values(start, end, perc):
    total_values = end - start + 1
    num_values_to_select = int(total_values * perc)
    
    all_values = list(range(start, end + 1))
    selected_values = random.sample(all_values, num_values_to_select)
    remaining_values = list(set(all_values) - set(selected_values))
    
    return np.array(selected_values), np.array(remaining_values) 
