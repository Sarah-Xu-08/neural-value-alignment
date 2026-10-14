import numpy as np


def label_transfer(labels, game_index, so_type, pe_type):
    """
    Transfer labels to suit a specific type of decoder

    Parameters:
        labels (list): List of labels
        game_index (int 1/2): Train/test index (game 1/2), different methods for labeling,
        so_type (str 'sub'/'obj'): Subjective/objective label type
        pe_type (str 's10'/'r10'/'pe10'/'srpe3?4'): SPE(0,1)/RPE(0,1)/PE(0,1)/SRPE (the most detailed one 0,1,2,3)

    Returns:
        new_labels (list): Transformed labels
    """
    game_labels = []
    new_labels = []


    if game_index == 1:
        # "S0R0": 0, "S1R0": 1, "S0R1": 2
        if pe_type == 'srpe3':
            game_labels = labels
        elif pe_type == 's10':
            game_labels = [0 if l==2 else l for l in labels]
        elif pe_type == 'r10':
            game_labels_0 = [0 if l==1 else l for l in labels]
            game_labels = [1 if l==2 else l for l in game_labels_0]
        elif pe_type == 'pe10':
            game_labels = [1 if l==2 else l for l in labels]
        else:
            ValueError('No or wrong value assigned to pe_type')

        return game_labels

    # Preprocess labels according to subjective/objective label
    elif game_index == 2:
        if so_type == 'sub':
            for this_label in labels:
                sub_label = this_label if isinstance(this_label, int) else this_label[0]
                sub_label = map_sub_labels(sub_label)
                game_labels.append(sub_label)
        elif so_type == 'obj':
            for this_label in labels:
                obj_label = this_label if isinstance(this_label, int) else this_label[1]
                obj_label = map_obj_labels(obj_label)
                game_labels.append(obj_label)

        # Change to specific label types: spe, rpe, pe
        for label in game_labels:
            new_label = map_pe_labels(label,pe_type)
            new_labels.append(new_label)

        return new_labels

def pick_data_by_sr(data, labels, sr_pick, pe_type):
    '''
    Pick data and labels according to specific sr_pick
    s10: SPE (0,1)
    r10: RPE (0,1)
    pe10: PE (0,1)
    sr11: SRE, RPE
    '''
    assert pe_type in ['srpe3', 'srpe4'], 'pe_type must be srpe3 or srpe4. Others not support yet'
    
    if pe_type in ['srpe3', 'srpe4']:
        if sr_pick == 's10':
            pick_indices =  np.where(np.isin(labels, [0, 1]))[0]
        elif sr_pick == 'r10':
            pick_indices = np.where(np.isin(labels, [0, 2]))[0]
        elif sr_pick == 'sr11':
            pick_indices = np.where(np.isin(labels, [1, 2]))[0]
        elif sr_pick == 'pe10':
            new_labels = [1 if l in [1, 2, 3] else 0 for l in labels]
            return data, new_labels
        else:
            raise ValueError('No or wrong value assigned to sr_pick')

        new_data = data[pick_indices]
        new_labels = labels[pick_indices]

        # collapse to 0/1
        uniq = np.unique(new_labels)
        mapper = {u: i for i, u in enumerate(uniq)}
        new_labels = np.vectorize(mapper.get)(new_labels)

        return new_data, new_labels

def map_obj_labels(label):
    """Map objective labels."""
    if label in [100, 40]:  # S0R0
        return 0
    elif label in [11, 12, 13, 14]:  # S1R0
        return 1
    elif label in [20, 41]:  # S0R1
        return 2
    elif label in [31, 32, 33, 34]:  # S1R1
        return 3

def map_sub_labels(label):
    """Map subjective labels."""
    if label == 40:
        return 0
    elif label == 41:
        return 2
    else:
        return label

def map_pe_labels(label, pe):
    """Map PE labels."""
    if pe == 's10':
        if label in [0, 2]: # S0R0, S0R1
            return 0
        elif label in [1, 3]: # S1R0, S1R1
            return 1
    elif pe == 'r10':
        if label in [0, 1]:  # S0R0, S1R0
            return 0
        elif label in [2, 3]: # S0R1, S1R1
            return 1
    elif pe == 'pe10':
        if label in [1, 2, 3]: # S1R0, S0R1, S1R1
            return 1

    return label  # Return original label if PE type is not specified


if __name__ == "__main__":
    import pandas as pd
    a = pd.read_pickle("A03_epochs_2.pkl")

    for t in [1, 2]:
        for so in ["sub", "obj"]:
            for pe in ["pe", "rpe", "spe", "srpe4"]:
                print(t, so, pe)
                print(label_transfer(a[2], t, so, pe))

