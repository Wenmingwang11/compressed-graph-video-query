import time

def find_all_increasing_sequences(index, current_row=1, current_sequence=None, all_sequences=None):
    """
    This function is used to find all increasing sequences in a given index.

    Parameters:
    index (dict): A dictionary where each key is an integer and the value is a list of integers.
    current_row (int): The current row being processed. Default is 1.
    current_sequence (list): The current sequence being built. Default is None.
    all_sequences (list): The list of all sequences found so far. Default is None.

    Returns:
    list: A list of all increasing sequences found in the index.

    """

    # Initialize current_sequence and all_sequences if they are None
    if current_sequence is None:
        current_sequence = []
    if all_sequences is None:
        all_sequences = []

    # print(current_sequence)


    # If we have completed the sequence
    if current_row > len(index):
        all_sequences.append(current_sequence)
        return

    # Get the last value in the current sequence, or negative infinity if the sequence is empty
    last_value = current_sequence[-1] if current_sequence else float('-inf')

    # Iterate over the values in the current row of the index
    for value in index[current_row]:
        # If the current value is greater than the last value in the sequence
        if value > last_value:

            # Try to extend the sequence
            find_all_increasing_sequences(index, current_row + 1, current_sequence + [value], all_sequences)

    # Return the list of all sequences
    return all_sequences

# def id_count_prune(current_sequence,):
#     matched_result_per_frame = collect_matched_result_per_frame2(new_frames,
#                                                                  frame_match_array_dict,
#                                                                  pattern_mapping_dict)
#     # print(new_frames)
#     # print(matched_result_per_frame)
#     index_id_count_dict = index_count_id_occurrences(matched_result_per_frame, id_type_dict)
#     pattern_id_count_dict = pattern_id_counts_list[pattern_id - 1]
#     result = check_pattern_matches2(pattern_id_count_dict, index_id_count_dict)

index1 = {
    1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    2: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    3: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    4: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    5: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    6: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    7: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    8: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    9: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    10: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}
inverted_index = {
        1: [3, 6],
        2: [4, 5, 7],
        3: [3],
        4: [7, 9],
        5: [8, 9, 10]
    }
s_time = time.time()
all_sequences = find_all_increasing_sequences(inverted_index)
print("Time taken:", time.time() - s_time)
print(len(all_sequences))
# for seq in all_sequences:
#     print(seq)
