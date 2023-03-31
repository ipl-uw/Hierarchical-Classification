import pandas as pd

# Change the path to the absolute path of the csv file on your PC.


# csv_input = pd.read_csv('Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark/hierarchical_classification_result.csv')
# csv_input = pd.read_csv("E:\hierarchy species id\keith_problem_data/tracking results folder\AK-50308-220423_214636-C1H-025-220524_210051_809_1/hierarchical_classification_result.csv")

# csv_input = pd.read_csv("Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\Vessel 6-210804_212407-C1H-001-210816_211530_546-result/flat_classification_result.csv")
# csv_input = pd.read_csv("Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\Vessel 6-210804_212407-C1H-001-210816_211530_546-result/hierarchical_classification_result_true!.csv")

# csv_input = pd.read_csv("Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\AK-50308-220423_214636-C1H-025-220524_210051_809_1-20230105T014047Z-001-result\AK-50308-220423_214636-C1H-025-220524_210051_809_1/flat_classification_result.csv")
# csv_input = pd.read_csv("Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\AK-50308-220423_214636-C1H-025-220524_210051_809_1-20230105T014047Z-001-result\AK-50308-220423_214636-C1H-025-220524_210051_809_1/hierarchical_classification_result_true!.csv")

# csv_input = pd.read_csv("Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\Vessel 6-210804_212407-C1H-001-210816_211530_546-result/flat_classification_result_aug.csv")
# csv_input = pd.read_csv("Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\AK-50308-220423_214636-C1H-025-220524_210051_809_1-20230105T014047Z-001-result\AK-50308-220423_214636-C1H-025-220524_210051_809_1/flat_classification_result_aug.csv")
csv_input = pd.read_csv("Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\AK-50308-220423_214636-C1H-025-220524_210051_809_1-20230105T014047Z-001-result\AK-50308-220423_214636-C1H-025-220524_210051_809_1/flat_classification_result_lmmd.csv")
csv_input['length'] = 0
csv_input['location'] = 0
csv_input['depredation'] = 0
csv_input['release'] = 0
csv_input['injury'] = 0
csv_input['labeled_class'] = 0
csv_input['labeled_box'] = 0
csv_input['labeled_id'] = 0
csv_input['labeled_length'] = 0
csv_input['labeled_kept'] = 0
csv_input['comment'] = ""
csv_input['damages'] = ""
csv_input['occluded'] = 0
csv_input['comment2'] = ""
csv_input['comment3'] = ""
csv_input.head()

# The output is saved under the same folder. You can change the name.
# csv_input.to_csv('Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark/hierarchical_classification_result_proecssed.csv', index=False)
# csv_input.to_csv('E:\hierarchy species id\keith_problem_data/tracking results folder\AK-50308-220423_214636-C1H-025-220524_210051_809_1/hierarchical_classification_result_proecssed.csv', index=False)
# csv_input.to_csv("Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\Vessel 6-210804_212407-C1H-001-210816_211530_546-result/hierarchical_classification_result_true!_processed.csv", index=False)

# csv_input.to_csv("Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\AK-50308-220423_214636-C1H-025-220524_210051_809_1-20230105T014047Z-001-result\AK-50308-220423_214636-C1H-025-220524_210051_809_1/flat_classification_result_procesed.csv", index=False)
# csv_input.to_csv("Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\AK-50308-220423_214636-C1H-025-220524_210051_809_1-20230105T014047Z-001-result\AK-50308-220423_214636-C1H-025-220524_210051_809_1/hierarchical_classification_result_true!-processed.csv", index=False)

# csv_input.to_csv("Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\Vessel 6-210804_212407-C1H-001-210816_211530_546-result/flat_classification_result_aug-processed.csv", index=False)
# csv_input.to_csv("Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\AK-50308-220423_214636-C1H-025-220524_210051_809_1-20230105T014047Z-001-result\AK-50308-220423_214636-C1H-025-220524_210051_809_1/flat_classification_result_aug-procesed.csv", index=False)
csv_input.to_csv("Z:\Jie Mei/rail data\hierarchy_data_for_Transformer-SVM/test_sleeper_shark\AK-50308-220423_214636-C1H-025-220524_210051_809_1-20230105T014047Z-001-result\AK-50308-220423_214636-C1H-025-220524_210051_809_1/flat_classification_result_lmmd-procesed.csv", index=False)
