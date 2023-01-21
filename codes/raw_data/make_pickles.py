import dataset


for name in ['yahoo']:
# for name in ['istella', 'mslr', 'yahoo']:
    data = dataset.get_dataset_from_json_info(
                    name,
                    'datasets_info.json',
                    feature_normalization=False
                  ).get_data_folds()[0]
    data.read_data()
for name in ['istella', 'mslr', 'yahoo']:
    data = dataset.get_dataset_from_json_info(
                    name,
                    'datasets_info.json',
                    feature_normalization=True
                  ).get_data_folds()[0]
    data.read_data()
