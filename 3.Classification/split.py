import csv

metric = "recall"
metric = "f1"
filename = 'CV_OVERSAMPLING_RANDOM_FOREST_' + metric + '.csv'

with open(filename, 'w', newline='') as csvf:
    writer = csv.writer(csvf)

    # ADABOOST
    # writer.writerow([metric, 'std_dev', 'learning_rate', 'n_estimators'])

    # DECISION TREE
    # writer.writerow([metric, 'std_dev', 'criterion', 'max_depth','min_samples_leaf', 'min_samples_split'])

    # KNN
    # writer.writerow([metric, 'std_dev', 'n_neighbors', 'weights'])

    # RANDOM FOREST
    writer.writerow([metric, 'std_dev', 'n_estimators'])

    with open("split_to_process.txt") as f:
        for line in f:
            x = line.split()

            # ADABOOST
            # div = [x[0], x[1].split("-")[-1][:-1], x[4].split(",")[0], x[6].split("}")[0]]

            # DECISION TREE
            # div = [x[0], x[1].split("-")[-1][:-1], x[4].split(",")[0].split("'")[1], x[6].split("'")[0][:-1], x[8][:-1], x[10].split("}")[0]]

            # KNN
            # div = [x[0], x[1].split("-")[-1][:-1],x[4].split(",")[0], x[6].split("'")[1]]

            # RANDOM FOREST
            div = [x[0], x[1].split("-")[-1][:-1],x[4].split(",")[0][:-1]]

            writer.writerow(div)
