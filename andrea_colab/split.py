import csv

metric = "recall"
metric = "f1"
filename = 'CV_KNN_' + metric + '.csv'

with open(filename, 'w', newline='') as csvf:
    writer = csv.writer(csvf)
    writer.writerow([metric, 'std_dev', 'n_neighbors','weights'])
    with open("split_to_process.txt") as f:
            for line in f:
                    x = line.split()
                    div = [x[0], x[1].split("-")[-1][:-1], x[4].split(",")[0], x[6].split("'")[1]]
                    writer.writerow(div)