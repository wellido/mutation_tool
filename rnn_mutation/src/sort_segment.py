import csv
import numpy as np


def read_csv(csv_path):
    """

    :param csv_path:
    :return:
    """

    csvFile = open(csv_path, 'r')
    reader = csv.reader(csvFile)
    result = []
    for item in reader:
        result.append(item)
    csvFile.close()
    result = np.array(result)
    return result


def influence_analyze(file_list, column):
    """

    :param file_list:
    :param column:
    :return:
    """
    distance_list = []
    for file_path in file_list:
        file_data = read_csv(file_path)
        distance_list.append([float(x[column]) for x in file_data])
    avg_distance = []
    time_step_num = len(distance_list[0])
    file_num = len(distance_list)
    if file_num > 1:
        for i in range(time_step_num):
            avg = 0
            for j in range(file_num):
                avg += distance_list[j][i]
            avg_distance.append(avg / file_num)
    else:
        avg_distance = distance_list[0]
    all_distance_np = np.asarray(avg_distance)
    sort_list = np.argsort(all_distance_np)
    print("after sorting time step: ", sort_list)


if __name__ == '__main__':
    csv_path = ["../../result/test.csv"]
    # result = read_csv(csv_path)
    # # print(read_csv(csv_path))
    # distance_1 = [float(x[2]) for x in result]
    # print(distance_1)
    influence_analyze(csv_path, 2)
