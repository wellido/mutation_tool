import csv
import numpy as np
from drawer import *
import argparse


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


def influence_analyze(file_list, column, save_path):
    """

    :param file_list:
    :param column:
    :param save_path:
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
    # print("after sorting time step: ", sort_list)
    np.savez(save_path + "KS2.npz", index=sort_list)
    draw(avg_distance, save_path)
    print("completed.")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", "-csv_path",
                        type=str,
                        help="csv path")
    parser.add_argument("--save_path", "-save_path",
                        type=str,
                        help="save path")
    parser.add_argument("--column", "-column",
                        type=int,
                        help="column")
    args = parser.parse_args()
    csv_path = []
    csv_path.append(args.csv_path)
    save_path = args.save_path
    column = args.column
    influence_analyze(csv_path, column, save_path)


if __name__ == '__main__':
    run()
    # csv_path = ["../../result/imdb_lstm_gf_data2_0.1.csv"]
    # # result = read_csv(csv_path)
    # # # print(read_csv(csv_path))
    # # distance_1 = [float(x[2]) for x in result]
    # # print(distance_1)
    # save_path = "../../result/"
    # influence_analyze(csv_path, 4)

# python sort_segment.py -csv_path ../../result/imdb_lstm_gf_data2_0.1.csv -save_path ../../result/ -column 4
