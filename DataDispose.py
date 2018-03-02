import csv

TrainingFilePath = "Datas//train.csv"
InputDataNumbers = 7
OutputDataNumbers = 2
TrainingDataNumbers = 800
"""
读取CSV数据到list数组中
0 index-> input , 1 -> index output
第一个返回值为训练数据，第二个为测试数据
"""
def read_csv_data():
    countOfLine = 0
    datas = []

    with open(TrainingFilePath, 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)

        trainDatas = []
        testDatas = []

        for row in reader:
            #跳过第一行
            if(row[0] == "PassengerId"):
                continue

            #自加计算
            countOfLine = countOfLine + 1

            # 乘客没有存活下来
            if (int(row[1]) == 0):
                ideal = [0, 1]
            # 乘客存活下来了
            else:
                ideal = [1, 0]

            if(row[5] == ""):
                age = 0
            else:
                age = float(row[5])

            try:
                # Pclass , Sex , Age , SibSp , Parch , Fare , Embarked
                input = [float(row[2]), change_sex_to_num(row[4]),
                         age, float(row[6]), float(row[7]),
                         float(row[9]), change_embarked_port_to_num(row[11])]
            except:
                print()


            if(countOfLine <= TrainingDataNumbers):
                trainDatas.append([input, ideal])
            else:
                testDatas.append([input, ideal])

    return trainDatas, testDatas


"""
将性别字符转化为数字
male(男性) = -1
female(女性) = 1
"""
def change_sex_to_num(sexString):
    if(sexString == "male"):
        return -1
    elif(sexString == "female"):
        return 1
    else:
        raise ValueError("sex string should be male or female!")

"""将上传地点转化为数字
Cherbourg(C) = 1
Queenstown(Q) = 2
Southampton(S) = 3
"""
def change_embarked_port_to_num(embarkedString):
    if(embarkedString == "C"):
        return 1
    elif(embarkedString == "Q"):
        return 2
    elif(embarkedString == "S"):
        return 3
    else:
        raise ValueError("embarked string should be C Q or S" + "Error Input : " + embarkedString)
