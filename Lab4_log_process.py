
f = open("log1_epoch4.txt")
lines = f.readlines()
for line in lines:
    line = line.split()
    loss = None
    acc = None
    for i in range(len(line)):
        if line[i] == "Loss":
            loss = float(line[i+1])
        if line[i] == "Accuracy":
            acc = float(line[i+1])
    if loss != None and acc != None:
        print(loss, ",", acc)