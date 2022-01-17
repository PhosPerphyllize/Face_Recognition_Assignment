

def getUntill(fp, target:str):
    line = ""
    while 1:
        word = fp.read(1)
        if word == target or word=="":
            break
        line = line + word
    return line

def readTxt(root:str):
    fp = open(root)
    path_tab = []
    target_tab = []
    step = 0
    while 1:
        path = getUntill(fp, " ")
        if path == "":
            break
        path_tab.append(path)
        target = []
        for i in range(10):
            x = float(getUntill(fp, " "))
            target.append(x)
        target_tab.append(target)

        getUntill(fp, "\n")
        fp.read(1)
        step += 1
    fp.close()
    return path_tab, target_tab

if __name__ == '__main__':
    # path_tab, target = readTxt("../Dataset/training.txt")
    path_tab, target_tab = readTxt("..\..\p1_test2.txt")
    print(path_tab[-1])
    print(target_tab[-1])

    print("Code end")
