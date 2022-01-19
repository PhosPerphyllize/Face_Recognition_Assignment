

def getUntill(fp, target:str, jump:bool=False):
# 读取文件，直到到target字符串为止，返回中间fp读取到的全部字符串 jump: 是否跳过同样的字符
    line = ""
    while 1:
        word = fp.read(1)
        if word == target or word=="":
            break
        line = line + word

    while jump:
        word = fp.read(1)
        if word == target:
            continue
        else:
            fp.seek(fp.tell()-1, 0)  # 当前位置前移动一格，吐出字符
            break
    return line

def readTxt(root:str):
# 读取对应模型的txt文件，参数空格分隔，返回图片路径，参数， root：txt文件路径
    fp = open(root, mode="r",encoding="utf-8")
    path_tab = []
    target_tab = []
    while 1:
        path = getUntill(fp, " ")
        if path == "":
            break
        path_tab.append(path)
        target = []
        for i in range(9):
            x = float(getUntill(fp, " ", jump=True))
            target.append(x)

        x = float(getUntill(fp, "\n"))
        target.append(x)
        target_tab.append(target)

    fp.close()
    return path_tab, target_tab

if __name__ == '__main__':
    # path_tab, target = readTxt("../Dataset/training.txt")
    path_tab, target_tab = readTxt("test.txt")
    print(path_tab[2])
    print(target_tab[2])

    print("Code end")
