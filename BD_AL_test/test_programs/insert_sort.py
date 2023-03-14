def insertionSort(a,b,c,d,e):
    alist = [a,b,c,d,e]
    for index in range(1,len(alist)):

        currentvalue = alist[index]
        position = index

        while position>0 and alist[position-1]>currentvalue:
            alist[position]=alist[position-1]
            position = position-1

        alist[position]=currentvalue
        print(alist)

if __name__ == '__main__':
    alist = [54,26,93,17,77,31,44,55,20]
    insertionSort(54,26,93,17,77,31,44,55,20)
    # print(alist)