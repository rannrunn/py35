def sort(arr):
    for i in range(len(arr), 1, -1):
        for j in range(i - 1):
            if arr[j] < arr[j + 1]:
                tmp = arr[j + 1]
                arr[j + 1] = arr[j]
                arr[j] = tmp
    return arr

def main():
    arr = [5, 4, 3, 6, 7]
    result = sort(arr)
    print(result)

if __name__ == '__main__':
    main()