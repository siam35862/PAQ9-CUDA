# compare.py

def compare_files(file1, file2):
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        data1 = f1.read()
        data2 = f2.read()

    if data1 == data2:
        print("Files are identical.")
        return

    print("Files are different.")
    print(f"{file1} size: {len(data1)} bytes")
    print(f"{file2} size: {len(data2)} bytes")

    min_size = min(len(data1), len(data2))

    for i in range(min_size):
        if data1[i] != data2[i]:
            print(f"\nFirst difference at byte {i}")
            print(f"{file1}: {data1[i]:02X}")
            print(f"{file2}: {data2[i]:02X}")
            break

    if len(data1) != len(data2):
        print(f"\nSize difference: {abs(len(data1) - len(data2))} bytes")


compare_files("mbfile", "mbfile2")