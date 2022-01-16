inp_str = "Python000423Journaldev"

print("Original String : " + inp_str)
num = ""
for c in inp_str:
    if c.isdigit():
        num = num + c
print("Extracted numbers from the list : " + num)
print(int(num))
