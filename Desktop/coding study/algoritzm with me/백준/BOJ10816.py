from bisect import bisect_left, bisect_right
import sys
num1 = int(sys.stdin.readline())
array1 = list(map(int, sys.stdin.readline().split()))
num2 = int(input())
array2 = list(map(int, sys.stdin.readline().split()))

array1 = sorted(array1)

for i in array2:
	a = 0
	start = 0
	end = len(array1)-1
	while start <= end:
		mid = (start + end) // 2

		if array1[mid] == i:
			print(bisect_right(array1,i) - bisect_left(array1,i), end=' ')
			a = 1
			break
			
		elif array1[mid] > i:
			end = mid-1
		else:
			start = mid + 1
	if a == 0:
		print(0, end=' ')