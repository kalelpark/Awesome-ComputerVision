import sys
input = sys.stdin.readline

tree_count, height = map(int, input().split())
tree_array = list(map(int, input().split()))
# 최댓값, 최솟값은 분기점을 만들어 주어야 한다.
# 분기점을 만들어서 그 값을 넘어가면 함수가 활용이 안되고, 전에 사용했던 값을 사용할 수 있어야 한다.
start, end, section = 0 , max(tree_array), 0

while start <= end:
    mid = (start + end) // 2
    
    tree_length = 0
    # for i in tree_array:
    #     if i > mid:
    #         tree_length += i - mid
    # 위의 반복문은 계산이 오래걸린다. 그러므로, 배열에서 mid 보다 큰것만을 출력하는 것으로 바꾸는 것이 좋다.
    # 아래가 더 빠르다
    tree_length = sum(i-mid if i - mid > 0 else 0 for i in tree_array)

    if tree_length >= height:
        section = mid            
        # 구하고자 할 때 사용하는 것을 사용해야 한다
        start = mid + 1
    else:
        end = mid - 1 

print(section)