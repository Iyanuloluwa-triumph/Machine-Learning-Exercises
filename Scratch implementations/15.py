__author__ = "Iyanucodez"


def two_sum(numbers, target):
    m = 0
    solution = []
    for i in range(len(numbers)):
        for m in range(len(numbers)):
            if numbers[i] + numbers[m] == target and i != m:
                solution.append(i)
                solution.append(m)
                return tuple(solution)
            else:
                pass


print(two_sum([1, 2, 3], 5))


def dig_pow(n, p):
    sol = []
    for item in str(n):
        sol.append(int(item) ** p)
        p += 1
    new_item = sum(sol)
    final = new_item / n
    if new_item % n == 0:
        return final

    return -1


print(dig_pow(46288, 3))
