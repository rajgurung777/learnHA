from typing import List


def permut(n: int, k: int) -> List[List[int]]:
    """
    Author: Amit Gurung
    :param self:
    :param n: the power of the multinomial
    :param k: the number of terms/variables in the multinomial
    :return: All the coefficients of the multinomial expansion of the form (a+b+c+ ... + k)^n
    """
    res = []
    myres = []

    def backtrack(start, comb):
        sumEle = 0
        for x in range(0, len(comb)):
            sumEle += comb[x]

        if len(comb) == k and sumEle == n:
            # myres.append(comb.copy())   # appends at the end
            myres.insert(0, comb.copy())  # appends at the start
            # print(myres)
        if len(comb) == k:
            res.append(comb.copy())
            return
        # else:
        #     pass

        for i in range(start, n + 1):  # n+1 because for-in-range in python does not include 'end' in range(0,end)
            comb.append(i)
            backtrack(start, comb)  # everytime the combination starts from 0
            comb.pop()

    backtrack(0, [])    # starts from 0
    # return res    # returns all the list without the constrain such that r1+r2+...+rk == n
    return myres  # returns all the list that satisfy the constrain such that r1+r2+...+rk == n


def factorial(i):
    if i == 0:
        return 1
    fact = 1
    for x in range(1, i + 1):
        fact *= x
    return fact


def compute_coeff(list_data):
    """
    Author: Amit Gurung
    :param list_data:
    :return: coefficient using the formula n!/(r1! * r2! * ... * rk!). where r1+r2+...+rk == n
    """
    sum_n = 0
    fact_powers = 1
    for i in list_data:
        sum_n += i
        fact_powers *= factorial(i)
    coefficient_val = factorial(sum_n) / fact_powers
    return coefficient_val


def multinomial(vars, powers):
    """
    Author: Amit Gurung
    :param vars: number of terms or variables
    :param powers: highest power
    :return: All the list of combinations/expansion of Multinomial along with the computed Coefficient of each term
     [coeff: double, followed by expansion_list]
    """
    comb_list = permut(powers, vars)
    # print(comb_list)
    combine_list = []
    for data in comb_list:
        val = compute_coeff(data)
        combine_list.append([val] + data)
        # print("Coefficient = ", val)
        # print(combine_list)

    return combine_list


if __name__ == "__main__":
    dim = 3+1
    degree = 2
    combine_list = multinomial(dim,degree)
    print(combine_list)