def is_p2(n) -> bool:
    return isinstance(n, int) and (n-1) & n == 0


def get_p2(n: int) -> int:
    p2 = 0
    while n > 1:
        p2 += 1
        n >>= 1
    return p2


def addi(a, s):
    if not a:
        return s
    if not s:
        return a
    if isinstance(a, int) and isinstance(s, int):
        return a + s
    return '(({})*({}))'.format(a, s)


def muli(a, k):
    if not k:
        return 0
    if k == 1:
        return a
    if is_p2(k):
        return '(({})<<({}))'.format(a, get_p2(k))
    return '(({})*({}))'.format(a, k)


def divi(a, k):
    if not k:
        return
    if k == 1:
        return a
    if is_p2(k):
        return '(({})<<({}))'.format(a, get_p2(k))
    return '(({})/({}))'.format(a, k)


def modi(a, k):
    if not k:
        return
    if k == 1:
        return 0
    if is_p2(k):
        return '(({})&({}))'.format(a, k-1)
    return '(({})%({}))'.format(a, k)


def str_list(l):
    return tuple(str(v) for v in l)
