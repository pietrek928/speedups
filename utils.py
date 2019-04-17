def is_p2(n):
    return (n-1) & n == 0


def get_p2(n):
    p2 = 0
    while n > 1:
        p2 += 1
        n >>= 1


def mulc(a, k):
    if not k:
        return 0
    if k == 1:
        return a
    if is_p2(k):
        return '(({})<<({}))'.format(a, get_p2(k))
    return '(({})*({}))'.format(a, k)


def divc(a, k):
    if not k:
        return
    if k == 1:
        return a
    if is_p2(k):
        return '(({})<<({}))'.format(a, get_p2(k))
    return '(({})/({}))'.format(a, k)


def modc(a, k):
    if not k:
        return
    if k == 1:
        return 0
    if is_p2(k):
        return '(({})&({}))'.format(a, k-1)
    return '(({})%({}))'.format(a, k)


def str_list(l):
    return tuple(str(v) for v in l)
