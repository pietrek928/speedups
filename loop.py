def array_op(**decor_args):
    def decor(func):
        def wrapper():
            input_arrays = {}
            params = {}
            for n, v in decor_args.items():
                if isinstance(v, ArrayArg):
                    input_arrays[n] =
                else:
                    params[n] = v
            sgn = '_'.join('{}.{}'.format(k, v) for k, v in sorted(params.items()))
            func(**input_arrays)
        return wrapper
    return decor

def gen_iter():
    pass
