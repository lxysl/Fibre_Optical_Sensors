import time


def time_calc(func):
    """
    calc:calculation 计算
    定义一个装饰器函数“用来计算时间”
    *args    位置参数 Positional Arguments
    **kargs 关键字参数 Keyword Arguments
    这个装饰器函数接收任何的关键字和位置参数，方便让这个装饰器函数计算任何函数的时间
    wrapper:包装材料；包装器   
    decorators：装饰器
    时间戳是一个浮点数，代表自 Unix 时间戳纪元（通常是 1970 年 1 月 1 日午夜 UTC）至今的秒数，包括小数部分，精确到秒以下的时间
    """
    def wrapper(*args, **kargs):
        start_time = time.time()
        f = func(*args, **kargs)
        print('{}: {:.2f} s'.format(func.__name__, time.time() - start_time))
        return f
    return wrapper