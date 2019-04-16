import time
from multiprocessing.pool import Pool
from multiprocessing.context import Process
from multiprocessing import Queue


def func1(x, q):
    time.sleep(5)
    x = ('sfsd', 21431, [3123, 123, (23214, 2)])
    q.put(x)


def func2(x, q):
    time.sleep(2)
    x = ('sfsd', 45343625463, [3123, [21312], 123, (23214, 2)])
    q.put(x)

def func3(l, q):
    with Pool(processes=4) as pool:
        time.sleep(3)
        z = pool.map(runner, range(3))
        pool.join()
    l.append(z)
    q.put(l)

def runner(x):
    for i in range(x):
        x = x ** 2
    return x


if __name__ == "__main__":
    # multiprocessing test
    q = Queue()
    p1 = Process(name="a", target=func1, args=((4, 2), q))
    p2 = Process(name="b", target=func2, args=([23, 12, 3], q))
    # p3 = Process(name="c", target=func3, args=([23, 12, 3], q))

    procs = [p1, p2]
    results = []
    
    for i in procs:
        i.start()

    while len(results) < len(procs):
        print(f"results: {len(results)} procs: {len(procs)} cond: {len(results) < len(procs)}")
        for proc in procs:
            proc.join(2)
            result = q.get()
            results.append(result)
            print(proc.name, result)
        print('for loop finished')
    print('while loop finished')
