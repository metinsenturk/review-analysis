import time
from multiprocessing import Process, Queue


def func1(x, q):
    time.sleep(10)
    x = ('sfsd', 21431, [3123, 123, (23214, 2)])
    q.put(x)


def func2(x, q):
    time.sleep(5)
    x = ('sfsd', 45343625463, [3123, [21312], 123, (23214, 2)])
    q.put(x)


if __name__ == "__main__":
    q = Queue()
    p1 = Process(name="a", target=func1, args=((4, 2), q))
    p2 = Process(name="b", target=func2, args=([23, 12, 3], q))

    for i in [p1, p2]:
        i.start()

    p1.join()
    print(p1.name)
    print(q.get())
    # /Users/owl/.vscode/extensions/ms-python.python-2019.3.6139/pythonFiles/lib/python/ptvsd/daemon.py

    p2.join()
    print(p2.name)
    print(q.get())
