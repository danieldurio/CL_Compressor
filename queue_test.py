import queue

q0 = queue.Queue(maxsize=0)
print(f"Queue(0) maxsize: {q0.maxsize} (Expected: 0 or infinite)")
try:
    for i in range(10):
        q0.put(i, block=False)
    print("Queue(0) accepted 10 items without blocking")
except queue.Full:
    print("Queue(0) blocked")

q1 = queue.Queue(maxsize=1)
print(f"Queue(1) maxsize: {q1.maxsize}")
try:
    q1.put(1, block=False)
    print("Queue(1) accepted 1 item")
    q1.put(2, block=False)
    print("Queue(1) accepted 2 items")
except queue.Full:
    print("Queue(1) blocked on 2nd item")
