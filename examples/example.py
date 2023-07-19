# pip install cudagrad; python ./examples/example.py
import cudagrad as cg

a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0])
b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0])
c = cg.tensor([2, 2], [10.0, 10.0, 10.0, 10.0])
d = cg.tensor([2, 2], [11.0, 11.0, 11.0, 11.0])
e = ((a @ b) + c) * d
f = e.sum()
f.backward()

print(f.data) # [2794.0]
print(f.size) # [1]
print(a.grad) # [143.0, 187.0, 143.0, 187.0]
print(b.grad) # [66.0, 66.0, 88.0, 88.0]
