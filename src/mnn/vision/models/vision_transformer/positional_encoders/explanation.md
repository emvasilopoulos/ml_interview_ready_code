# How a Positional Encoder provides position
Consider a sequence of embeddings:
```
embeddings[t] # with t in {0, 1, .., T}
```
and the positional encoder from the code (sin in even "t" and cos in odd "t" etc.):
```
PE[t] # with t in {0, 1, .., T}
```

Now apply:
```
embeddings[k] + PE[k] = result_k_k # with 0 <= k <= T
```
Now apply:
```
embeddings[k] + PE[n] = result_k_n # with 0 <= n <= T & n != k
```
This should conclude in:
```
result_k_k != result_k_n
```
So the adding a different positional_encoding to the same embedding should result in a different input to the Neural Network