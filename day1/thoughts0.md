LLM inference has **two fundamentally different phases**:

1. **Prefill** (also called *prompt processing*)
2. **Decode** (also called *generation* or *autoregressive decoding*)

Understanding these two phases is essential because almost every inference optimization (vLLM, TensorRT-LLM, paged attention, continuous batching, speculative decoding, FlashAttention, KV Cache, etc.) is designed around them.

---

# High-Level Analogy

Imagine a student taking an exam.

**Question:**

> "Explain why the sky appears blue."

The student:

1. **Reads the entire question** → Prefill
2. **Writes the answer one word at a time** → Decode

The reading phase happens once.

The writing phase repeats until the answer is complete.

LLMs work exactly the same way.

---

# Example

Prompt:

```
Translate to French:

I love artificial intelligence because
```

Tokenized:

```
[I]
[love]
[artificial]
[intelligence]
[because]
```

Suppose there are 5 prompt tokens.

The model generates

```
j'
adore
l'
intelligence
artificielle
...
```

---

# Phase 1 — Prefill

Also called

* Prompt processing
* Context encoding
* Initial forward pass

The entire prompt is processed simultaneously.

Instead of generating words, the model computes internal representations.

Input:

```
Token1
Token2
Token3
Token4
Token5
```

Transformer receives

```
5 tokens
```

at once.

Self-attention looks like

```
        T1 T2 T3 T4 T5

T1       ✓
T2       ✓ ✓
T3       ✓ ✓ ✓
T4       ✓ ✓ ✓ ✓
T5       ✓ ✓ ✓ ✓ ✓
```

Every token attends to all previous tokens.

This is possible because the entire prompt is already known.

---

# What Happens During Prefill?

For every transformer layer:

```
Embedding

↓

Attention

↓

Feed Forward

↓

LayerNorm

↓

Next Layer
```

This happens across all prompt tokens.

Suppose

Prompt length = 2000 tokens

Hidden size = 4096

Layers = 32

The GPU computes

```
2000 tokens

×

32 layers

×

4096 dimensions
```

This is a huge matrix computation.

GPUs excel at this.

---

# Output of Prefill

The important output is **the KV Cache**.

For every token and every transformer layer the model stores:

```
Key vectors

Value vectors
```

Example

```
Layer 1

Token1 → K,V
Token2 → K,V
Token3 → K,V
...

Layer 32

Token1 → K,V
Token2 → K,V
...
```

These are cached.

Without caching we'd recompute them every generation step.

---

# Why Prefill is Expensive

Complexity grows roughly as

```
O(n²)
```

because every prompt token attends to every other previous token.

For

```
10 tokens
```

Attention matrix

```
10 × 10
```

For

```
1000 tokens
```

Attention matrix

```
1000 × 1000
```

Very expensive.

---

# GPU Utilization During Prefill

GPU utilization is very high.

Reason:

Large matrix multiplications.

Example

```
4096 × 4096

8192 × 4096

16384 × 4096
```

Tensor cores become fully utilized.

So

```
Prefill

↓

High throughput

↓

GPU Compute Bound
```

---

# Phase 2 — Decode

Now the prompt is finished.

The model predicts

only

```
ONE

NEW

TOKEN
```

Suppose next token is

```
"j'"
```

The model outputs

```
j'
```

Now prompt becomes

```
Prompt

+

j'
```

Next step

Model predicts

```
adore
```

Then

```
Prompt

+

j'

+

adore
```

Repeat.

---

# Decode Loop

```
while not EOS:

    predict next token

    append token

    update KV cache
```

Only one token is generated per iteration.

---

# Why Decode is Different

Instead of processing

```
2000 tokens
```

the model processes

```
ONLY

1 NEW TOKEN
```

The previous tokens are already represented in the KV cache.

Only the new token's Query, Key, and Value vectors are computed.

The Query attends to all cached Keys, producing attention over the entire context, while the new Key and Value are appended to the cache for future steps.

---

# Attention During Decode

Suppose

Prompt length

```
1000 tokens
```

Already cached.

Generate token 1001.

Attention becomes

```
Q(new token)

↓

attends to

K1

K2

K3

...

K1000
```

No need to recompute

```
K1

K2

...

K1000
```

Huge savings.

---

# Decode Complexity

Per generated token, attention scales approximately as

```
O(n)
```

with respect to context length because the new query compares against all cached keys.

However, generating an entire sequence still grows with both prompt length and output length.

---

# GPU Utilization During Decode

This surprises many engineers.

GPU utilization often drops.

Why?

Instead of large matrix multiplications

```
4096 × 4096
```

you're doing much smaller operations

```
1 × 4096
```

The GPU spends much of its time waiting on memory accesses to fetch the KV cache.

So decode is often **memory-bandwidth bound**, whereas prefill is typically **compute-bound**.

---

# Timeline

```
Prompt

↓

Prefill

↓

KV Cache Created

↓

Generate Token 1

↓

Update KV Cache

↓

Generate Token 2

↓

Update KV Cache

↓

Generate Token 3

↓

...
```

---

# Visualization

```
                PROMPT

Token1
Token2
Token3
Token4
Token5

        │
        ▼

====================
PREFILL
====================

All tokens processed together

↓

KV Cache

↓

====================
DECODE
====================

Generate Token6

↓

Append KV

↓

Generate Token7

↓

Append KV

↓

Generate Token8

↓

...
```

---

# Latency Characteristics

There are two common latency metrics:

* **Time to First Token (TTFT):** Dominated by the prefill phase. Longer prompts generally increase TTFT because the model must process the entire prompt before it can generate anything.
* **Time Per Output Token (TPOT):** Dominated by the decode phase. This measures how quickly subsequent tokens are generated.

For example:

```
Prompt: 4,000 tokens
Output: 100 tokens

Prefill:   800 ms
First token appears

Decode:
100 tokens × 20 ms/token = 2,000 ms

Total ≈ 2.8 s
```

---

# Why Modern Inference Systems Optimize Both

Different optimizations target different phases:

| Optimization         | Primarily Helps             | Why                                                                                                  |
| -------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------- |
| FlashAttention       | Prefill (also helps decode) | Makes attention computation more efficient and reduces memory traffic.                               |
| KV Cache             | Decode                      | Avoids recomputing keys and values for previous tokens.                                              |
| PagedAttention       | Decode                      | Manages large KV caches efficiently across many concurrent requests.                                 |
| Continuous batching  | Both                        | Keeps the GPU busy by mixing requests in prefill and decode stages.                                  |
| Speculative decoding | Decode                      | Reduces the effective cost per generated token by accepting multiple predicted tokens when possible. |
| Prefix caching       | Prefill                     | Reuses computation for identical prompt prefixes across requests.                                    |

---

# Key Takeaways

* **Prefill** processes the entire prompt in parallel, creates the KV cache, is dominated by attention over the prompt, is typically **compute-bound**, and largely determines **Time to First Token (TTFT)**.
* **Decode** generates **one token at a time**, reuses the KV cache, appends new key/value pairs after each generated token, is typically **memory-bandwidth bound**, and determines **Time Per Output Token (TPOT)**.
* Efficient LLM serving systems are designed to optimize **both phases**, since improving only one still leaves the other as a bottleneck.
