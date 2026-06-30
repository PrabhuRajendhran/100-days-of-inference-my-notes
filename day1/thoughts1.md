# LLM Inference Deep Dive — Prefill, Decode & Attention Architecture

*A consolidated reference from today's discussion*

---

# 1. Big Picture

An LLM performs inference in **two completely different phases**.

```text
                 Prompt
                    │
                    ▼
            ==================
               PREFILL
            ==================
                    │
             Build KV Cache
                    │
                    ▼
            ==================
                DECODE
            ==================
                    │
        Generate One Token at a Time
```

Think of it like a student.

```
Read the entire question  →  Prefill

Write the answer          →  Decode
```

The reading happens once.

The writing happens one word at a time.

---

# 2. Prefill

Suppose the prompt is

```
"I love AI because it helps."
```

Tokenizer

```
T1  T2  T3  T4  T5  T6
```

The entire prompt enters the transformer.

```
                T1
                T2
                T3
                T4
                T5
                T6
                  │
                  ▼
             Transformer
```

Everything happens **before the first output token**.

---

## What happens?

Every transformer layer processes

```
ALL TOKENS
```

simultaneously.

Example

```
Prompt length = 2048 tokens

Hidden size = 4096

Input Matrix

2048 × 4096
```

This matrix flows through Layer 1.

---

# 3. Parallelism During Prefill

One of the biggest misconceptions.

## Are all tokens processed together?

YES.

## Are all layers processed together?

NO.

The execution is

```
Layer1

All 2048 tokens
processed together

↓

Layer2

All 2048 tokens
processed together

↓

Layer3

All 2048 tokens
processed together

↓

...

↓

Layer32
```

So

```
Parallel across Tokens

Sequential across Layers
```

---

# Why?

Layer2 needs the output of Layer1.

```
H1 = Layer1(X)

H2 = Layer2(H1)

H3 = Layer3(H2)
```

Layer3 cannot begin before Layer2 finishes.

---

# GPU View

GPU doesn't see

```
Token1

Token2

Token3
```

It sees

```
2048 × 4096
```

Everything becomes matrix multiplication.

---

# 4. Zoom into One Transformer Layer

One transformer layer consists of

```
Input

↓

Q,K,V Projection

↓

Attention

↓

Output Projection

↓

Residual

↓

LayerNorm

↓

MLP

↓

Residual

↓

LayerNorm

↓

Output
```

Let's understand every block.

---

# 5. Input

Suppose

```
3 tokens

Hidden size = 4
```

Input matrix

| Token | Vector  |
| ----- | ------- |
| T1    | 1 2 3 4 |
| T2    | 2 3 1 5 |
| T3    | 4 1 2 3 |

Shape

```
3 × 4
```

---

# 6. Q,K,V Projection

Learned matrices

```
WQ

WK

WV
```

Conceptually

```
Q = XWQ

K = XWK

V = XWV
```

---

GPU computes

```
3×4

×

4×4

↓

3×4
```

Entire matrix.

NOT

```
Token1

then

Token2

then

Token3
```

---

# 7. Attention

Now

```
Q

×

KT
```

Example

```
      T1 T2 T3

T1    ✓

T2    ✓ ✓

T3    ✓ ✓ ✓
```

Future tokens are masked.

```
Before

5 2 1

↓

After Mask

5 -∞ -∞
```

Then

```
Softmax

↓

Attention Probabilities

↓

Weighted Sum of Values
```

Output

```
Context Matrix

3 × 4
```

---

# 8. Multi-Head Attention

Suppose

```
Hidden size = 4096

Heads = 8
```

Many beginners imagine

```
Head1 gets first 512 dims

Head2 gets next 512 dims
```

This is WRONG.

---

## What actually happens?

Every head receives the FULL input.

```
              Token

4096 dimensions

      │
      ├────────► Head1

      ├────────► Head2

      ├────────► Head3

      ├────────► ...

      └────────► Head8
```

Every head sees the SAME token.

---

# Why?

Each head learns a different projection.

Imagine

```
Medical Report
```

8 doctors.

Do you give

Doctor1

only page1?

No.

Every doctor reads the FULL report.

Each focuses on something different.

Exactly the same idea.

---

# 9. Dimensions of WQ WK WV

Suppose

```
d_model = 4096

Heads = 8

head_dim = 512
```

Conceptually

Every head has

```
WQ

4096 × 512
```

Same for

```
WK

WV
```

---

Framework implementation

Instead of storing

```
8 separate matrices
```

they concatenate them.

```
WQ

4096 × 4096

WK

4096 × 4096

WV

4096 × 4096
```

GPU performs

```
2048×4096

×

4096×4096

↓

2048×4096
```

Then reshapes

```
2048×4096

↓

2048×8×512
```

---

# 10. Attention Heads

Each head computes

```
2048 × 512
```

Suppose

```
8 heads
```

Outputs

```
Head1

2048×512

Head2

2048×512

...

Head8

2048×512
```

---

# 11. Concatenation

Concatenate

ALONG FEATURE DIMENSION

NOT TOKEN DIMENSION.

```
Head1

2048×512

+

Head2

2048×512

+

...

↓

2048×4096
```

Row-wise

```
Head1

[a b]

Head2

[c d]

↓

[a b c d]
```

---

# 12. Output Projection (Wₒ)

Now

```
2048×4096

↓

WO

↓

2048×4096
```

WO

```
4096×4096
```

---

## Why?

Concatenation merely places the heads side by side.

```
[Grammar]

[Syntax]

[Coreference]

[Long Context]
```

Wₒ learns

how to mix these different viewpoints into one coherent representation.

Think of it as an editor combining reports from multiple specialists.

---

# 13. Residual Connection

Instead of

```
Output = Attention
```

we do

```
Output

=

Attention

+

Original Input
```

Why?

If attention makes a poor update,

the original information is still preserved.

The layer learns

```
"What should I change?"

instead of

"Rebuild everything."
```

---

# 14. LayerNorm

Different tokens can have wildly different scales.

Example

```
5000

-3000

8000

100
```

LayerNorm normalizes each token independently.

Purpose

```
Stable activations

↓

Stable training

↓

Deep networks work
```

---

# 15. MLP

Often underestimated.

Attention answers

```
Who should I look at?
```

MLP answers

```
Now that I have that information,

what should I compute?
```

Example

```
4096

↓

16384

↓

4096
```

The expansion gives the model more capacity to learn complex, non-linear transformations before compressing back to the original hidden size.

---

# 16. Why Expand?

Imagine solving a complex problem.

Instead of using

```
One sheet
```

you use

```
A huge whiteboard
```

Then summarize.

That's exactly

```
4096

↓

16384

↓

4096
```

---

# 17. Decode Phase

After prefill,

KV cache exists.

Now

ONLY

ONE

NEW TOKEN

is processed.

```
Prompt

↓

Generate Token1

↓

Append KV

↓

Generate Token2

↓

Append KV
```

Loop

```
while not EOS

Predict next token

Append KV
```

---

# 18. Why KV Cache?

Without cache

every generation step would recompute

```
ALL PREVIOUS TOKENS
```

Again.

With cache

old K,V are reused.

Only

```
New Query

New Key

New Value
```

are computed.

---

# 19. Why Decode is Slower

Prefill

```
2048 × 4096
```

Large GEMMs.

GPU loves this.

Compute Bound.

---

Decode

```
1 × 4096
```

Very small GEMMs.

GPU spends much of its time reading the KV cache from memory.

Memory Bandwidth Bound.

---

# 20. Attention Variants

## Multi-Head Attention (MHA)

Every head has

its own

```
Q

K

V
```

```
Q1 K1 V1

Q2 K2 V2

...

Q8 K8 V8
```

Largest KV cache.

Best flexibility.

---

## Multi-Query Attention (MQA)

Queries remain separate.

Keys

shared.

Values

shared.

```
Q1

Q2

...

Q8

↓

Shared K

Shared V
```

KV cache

8×

smaller.

Excellent inference speed.

Slight quality tradeoff.

---

## Grouped Query Attention (GQA)

Compromise.

Example

```
8 Query Heads

2 KV Groups
```

```
Q1

Q2

Q3

Q4

↓

KV Group1



Q5

Q6

Q7

Q8

↓

KV Group2
```

Memory

4×

smaller.

Quality almost equal to MHA.

Most modern LLMs use this.

---

## Multi-Head Latent Attention (MLA)

Instead of storing full Keys/Values,

compress them first.

```
Token

↓

Latent Representation

↓

Cache

↓

Reconstruct K/V
```

Much smaller KV cache.

Designed for extremely long contexts.

---

# 21. Comparison

| Architecture | Query Heads |             KV Heads | KV Cache | Quality | Inference                  |
| ------------ | ----------: | -------------------: | -------: | ------- | -------------------------- |
| MHA          |           8 |                    8 |  Largest | ⭐⭐⭐⭐⭐   | Slowest                    |
| GQA          |           8 |                  2–4 |   Medium | ⭐⭐⭐⭐☆   | Fast                       |
| MQA          |           8 |                    1 |    Small | ⭐⭐⭐⭐    | Faster                     |
| MLA          |    Multiple | Compressed latent KV | Smallest | ⭐⭐⭐⭐⭐   | Excellent for long context |

---

# 22. The Most Important Mental Models

## A Transformer Layer

```
Attention
    │
    ▼
Move information
between tokens

------------------

MLP
    │
    ▼
Transform information
within each token

------------------

Residual
    │
    ▼
Preserve previous knowledge

------------------

LayerNorm
    │
    ▼
Keep activations stable

------------------

Output Projection
    │
    ▼
Merge the perspectives
from multiple heads
```

---

## Parallelism

```
Within a Layer

Tokens

████████████████

Processed Together

↓

Next Layer

████████████████

Processed Together

↓

Next Layer
```

**Parallel across tokens.**

**Sequential across layers.**

---

## Attention Heads

```
Same Input

      │
      ├────► Head1
      ├────► Head2
      ├────► Head3
      ├────► Head4
      └────► ...

Different Learned Projections

↓

Different Views

↓

Concatenate

↓

Output Projection

↓

One Unified Representation
```

---

# 23. Final Takeaways

1. **Prefill** processes the entire prompt in parallel, creates the KV cache, and is compute-bound.
2. **Decode** generates one token at a time using the KV cache and is memory-bandwidth bound.
3. **Within a transformer layer, tokens are processed in parallel, but layers execute sequentially.**
4. **Attention** decides **where to gather information** from other tokens.
5. **The MLP** decides **how to transform and reason over** that information.
6. **Every attention head receives the full token representation** and learns a different projection of it; heads are **not** created by splitting the input features.
7. **Concatenation** stacks head outputs along the feature dimension, and **`Wₒ`** learns how to fuse those different perspectives into one representation.
8. **Residual connections** preserve information and make deep networks trainable, while **LayerNorm** keeps activations stable.
9. **MHA, GQA, MQA, and MLA** all keep the same overall attention mechanism, but differ in how Keys/Values are shared or compressed to reduce KV-cache memory and improve inference efficiency.

This mental model forms the foundation for understanding advanced inference engines such as **vLLM**, **TensorRT-LLM**, **FlashAttention**, **PagedAttention**, **continuous batching**, and **speculative decoding**, because almost all of their optimizations are built around improving either the **prefill** phase, the **decode** phase, or the **KV-cache**.
