# Demystifying Transformers: Answering Common Doubts

Transformers, the powerful architecture behind models like GPT<sup>[2]</sup> and BERT<sup>[3]</sup>, can be quite challenging to understand in depth. As I was learning about Transformers and the "Attention Is All You Need" paper<sup>[1]</sup>, I encountered several doubts. This post compiles those questions and answers for others who may be facing similar challenges.

---

## 1. **How do Transformers solve the problem of long sequences, compared to RNNs?**
Transformers solve the long sequence problem by using **self-attention**, which allows direct connections between all tokens in a sequence. Unlike RNNs, which need to process sequences token-by-token, Transformers can process the entire sequence in **parallel**. This eliminates the vanishing gradient issues and allows **global context** to be captured more effectively, even for long sequences. <sup>[4]</sup>

---

## 2. **How are Key, Query, and Value vectors calculated for each word embedding?**
The Key, Query, and Value vectors are calculated by applying **learnable linear transformations** to each word embedding. For each word (represented as a vector of dimension `d_model`), we apply a linear transformation using weight matrices \( W_Q \), \( W_K \), and \( W_V \) to get the query, key, and value vectors, respectively. These transformations project the embedding to different subspaces that the attention mechanism uses. <sup>[5]</sup>

---

## 3. **Are these weight matrices learnable? How do linear transformations learn?**
Yes, the weight matrices for the Key, Query, and Value vectors are **learnable parameters**. These matrices are optimized during training using **backpropagation**, like any neural network. Linear transformations in deep learning "learn" by adjusting their weights to minimize the loss function, ensuring that the model captures meaningful relationships between the input tokens. <sup>[6]</sup>

---

## 4. **If embedding dimension is 512 and Key/Query/Value dimension is 64, do we use neural networks for the transformation?**
Yes, it's essentially a linear layer. If the embedding dimension is 512 and the desired dimension for the Key/Query/Value is 64, we apply three **linear projections** (neural networks without activation functions) that map the 512-dimensional embeddings to 64-dimensional query, key, and value vectors.

---

## 5. **Why do we need multiple attention heads?**
Multiple heads in **multi-head attention** allow the model to capture **different types of relationships** between words. Each attention head can focus on different aspects of the sequence—one might focus on **syntactic structure**, while another might capture **semantic relations**. This gives the model a more nuanced understanding of the input by looking at it from different perspectives simultaneously.

---

## 6. **If multiple heads capture different information, why don’t they end up learning the same thing during training?**
This is an interesting question! While multiple heads could theoretically converge to the same result, in practice they don’t due to **random initialization** and **regularization effects** during training. Moreover, having multiple heads with independent weight matrices encourages the model to explore **different attention patterns**, even though they’re trained on the same data.

---

## 7. **Why not increase the number of heads instead of stacking attention layers?**
While increasing the number of attention heads helps capture diverse features in **parallel**, stacking multiple layers allows for a **hierarchical representation** of the input. Each layer refines and builds upon the previous layer’s representation, enabling deeper understanding. Think of it like deep neural networks: deeper models can capture more complex patterns than shallow models with wide layers.

---

## 8. **What’s the purpose of positional encoding?**
Transformers process sequences in parallel, so they need a way to encode the **order** of the tokens in the sequence. Positional encodings introduce this ordering by adding a **position-dependent signal** to each word embedding. This allows the model to know, for example, that the first word comes before the second one, and so on.

---

## 9. **Why use sine-cosine for positional encoding instead of simple integers?**
Sine and cosine functions are continuous and allow the model to generalize to **sequences longer than those seen during training**. Unlike simple integers, sine and cosine provide a smooth representation of positions, making it easier for the model to **interpolate** between positions. They also create unique positional encodings for each dimension, which helps with capturing complex positional relationships.

---

## 10. **Can the number of encoders and decoders be different in a Transformer?**
Yes, the number of encoder and decoder layers in a Transformer doesn’t have to be the same. In the original Transformer architecture, both the encoder and decoder typically have the same number of layers (e.g., 6 each), but this is a design choice and can be adjusted based on the task or resource constraints.

---

## 11. **How does masked attention work in the decoder?**
In the decoder, **masked attention** ensures that when predicting a token at position \( t \), the model only attends to previous tokens (i.e., tokens before or at position \( t \)). This is crucial for tasks like text generation where the model generates tokens one by one. The masking prevents the model from “seeing” future tokens during training, enforcing an **autoregressive** property.

---

## 12. **Is there only one masked attention layer in the decoder or can there be multiple?**
Each **decoder layer** contains one **masked self-attention** mechanism. However, since the Transformer decoder is typically made up of multiple layers (e.g., 6 layers), you end up with **multiple masked attention layers**—one in each decoder layer. These layers work together to generate the output sequence token-by-token.

---

## 13. **How is the output of the encoder combined with the output of the previous layer in the decoder?**
The **output of the encoder** is combined with the decoder's previous layer output using the **encoder-decoder attention mechanism**. Here, the **queries** come from the decoder's previous layer, and the **keys and values** come from the encoder's final output. This allows the decoder to attend to relevant information from the encoder (input sequence) while generating each token in the output.

---

## 14. **Why is the Transformer called an autoregressive model?**
The Transformer decoder is **autoregressive** because it generates tokens **one at a time**, with each token prediction depending on previously generated tokens. This is enforced by the **masked attention** mechanism, which ensures that at each step, the model can only access tokens generated before the current position, making the generation process **sequential**.

---

Feel free to reach out or contribute to the discussion if you have more questions!

---

### References:
1. Vaswani et al. (2017). ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
2. Radford et al. (2018). ["Improving Language Understanding
by Generative Pre-Training"](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
3. Devlin et al., NAACL 2019 ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805)
4. [Answer](https://ai.stackexchange.com/q/20075/51861) on stack-exchange
5. [Answer](https://stats.stackexchange.com/a/463320/333903) on stack-exchange
6. [Answer](https://stats.stackexchange.com/a/626483/333903) on stack-exchange
