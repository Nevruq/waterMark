Watermarking AI-Generated Text via Logit Modification

This repository explores methods for embedding statistical watermarks into AI-generated text by modifying the model’s logits during decoding. The goal is to enable retrospective, statistical verification that a given text was produced using a known watermarking scheme, without significantly degrading text quality.

Overview

Watermarking through logit adjustment is a technique in which the probability distribution over the next token is subtly biased according to a secret key or deterministic rule. These biases introduce hidden statistical signals that can later be detected, even if the text has been copied, reformatted, or distributed without metadata.

This repository demonstrates:

How to apply a watermarking function to logits before sampling.

How to generate text containing a reproducible statistical watermark.

How to perform retrospective verification on arbitrary text samples.

How to evaluate detection accuracy, false-positive rates, and robustness.

Motivation

As large language models become widely used, it becomes increasingly important to identify the origin of generated content. Logit-based watermarking provides a mechanism for:

verifying whether text was produced by an AI system,

establishing content provenance,

supporting responsible deployment and monitoring.

Unlike metadata-based approaches, logit watermarking remains attached to the text content itself.

Methodology
Logit Biasing and Vocabulary Partitioning

A common approach divides the model’s vocabulary into two dynamic sets, often referred to as a "greenlist" and a "redlist." This partition is computed from:

a secret key,

the current context,

a hash function mapping tokens to deterministic groups.

During generation, the logits of tokens in the preferred set are increased by a small bias. This results in:

slightly higher probability for greenlist tokens,

statistically detectable patterns in the final text.

The method is compatible with sampling, temperature scaling, and other non-deterministic decoding strategies.

Retrospective Detection

Given a piece of text, the detection process consists of:

Recomputing the greenlist sequence using the same secret key and hashing process.

Computing how many tokens in the sample fall into the biased set.

Calculating a statistical score (such as a z-score or log-likelihood ratio).

Comparing the score to a predefined threshold.

A significantly higher-than-expected alignment with the biased set indicates that the text is watermarked.
