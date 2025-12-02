# Watermarking AI-Generated Text via Logit Adjustment

This repository provides an implementation and explanation of how to watermark AI-generated text by modifying token logits during generation. The goal is to embed a statistical signal that can be detected retrospectively, allowing verifiable proof that a text was produced with a watermark-enabled model while preserving fluency and semantic quality.

## Motivation

As AI-generated text becomes increasingly widespread, reliable attribution mechanisms are essential. Cryptographic or metadata-based tagging methods are easily removed or lost. This project focuses on statistical watermarking, where the watermark is embedded directly into the generative process itself.  
Because the watermark influences token selection probabilities rather than the text surface form, it remains robust to copying, formatting changes, or minor edits.

## Core Idea

The method operates by adjusting logits during sampling. A pseudorandom function (PRF), seeded with a secret key, assigns each possible token to either a preferred or non-preferred subset for each generation step.

During sampling:
- Logits of preferred tokens are increased by a small bias.
- Logits of non-preferred tokens remain unchanged.
- The model samples from the modified distribution as usual.

This adjustment results in text that statistically favors the preferred token subset. With knowledge of the secret key, detectors can test text for the expected statistical pattern and determine whether it was generated using the watermark.

## Features

- Vocabulary partitioning via PRF using a secret key  
- Logit biasing controlled by a configurable strength parameter  
- Detector that computes p-values and likelihood scores on arbitrary text  
- Modular architecture for integration with existing LLM generation pipelines  

## Watermarking Process

1. Initialize the PRF with a secret key.  
2. For each generation step, use the PRF to map the current context to a binary token partition.  
3. Apply a positive logit bias to all tokens in the preferred set.  
4. Sample the next token from the biased logits.  
5. Repeat for all subsequent tokens.  

The watermark is invisible in the generated text yet detectable using the same PRF and key.

## Detection

Given a piece of text and the secret key, the detector:
1. Reconstructs the preferred token sets for each position.  
2. Counts how often the text uses preferred vs. non-preferred tokens.  
3. Computes a statistical score (typically a z-score or log-likelihood ratio).  
4. Determines whether the text is consistent with the biased distribution.

Because the watermark induces a measurable deviation from random choice, detector confidence increases with text length.

## Repository Structure

