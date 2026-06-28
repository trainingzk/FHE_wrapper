# Authenticated Workflow Benchmark

## Overview

This repository contains the reference implementation accompanying the paper:

> **Authenticated Workflow for Homomorphic Computation: Task Binding, Replay Protection, and Worker Accountability**

The implementation realizes the authenticated workflow described in the paper and reproduces the benchmark results, including execution-time measurements, scalability analysis, and publication-quality figures.

## Features

* Implementation of the protocol algorithms:

  * CreateTask
  * ProcessTask
  * VerifyResult
* Hybrid public-key encryption using RSA-OAEP and AES-256-GCM
* ECDSA (P-256) digital signatures
* Support for real CKKS evaluation using TenSEAL
* Automatic fallback to an execution-time simulator when TenSEAL is unavailable
* Benchmark generation
* Automatic generation of publication figures
* Automatic generation of a LaTeX performance table

## Requirements

* Python 3.12 or later

Python packages:

* cryptography
* numpy
* matplotlib

Optional:

* tenseal (for real CKKS evaluation)

If TenSEAL is not installed, the program automatically switches to a timing simulator for the FHE evaluation.

## Running

Execute

```bash
python main_v3_stable.py
```

The program benchmarks the authenticated workflow for different numbers of homomorphic operations.

## Output

The implementation generates:

* `fig-breakdown.pdf`
* `fig-scalability.pdf`

It also prints a LaTeX table containing detailed benchmark results to the console.

## Notes

This implementation is intended for experimental evaluation of the authenticated workflow described in the accompanying paper. It is a research prototype designed for benchmarking and reproducibility rather than production deployment.
