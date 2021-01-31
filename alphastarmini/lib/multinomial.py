#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Stable multinomial, not making inf/nan problem "

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

debug = True


def stable_multinomial(probs=None, logits=None, temperature=1, num_samples=1,
                       min_prob=1e-10, max_logit=1e+10,
                       min_temperature=1e-10, max_temperature=1e+10):
    '''
        this stable_multinomial will avoid THCNumerics<T>::ge(val, zero) failed
    '''

    if probs is not None:
        probs = probs.clamp(min=min_prob)
        logits = torch.log(probs)

    logits = logits.clamp(max=max_logit)
    temperature = np.clip(temperature, min_temperature, max_temperature)
    logits = (logits - logits.max()) / temperature
    probs = torch.exp(logits)

    print('probs:', probs) if debug else None
    print('max probs:', probs.max()) if debug else None
    print('min probs:', probs.min()) if debug else None

    return torch.multinomial(probs, num_samples)
