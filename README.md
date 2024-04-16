# LSTM-PoS

Here are the mathematical equations for each step in the forward and backward propagation of the LSTM model:

### Forward Propagation:
1. Concatenated input: 
   - \( \text{concat\_inputs}[q] = [\text{hidden\_states}[q - 1]; \text{inputs}[q]] \)

2. Forget gate:
   - \( \text{forget\_gates}[q] = \sigma(\text{wf} \times \text{concat\_inputs}[q] + \text{bf}) \)

3. Input gate:
   - \( \text{input\_gates}[q] = \sigma(\text{wi} \times \text{concat\_inputs}[q] + \text{bi}) \)

4. Candidate gate:
   - \( \text{candidate\_gates}[q] = \tanh(\text{wc} \times \text{concat\_inputs}[q] + \text{bc}) \)

5. Output gate:
   - \( \text{output\_gates}[q] = \sigma(\text{wo} \times \text{concat\_inputs}[q] + \text{bo}) \)

6. Cell state:
   - \( \text{cell\_states}[q] = \text{forget\_gates}[q] \times \text{cell\_states}[q - 1] + \text{input\_gates}[q] \times \text{candidate\_gates}[q] \)

7. Hidden state:
   - \( \text{hidden\_states}[q] = \text{output\_gates}[q] \times \tanh(\text{cell\_states}[q]) \)

8. Output:
   - \( \text{outputs}[q] = \text{wy} \times \text{hidden\_states}[q] + \text{by} \)

### Backward Propagation:
1. Final gate weights and biases errors:
   - \( \text{dwy} += \text{error} \times \text{hidden\_states}[q]^T \)
   - \( \text{dby} += \text{error} \)

2. Hidden state error:
   - \( \text{d\_hs} = \text{wy}^T \times \text{error} + \text{dh\_next} \)

3. Output gate weights and biases errors:
   - \( \text{d\_o} = \tanh(\text{cell\_states}[q]) \times \text{d\_hs} \times \sigma'(\text{output\_gates}[q]) \)
   - \( \text{dwo} += \text{d\_o} \times \text{inputs}[q]^T \)
   - \( \text{dbo} += \text{d\_o} \)

4. Cell state error:
   - \( \text{d\_cs} = \tanh'(\tanh(\text{cell\_states}[q])) \times \text{output\_gates}[q] \times \text{d\_hs} + \text{dc\_next} \)

5. Forget gate weights and biases errors:
   - \( \text{d\_f} = \text{d\_cs} \times \text{cell\_states}[q - 1] \times \sigma'(\text{forget\_gates}[q]) \)
   - \( \text{dwf} += \text{d\_f} \times \text{inputs}[q]^T \)
   - \( \text{dbf} += \text{d\_f} \)

6. Input gate weights and biases errors:
   - \( \text{d\_i} = \text{d\_cs} \times \text{candidate\_gates}[q] \times \sigma'(\text{input\_gates}[q]) \)
   - \( \text{dwi} += \text{d\_i} \times \text{inputs}[q]^T \)
   - \( \text{dbi} += \text{d\_i} \)

7. Candidate gate weights and biases errors:
   - \( \text{d\_c} = \text{d\_cs} \times \text{input\_gates}[q] \times \tanh'(\text{candidate\_gates}[q]) \)
   - \( \text{dwc} += \text{d\_c} \times \text{inputs}[q]^T \)
   - \( \text{dbc} += \text{d\_c} \)

8. Concatenated input error:
   - \( \text{d\_z} = \text{wf}^T \times \text{d\_f} + \text{wi}^T \times \text{d\_i} + \text{wc}^T \times \text{d\_c} + \text{wo}^T \times \text{d\_o} \)

9. Error of hidden state and cell state at next time step:
   - \( \text{dh\_next} = \text{d\_z}[:\text{hidden\_size}, :] \)
   - \( \text{dc\_next} = \text{forget\_gates}[q] \times \text{d\_cs} \)

Note: \( \sigma(x) \) is the sigmoid function, \( \tanh(x) \) is the hyperbolic tangent function, and \( \sigma'(x) \) and \( \tanh'(x) \) are their derivatives.