![Python 3.7](https://img.shields.io/badge/python-3.7-blueviolet.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-MIT-05b502.svg?style=plastic)

# Causal Proxy Models For Concept-Based Model Explanations (CPM)
<p align="center">
  <b><a href="https://nlp.stanford.edu/~wuzhengx/">Zhengxuan Wu</a>*, <a href="https://www.kareldoosterlinck.com/">Karel D'Oosterlinck</a>*, <a href="https://atticusg.github.io/">Atticus Geiger</a>*, <a href="https://www.linkedin.com/in/amir-zur-a924ba187/">Amir Zur</a>, <a href="https://web.stanford.edu/~cgpotts/">Christopher Potts</a></b></span>
</p>

The codebase contains some implementations of our preprint [Causal Proxy Models For Concept-Based Model Explanations](https://nlp.stanford.edu/~wuzhengx/). In this paper, we introuce two variants of CPM, 
* CPM<sub>IN</sub>: Input-base CPM uses auxiliary token to represent the intervention, and is trained in a supervised way of predicting counterfactual output. This model is built on an input-level intervention.
* CPM<sub>HI</sub>: Hidden-state CPM uses Interchange Intervention Training (IIT) to localize concept information within its representations, and swaps hidden-states to represent the intervention. It is trained in a supervised way of predicting counterfactual output. This model is built on a hidden-state intervention.

This codebase contains implementations and experiments of **CPM<sub>HI</sub>**. If you experience any issues or have suggestions, please contact me either thourgh the issues page or at wuzhengx@cs.stanford.edu. 

