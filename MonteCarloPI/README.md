# Monte Carlo PI Approximation

Two version are proposed: one is the optimal, the second is the complex that is only used to deepen some topics.


Usually this problem is solved assigning a batch to every process, and doing a reduce collective in the end. So the communication is limited and the agorithm is fast, but more complex concepts are not involved so i've decided to produce an optimal version and a complex version to handle complex concepts like composite datatypes and complex communication patters.
