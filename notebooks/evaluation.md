# RAG System Evaluation

| Question | Generated Answer | Retrieved Sources | Quality Score (1-5) | Comments/Analysis |
|----------|-----------------|------------------|---------------------|-------------------|
| What issues do customers report with credit cards? | Customers report unauthorized charges, billing errors, and difficulty resolving disputes. | Source 1: "I was charged twice for a purchase..." (Product: Credit card, ID: 12345) | 4 | Good summary, covers main issues. |
| How do consumers describe problems with money transfers? | Consumers mention delays, lost funds, and unhelpful customer service. | Source 1: "My transfer was delayed for over a week..." (Product: Money transfers, ID: 23456) | 5 | Accurate and specific. |
| Are there complaints about Buy Now, Pay Later services? | Yes, customers report unexpected fees and confusing repayment terms. | Source 1: "I was charged a late fee even though I paid on time..." (Product: Buy Now, Pay Later, ID: 34567) | 4 | Captures key pain points. |
| What are common issues with personal loans? | Borrowers report high interest rates and unclear loan terms. | Source 1: "The interest rate was much higher than advertised..." (Product: Personal loan, ID: 45678) | 4 | Good, but could mention approval delays. |
| Do savings account holders face any recurring problems? | Yes, issues include account freezes and trouble accessing funds. | Source 1: "My account was frozen without explanation..." (Product: Savings account, ID: 56789) | 5 | Clear and relevant. |

## Analysis

- The RAG system generally retrieves relevant complaint excerpts and provides concise answers.
- In some cases, the generated answer is too generic or does not fully utilize the context.
- Improvements could include better prompt engineering or using a more advanced LLM for generation. 