---
layout: distill
title: "Critical Review of “Agentic AI in Financial Services: Opportunities, Risks, and Responsible Implementation” by IBM Consulting"
date: 2025-05-16
description: "The recent IBM Consulting publication Agentic AI in Financial Services: Opportunities, Risks, and Responsible Implementation (May 2025, 36 pages) addresses the potential benefits and pitfalls of agent-based AI solutions in the financial sector. However, a close reading exposes gaps between the paper’s broad aspirations and the strict regulatory and practical realities—especially regarding compliance with EU privacy rules (GDPR) and the complexities of using real operational data. The following critique highlights the major oversights."
tags:
  [
    "agentic-ai",
    "ibm",
    "enterprise-ai",
    "ai-strategy",
    "eu-ai-act",
    "gdpr",
    "data-privacy",
    "financial-services",
    "eu",
  ]
category: "AIOps"
comments: true
---

<br>
# Critical Review of “Agentic AI in Financial Services: Opportunities, Risks, and Responsible Implementation” by IBM Consulting

The recent IBM Consulting publication Agentic AI in Financial Services: Opportunities, Risks, and Responsible Implementation (May 2025, 36 pages) addresses the potential benefits and pitfalls of agent-based AI solutions in the financial sector. However, a close reading exposes gaps between the paper’s broad aspirations and the strict regulatory and practical realities—especially regarding compliance with EU privacy rules (GDPR) and the complexities of using real operational data. The following critique highlights the major oversights.

Link to the document IBM Consulting published: [governing-agentic-ai-for-financial-services.pdf](https://www.ibm.com/downloads/documents/gb-en/12f5a71117cdc329)

⸻

## Overlooking Strict Data Use Constraints

On pages 7–8, the authors tout “AI-Powered Customer Engagement & Personalisation” and “AI-Driven Operational Excellence & Governance” as promising areas, referencing scenarios like automated onboarding (p. 8) or AML optimization (p. 7). In reality, training a model or even ingesting operational data for these tasks in the EU is fraught with hurdles.

- Operational data for training: Under GDPR, re-purposing a client’s personal data typically requires explicit consent or a well-defined legal basis. Furthermore, individuals have the right to request erasure (“right to be forgotten”), which complicates direct ingestion of raw operational data into any model. The paper (particularly in the sections on data privacy, pp. 13–14) acknowledges “some” compliance concerns but provides neither a robust plan nor an acknowledgment of how banks might face an outright block on training with personal data without granular policies, anonymization, or specialized transformations.

- Inadequate discussion of data minimization: The authors propose (p. 27) “robust data governance frameworks” but do not detail how to keep day-to-day compliance on track with the EU’s data minimization principle. Simply referencing “data governance” overlooks that banks need to implement fine-grained pseudonymization, encryption, and local data processing strategies to avoid infringing on privacy regulations.

⸻

## Naive Treatment of Code and IP Considerations

Pages 9–10 discuss “AI-Augmented Technology & Software Development,” as though enterprise developers can freely paste proprietary code into an LLM for debugging or enhancement. This is at odds with actual practice:

- Vendor TOS: Many cloud-based code-generation services (e.g., Azure OpenAI, GitHub Copilot) prohibit or limit the uploading of confidential or personal data. Additionally, an institution’s internal compliance often disallows unprotected code sharing with external tools.

- Intellectual Property: The paper does not clarify how dev teams protect IP or prevent generative AI models from storing or reusing proprietary code for other users. Nor does it delve into versioning or data retention obligations.

This gap undermines the practicality of “AI-augmented coding” at scale when engineers cannot simply hand over sensitive code bases to black-box third-party systems.

⸻

## Shallow Acknowledgment of the EU AI Act

Though the text covers “Compliance-Proofing in an Uncertain Regulatory Landscape” (pp. 18–19), it primarily references the Australian Proposals Paper. The section on the EU AI Act does note that certain use cases (e.g. loan approvals) might be “high-risk,” yet the guidance remains broad. For instance, while the Act stipulates mandatory human oversight (Article 14) and robust record-keeping, the IBM paper mainly repeats the idea of “compliance by design” without exploring how a financial institution would genuinely operationalize data erasure rights, model documentation, or real-time user control.

⸻

## Repetition Over Depth

The booklet frequently reiterates the same risk categories—data privacy, drift, misalignment, etc.—across multiple sections (see especially pp. 10–17, and again on pp. 20–21). However, the level of detail on actual implementation challenges remains limited. For example, the repeated call for “robust guardrails,” “real-time monitoring,” or “codified controls” (pp. 20–21, 33–34) sounds comprehensive, but the authors rarely move beyond conceptual statements toward actionable tactics that satisfy strict GDPR or local financial regulations.

⸻

## Missing Practical Details for EU Readiness

In real-world EU financial contexts, the paper’s scenarios would demand:

- Granular data-handling policies: A technical method for how an AI system filters out sensitive fields, respects erasure requests, and tracks purpose limitation (i.e., ensuring data is used only for the originally stated purpose).
- Vendor-side privacy proofs: If a bank leverages external LLMs (p. 29’s examples of OpenAI Operator or Microsoft 365 Copilot), it must demonstrate compliance with EU cross-border data transfer rules, robust encryption, and capacity to retrieve and delete data upon customer request.
- Detailed risk evaluations: Concrete references to data classification, retention schedules, lawful bases for each subset of personal data, and privacy impact assessments that align with Article 35 of the GDPR.

Without these specifics, the “Agentic AI” vision may conflict with everyday privacy demands in Europe.

⸻

## Conclusion

In Agentic AI in Financial Services, IBM Consulting offers a thorough overview of potential agent-based AI benefits—improving customer onboarding, governance, and software development—yet the document tends to downplay or oversimplify the tangible constraints in EU contexts. While it recommends broad frameworks—risk management, compliance by design, or real-time monitoring—the discussion of operational data usage, data-subject rights, vendor IP rules, and the EU AI Act remains high-level. Financial institutions pursuing agentic AI will need far more detailed strategies to ensure that using client data for training or generative tools for coding is both practical and lawful in jurisdictions like the EU.

In short, the paper’s conceptual overview cannot substitute for a rigorous, domain-specific approach that addresses GDPR constraints, vendor trust issues, and strict EU AI Act obligations. Enterprises evaluating these solutions should pair IBM’s high-level risk categories with deep operational expertise to remain compliant, protect sensitive data, and build truly sustainable AI programs.
