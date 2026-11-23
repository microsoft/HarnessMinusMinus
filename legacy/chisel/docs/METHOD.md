# HarnessMM execution control flow

HarnessMinusMinus (HarnessMM) optimizes agent execution through guarded decision reuse and recursive self-improvement. The standalone controllers use the `chisel` Python namespace and the `chisel-jev` distribution name.

## Executable Decision Graphs (EDG)

Slow reasoning emits the current native action and a bounded graph of future decisions. Each future node stores an exact native action, complete literal arguments, a predecessor, and a public execution guard. Edges connect the root or an earlier node to its successor, forming an acyclic graph.

A decision is resolved when prior reasoning has fixed its action and literal arguments. The future receipt determines whether that stored action is ready to execute. Later feedback supplies execution evidence; new semantic inference returns control to the slow actor.

Serialized action banks represent the nodes and edges. The graph hash identifies their authored content. Benchmark-specific public-state adapters retain their own native contracts; their input projections and dynamic binding mechanisms must retain their actual intervention identity.

## Guarded Decision Reuse (GDR)

GDR retains the authored graph across tool transitions. A node is eligible when its predecessor has completed, its public guard holds on observed receipts, and its arguments are closed without new semantic inference. Missing dependencies, unsupported observations, unknown guards, and false guards return control to slow reasoning.

A typed selector returns an eligible candidate ID or abstains without revising its arguments. A unique eligible node can execute deterministically. Empty eligible sets and abstention invoke the original actor. Each actual execution receipt updates the same graph, and the actor can author a new graph when fresh reasoning is needed.

The Jev-motivated candidate-ID/abstention interface uses single-token decoding when selection requires a model call. Typed output alone does not establish removal of a reasoning callback.

## Certified Callback Substitution (CCS)

CCS executes an admitted stored action through the original native interface in place of a registered slow callback. Its substitution certificate binds:

- Authored graph hash and decision ID.
- Completed predecessor receipt.
- Exact executed arguments and actual tool receipt.
- Bypassed callback ID.

The audit matches the authored decision, predecessor completion, native execution, and registered parent boundary. It verifies omission of at least one slow callback without predicting that callback's unobserved response or token count. Certificates establish execution and omission; the native scorer establishes task correctness.

The public controller helpers expose candidate support, native dispatch, and execution records. Complete CCS evidence also requires the integrating harness's callback registration and physical request ledger. A generic support trace or a tool-execution receipt alone does not provide all certificate fields.

## Complete-task accounting

Complete episode cost sums measured actor input and output tokens for authorship, selection, added context, remaining slow calls, fallback, recovery, finalization, and failed paid executions. Cached input remains counted. Deterministic checks add CPU work and no actor tokens. Shared authoring is charged exactly once to its declared cohort. Development cost is reported separately, including training, proposals, validation, and rejected attempts.

Report consumption as **Tokens/task (k)** with two decimal places: measured actor input plus output tokens divided by the complete matched execution population and by 1,000. Unknown usage remains unknown. Token reductions do not establish latency or energy reductions.

The **Substitution Paradox** describes fast-slow composition that adds a selector without removing downstream deliberation. A beneficial substitution requires eliminated computation to exceed added authorship, selection, context, and recovery, while satisfying the predeclared quality constraint.

## Core mechanism and companion configurations

EDG, GDR, and CCS form the core execution method. Prompt editing, context compaction, instruction optimization, answer policies, finalization, and RSI configure the surrounding harness. Training develops those settings; disjoint validation selects among training-frozen candidates; the selected configuration is frozen before testing. The runtime selector chooses an eligible stored action or abstains.

The paper's native harness pairs measure integrated quality and complete cost. Its matched MBPP-sanitized ablation holds authoring and context settings common and tests persistence and callback removal. Instruction-only and other benchmark-specific companion studies retain their own mechanism, population, and cost scope. Their results cannot be attributed exclusively to EDG/GDR/CCS.

## Released interfaces

[INTEGRATION.md](INTEGRATION.md) documents the actual callable interfaces. [SOURCE_MAP.md](SOURCE_MAP.md) and `SOURCE_MANIFEST.json` identify the released source. Frozen benchmark implementations, callback audits, physical ledgers, and result artifacts remain separate reproduction assets. The README reports the manuscript's measured results; the synthetic tests validate the released controller behavior.
