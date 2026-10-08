# Run a human acceptance study

This packet is ready for a researcher to prepare and conduct. It contains no
human observations. A study becomes confirmatory only after its exact wheel,
documentation, artifact, environment, assignments and analysis are frozen.
Its target is the existing [ergonomics standard](ergonomics.md): maximize
`(Q, -T95, -K)` in that order, with correct execution **and** interpretation
required for Q. The prospective targets remain Q ≥ 90% overall and ≥ 80% in
every assigned task/persona cell; they are not measured results.

The linked research packet is included in the source checkout and is not
distributed in the wheel. Copy its `studies/human/` folder and frozen public
documentation together when sharing the study with a researcher.

Use the [researcher protocol](studies/human/PROTOCOL.md)
to prepare the environment, freeze the packet and record the session. Give
participants only the [participant packet](studies/human/PARTICIPANT.md),
its fixture helper and the frozen public documentation. Keep the observer key,
rubric and prospective analysis outside their workspace.

The packet tests ordinary installed-wheel Choice setup, application IDs, batch
association, error recovery and failure-inclusive evaluation. Controlled logits
isolate Choice, Score and Noul interpretation from model quality. A fixed scorer
cannot pass the real setup task. Score/Noul readout demonstrations do not add
these capabilities to ordinary `wyat.load`. No task performs a real business
action. Use the existing prediction evaluators for model-quality work, separately
from this human study.

Run an exploratory pilot first if the packet or environment needs repair. Keep
pilot participants and measurements outside confirmation. The initial proposed
confirmation proposal is 30 people in each of the beginner, integrator and
scorer-author personas, 90 total. This optional proposal is a practical minimum
for that confirmation design, not an assumed available cohort, a mandatory
release prerequisite, a power calculation or a guarantee of precision. Report
cell denominators and uncertainty; small samples and shared recruitment can
limit interpretation. Agent trials do not count as human enrollment.

Researchers must arrange human participation under their own applicable study
requirements. This repository neither recruits participants nor supplies an
ethics approval. Use anonymous IDs and fictional inputs. The protocol asks for
no name, employer, personal message, model credential or screen recording.

Until a completed, frozen study is available, human Q, T95 and K remain
**unmeasured**. Passing software checks establishes packet mechanics only.
