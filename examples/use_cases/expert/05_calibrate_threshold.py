# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Turn labelled examples into a deployable policy: temperature + threshold.

Before automating, measure. With a few dozen labelled cases you can:
1. evaluate the model (failures count as wrong, nothing is hidden),
2. fit a temperature so probabilities better match observed accuracy,
3. pick the smallest confidence margin that reaches your target accuracy,
   accepting that low-margin cases go to a human,
4. check that policy on cases you did not use to choose it.

Include the messages that are genuinely hard to route. If every tune case is
easy, the fit only learns "be more confident": it ends on the edge of its
search range, and a margin threshold then filters nothing. This example
refuses to emit a policy in that case.
"""

import json
import math

from common import load_model

import wyat
from wyat.evaluation import EvaluationCase, evaluate

TARGET_ACCURACY = 0.9
TEMPERATURE_BOUNDS = (0.05, 10.0)
QUEUES = {
    "billing": "Billing: charges, refunds, invoices.",
    "technical": "Technical: errors, crashes, login problems.",
    "delivery": "Delivery: shipping, tracking, parcels.",
}
# Labels follow the team rule: route to the queue that can fix the cause.
TUNE = [
    ("I was charged twice", "billing"),
    ("Refund for my last invoice please", "billing"),
    ("Charged again after I cancelled my subscription", "billing"),
    ("I paid but my order still says unpaid", "billing"),
    ("The app crashes on startup", "technical"),
    ("Login error code 403", "technical"),
    ("Cannot download my invoice, page errors", "technical"),
    ("The refund button does nothing when I click it", "technical"),
    ("Where is my parcel?", "delivery"),
    ("Tracking number does not work", "delivery"),
    ("Payment taken but order not shipped", "delivery"),
    ("I want my money back, the parcel never came", "delivery"),
]
CHECK = [
    ("Wrong amount charged on my card", "billing"),
    ("Can I get a receipt for my last payment?", "billing"),
    ("The website shows an error when I pay", "technical"),
    ("Password reset email never arrives", "technical"),
    ("Parcel arrived damaged", "delivery"),
    ("Courier left my parcel at the wrong address", "delivery"),
    ("Charged for shipping I did not choose", "billing"),
    ("Order confirmation page is blank", "technical"),
]


def cases(rows: list[tuple[str, str]], prefix: str) -> list[EvaluationCase]:
    return [
        EvaluationCase(
            f"{prefix}-{index}",
            wyat.Request.choice("Which queue handles this?", QUEUES, context=text),
            (label,),
        )
        for index, (text, label) in enumerate(rows)
    ]


model = load_model(__doc__)
tune = evaluate(model, cases(TUNE, "tune"))
temperature, calibrated = tune.calibrate(bounds=TEMPERATURE_BOUNDS)
print(f"accuracy {tune.accuracy:.2f}, coverage {tune.coverage:.2f}")
# The fit minimizes log loss; calibration error is noisy on a dozen cases.
print(f"log loss {tune.log_loss:.3f} -> {calibrated.log_loss:.3f}", end=", ")
print(f"calibration error {tune.expected_calibration_error:.3f}", end=" -> ")
print(f"{calibrated.expected_calibration_error:.3f} at temperature {temperature:.2f}")
if any(math.isclose(temperature, bound, rel_tol=0.01) for bound in TEMPERATURE_BOUNDS):
    print("\nThe fit hit a search bound: the tune cases are too easy to calibrate on.")
    print("Add hard, labelled cases before choosing a policy.")
    raise SystemExit(0)

print("\nmargin  accuracy  automated")
chosen = None
for margin in (0.0, 0.2, 0.4, 0.6, 0.8):
    # Accuracy on the cases the policy automates, and the share it automates.
    selected = calibrated.selective(margin)
    accuracy = "-" if selected.accuracy is None else f"{selected.accuracy:.2f}"
    print(f"{margin:>5.1f}  {accuracy:>8}  {selected.automation_rate:>9.0%}")
    if (
        chosen is None
        and selected.accuracy is not None
        and selected.accuracy >= TARGET_ACCURACY
    ):
        chosen = margin

if chosen is None:
    print("\nNo margin reaches the target: keep a human in the loop.")
else:
    check = evaluate(model, cases(CHECK, "check"), temperature=temperature)
    checked = check.selective(chosen)
    policy = {
        "model_id": model.model_id,
        "temperature": round(temperature, 3),
        "min_margin": chosen,
        "check": {
            "accuracy": checked.accuracy,
            "automated": checked.automation_rate,
        },
    }
    print("\npolicy:", json.dumps(policy))
    # In production: wyat.load(path, temperature=policy["temperature"]), then
    # automate only decisions with probability_margin >= policy["min_margin"].
