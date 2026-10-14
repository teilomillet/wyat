# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Pydantic: describe the labels you want as a model; get a validated object.

Like structured-output libraries for LLMs, but every field is a scored
decision rather than generated text. The helper turns field types into
questions: an ``Enum`` becomes a Choice, a ``bool`` a Noul, and an ``int``
marked with a ``Rubric`` a Score. One ``predict`` call fills the whole object,
and the result passes normal Pydantic validation.

    uv run --with 'pydantic>=2' python 08_pydantic_schema.py --demo
"""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any

from common import load_model, missing

import wyat

try:
    from pydantic import BaseModel, Field
except ImportError as error:
    missing("pydantic>=2", error)


@dataclass(frozen=True)
class Rubric:
    levels: tuple[str, ...]


def decide_into[T: BaseModel](model: wyat.Model, schema: type[T], state: Any) -> T:
    questions: dict[str, dict[str, Any]] = {}
    for name, field in schema.model_fields.items():
        asked = field.description or name
        rubric = next((m for m in field.metadata if isinstance(m, Rubric)), None)
        if isinstance(field.annotation, type) and issubclass(field.annotation, Enum):
            criteria = {member.name: member.value for member in field.annotation}
            questions[name] = {
                "type": "choice",
                "instructions": asked,
                "criteria": criteria,
            }
        elif field.annotation is bool:
            questions[name] = {"type": "noul", "instructions": asked}
        elif rubric is not None:
            questions[name] = {
                "type": "score",
                "instructions": asked,
                "criteria": list(rubric.levels),
            }
        else:
            raise TypeError(f"{name}: use an Enum, bool or Annotated[int, Rubric]")
    answers = model.predict({"state": state, "questions": questions})
    values: dict[str, Any] = {}
    for name, answer in answers.items():
        if isinstance(answer, wyat.ChoiceAnswer):
            enum: Any = schema.model_fields[name].annotation
            values[name] = enum[answer.choice]  # option IDs are member names
        elif isinstance(answer, wyat.NoulAnswer):
            values[name] = answer.probability_true >= 0.5
        else:
            values[name] = round(answer.expected_ordinal_value)
    return schema.model_validate(values)


class Queue(Enum):
    billing = "Billing: charges, refunds, invoices."
    technical = "Technical: errors, crashes, login."
    delivery = "Delivery: shipping, parcels, tracking."


class TicketLabels(BaseModel):
    queue: Queue = Field(description="Which team should handle the ticket?")
    refund_requested: bool = Field(description="Does the customer ask for a refund?")
    urgency: Annotated[
        int, Rubric(("Can wait.", "Today.", "Now: money or security at risk."))
    ] = Field(description="How urgent is the ticket?")


model = load_model(__doc__)
ticket = {"subject": "Charged twice", "body": "Please refund the double charge today."}
labels = decide_into(model, TicketLabels, ticket)
print(f"queue={labels.queue.name}")
print(f"refund_requested={labels.refund_requested}")
print(f"urgency={labels.urgency}")
