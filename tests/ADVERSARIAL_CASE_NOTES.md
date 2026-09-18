# Adversarial test scope and specification boundaries

These tests derive expected behavior from the official problem statement and
public sample pack. They do not require a particular optimal hourly sequence
when multiple schedules have equal cost.

Explicitly undefined or not directly stated:

- **Overlapping `solar_reduction` directives:** the documents define
  `original_solar[h] * factor` for each directive but do not define whether
  overlapping factors multiply, take the minimum, or use another precedence
  rule. Production code rejects this combination with `DirectiveOverlapError`.
- **A natural-language interval ending at “midnight”:** the documents require
  output hours in `0..23` and end-exclusive intervals, but do not explicitly say
  how the word “midnight” should be normalized when it is the end boundary. The
  corresponding test is marked skipped instead of assuming `[22, 23]` or another
  mapping.
- **Fractional wording:** fractions are not given as official examples. Tests
  use ordinary mathematical meaning as adversarial LLM expectations, based on
  the requirement that hidden cases may paraphrase supported directives.
- **Constant tariffs:** optimal schedules are non-unique. Tests assert replayed
  validity and optimal totals, never an exact battery-action sequence.

Repeated non-solar directives are simultaneous hard constraints. Therefore,
overlapping reserve floors naturally use the strongest floor, overlapping grid
caps use the tightest cap, and repeated no-charge/no-discharge windows use the
union of their listed hours.
