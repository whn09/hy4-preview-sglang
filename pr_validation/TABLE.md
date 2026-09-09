Model: qwen3moe-fp8_tp2_ep2_a2a-deepep-auto

| a2a backend | hook | assert order | ep_dispatch_algorithm | outcome | exception | completions |
|---|---|---|---|---|---|---|
| deepep-auto | present | base | none | **served** | — | same as control |
| deepep-auto | present | base | static | **served** | — | same as control |
| deepep-auto | present | fixed | none | **served** | — | same as control |
| deepep-auto | present | fixed | static | **served** | — | same as control |
| deepep-auto | removed | base | none | **crashed** | `AssertionError` (**bare**) | — |
| deepep-auto | removed | fixed | none | **served** | — | same as control |
| deepep-auto | removed | fixed | static | **crashed** | `AssertionError: no expert location metadata: the model class is missing get_model_config_for_expert_location` | — |
Model: qwen15moe

| a2a backend | hook | assert order | ep_dispatch_algorithm | outcome | exception | completions |
|---|---|---|---|---|---|---|
| deepep | present | base | none | **crashed** | `NotImplementedError: Unquantized DeepEP MoE currently supports low_latency mode only` | — |
| deepep | present | fixed | none | **crashed** | `NotImplementedError: Unquantized DeepEP MoE currently supports low_latency mode only` | — |
| deepep | removed | base | none | **crashed** | `AssertionError` (**bare**) | — |
| none | present | base | none | **served** | — | e08bee88e2d7 |
| none | removed | base | none | **served** | — | e08bee88e2d7 |
