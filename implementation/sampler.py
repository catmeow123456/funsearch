# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from collections.abc import Collection, Sequence

import numpy as np

from funsearch.implementation import evaluator
from funsearch.implementation import programs_database

from openai import OpenAI
client = OpenAI()

class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    # print('-' * 20, 'prompt', '-' * 20)
    # print(prompt)
    # print('-' * 20, 'prompt', '-' * 20)
    completion = client.chat.completions.create(
      model="gpt-4o-2024-11-20",
      messages=[
        {
          "role": "system",
          "content": """\
You are a helpful assistant, you can help me write some code,
Your output format is ```python\ndef FUNCTION_NAME(...):...\n```
"""
        },
        {"role": "user", "content": prompt}
      ],
      stream=False
    )
    res = completion.choices[0].message.content
    # 截取 ```python 到 ``` 的代码
    res = res[res.find('```python') + 9:]
    res = res[res.find('def') + 4:]
    res = res[res.find('\n') + 1:]
    res = res[:res.find('```')]
    print('-' * 20, 'code', '-' * 20)
    print(res)
    print('-' * 20, 'code', '-' * 20)
    return res

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt)

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    while True:
      prompt = self._database.get_prompt()
      samples = self._llm.draw_samples(prompt.code)
      # This loop can be executed in parallel on remote evaluator machines.
      for sample in samples:
        chosen_evaluator = np.random.choice(self._evaluators)
        chosen_evaluator.analyse(
            sample, prompt.island_id, prompt.version_generated)
