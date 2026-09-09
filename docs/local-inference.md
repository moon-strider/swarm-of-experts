# Local inference and OpenClaw

## llama.cpp on CPU

Use a GGUF model you have downloaded and verified. The recorded experiments use llama.cpp b10867 with CPU inference, four generation threads and one inference slot:

~~~bash
llama-server -m /absolute/path/model.gguf --alias local-model \
  --host 127.0.0.1 --port 8080 -ngl 0 -t 4 -tb 4 -np 1 -c 16384 \
  --jinja --chat-template-kwargs '{"enable_thinking":false}' --reasoning off
~~~

In a second terminal:

~~~bash
export LLM_BASE_URL=http://127.0.0.1:8080/v1
export LLM_MODEL=local-model
uv run swarm-of-experts serve
~~~

The model alias must agree with the configured upstream model. One llama.cpp slot serializes inference even when Swarm has concurrent pending requests; parallel voting does not imply parallel CPU generation.

The same API connection can target another OpenAI-compatible local server. Ollama itself was not used in the recorded runs.

## OpenClaw

The integration was exercised with OpenClaw 2026.9.3 and Node 24.19.0 using [isolated agent exec](https://docs.openclaw.ai/cli/agent#agent-exec). Install the chosen version and save [examples/openclaw.json](../examples/openclaw.json) as an explicit run configuration.

~~~bash
npm install --prefix .openclaw-test --omit=dev --no-audit --no-fund openclaw@2026.9.3
.openclaw-test/node_modules/.bin/openclaw agent exec \
  --config examples/openclaw.json --cwd /absolute/path/workspace \
  --thinking off --code-mode direct --local-model-lean \
  --json 'Read README.md and describe this project'
~~~

The current `agent exec` accepts the prompt positionally or through `--message-file`. For stdin, use:

~~~bash
printf '%s' 'Read README.md and describe this project' | \
  .openclaw-test/node_modules/.bin/openclaw agent exec \
  --config examples/openclaw.json --cwd /absolute/path/workspace \
  --thinking off --code-mode direct --local-model-lean --message-file - --json
~~~

Use `local-single` with tools. For `local-swarm`, disable all tools in the OpenClaw configuration; this route merges text answers and intentionally rejects executable tool calls.

The [long-task integration check](https://github.com/moon-strider/openclaw-long-tasks/blob/main/scripts/check_openclaw.py) reads a random sentinel through the real OpenClaw read tool. The sentinel is absent from its prompt, so merely returning a plausible answer cannot pass it. The report distinguishes a controlled upstream fixture from actual local-model inference.

## MAKER

Run the benchmark from an installed `openclaw-long-tasks` checkout:

~~~bash
uv run python scripts/benchmark_hanoi.py \
  --base-url http://127.0.0.1:8000/v1 --model local-single \
  --state-mode deterministic --prompt-mode phase \
  --temperature 0.7 --disks 2 3 4 --seeds 11 29 \
  --max-samples 12 --output results/hanoi
~~~

MAKER owns its sample voting and checkpoint journal. Use `local-single` as its endpoint so a vote corresponds to one model call. A Swarm text merger is not a first-to-ahead-by-k voter.

See the [experiment report and limitations](https://github.com/moon-strider/openclaw-long-tasks/blob/main/docs/experiments.md) before interpreting a small-task success as evidence about long runs.
