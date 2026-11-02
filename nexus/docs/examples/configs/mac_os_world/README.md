# macOSWorld configurations

Run commands from the repository root. All macOSWorld runnable examples live in
this directory; credentials are supplied through the environment variables named
in each example. These presets retain their original models, budgets and dataset
versions so moving them does not silently change an evaluation.

| Execution mode | Example |
| --- | --- |
| GPT CUA | `mac_os_world_gptcua_devcloud.yaml` |
| GPT CUA with paper-aligned settings | `mac_os_world_gptcua_paper_devcloud.yaml` |
| Claude CUA | `mac_os_world_claudecua_devcloud.yaml` |
| Screenshot VLM with the upstream GPT-4o prompt | `mac_os_world_gpt4o_paper_devcloud.yaml` |
| Generic main-agent tool-calling baseline | `mac_os_world_gpt4o_main_devcloud.yaml` |
| Claude Code harness | `mac_os_world_claude_code_devcloud.yaml` |
| Claude Code harness with HY | `mac_os_world_claude_code_hunyuan_devcloud.yaml` |
| Codex harness | `mac_os_world_codex_devcloud.yaml` |
| Codex harness with HY | `mac_os_world_codex_hunyuan_devcloud.yaml` |

`mac_os_world_codex_hunyuan_platform_params.json` is a platform parameter template,
not a standalone Nexus run configuration. For HY platform runs, select the
ONE_STOP_SERVICE model source and preserve its managed routing.

The `mac_os_world_gpt56_luna_devcloud_test.yaml` and
`mac_os_world_gpt56_luna_full.yaml` presets are retained for historical
reproducibility. The latter pins an older full dataset split; it is not a selection
of the 171 leaderboard tasks. Use an explicitly verified dataset version/split for
new leaderboard comparisons.

See the [runtime guide](../../../how-to/use-runtime/use-macosworld-devcloud.md) for
setup and the [adapter guide](../../../developing/macosworld-adapters.md) for entry
points and configuration migration.
