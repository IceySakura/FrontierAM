# Webarena

将下面的配置保存到 `./workspace/webarena/test.yaml` ，填写 `openai_clients` 相关配置之后执行 `uv run nexus run-single ./workspace/webarena/test.yaml` 测试。

配置中 task_config 是来自 https://github.com/web-arena-x/webarena/blob/main/config_files/test.raw.json 的题目， `nexus/extensions/tasks/webarena/generator.py` 是支持批量运行的生成器。

```yaml
# WebArena Task Configuration
# task_config from: https://github.com/web-arena-x/webarena/blob/main/config_files/test.raw.json

imports:
- nexus.extensions.tasks.webarena
- nexus.extensions.agents.browser_gym
task_name: webarena
task_config:
  task_id: 0
  task_config:
    sites:
    - shopping_admin
    task_id: 0
    require_login: true
    storage_state: ./.auth/shopping_admin_state.json
    start_url: __SHOPPING_ADMIN__
    geolocation: null
    intent_template: What is the top-{{n}} best-selling product in {{year}}
    instantiation_dict:
      n: 1
      year: 2022
    intent: What is the top-1 best-selling product in 2022
    require_reset: false
    eval:
      eval_types:
      - string_match
      reference_answers:
        exact_match: Quest Lumaflex™ Band
      reference_url: ''
      program_html: []
      string_note: ''
      reference_answer_raw_annotation: Quest Lumaflex™ Band
    intent_template_id: 279
  runtime_name: main
  timeout: 600
  with_na_hint: false
  with_homepage_hint: false
tool_configs: {}
runtime_provider_name: "docker"
runtime_configs: {}
agent_name: browser_gym_agent
agent_config:
  max_iterations: 30
openai_clients:
  main:
    name: "litellm"
    config:
      client_args:
        api_key: "<user>:<key>?provider=anthropic&timeout=60"
        base_url: "http://trpc-gpt-eval.production.polaris:8080"
        cache_control_enabled: true
        max_retries: 5
      request_args:
        model: "claude-opus-4-5-20251101"
    # name: "openai"
    # config:
    #   client_args:
    #     api_key: "<user>:<key>?provider=openai&timeout=60"
    #     base_url: "http://api.model-eval.woa.com/v1"
    #   request_args:
    #     model: "gpt-4o-2024-08-06"
processors: {}
```
