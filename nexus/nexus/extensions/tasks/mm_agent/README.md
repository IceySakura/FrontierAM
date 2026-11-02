# Search Task

基于 Nexus 的搜索任务评估框架，用于自动化测试和评估多模态Agent。

## ⚡ 快速开始

```bash
uv venv
uv pip install -e ".[dev]"
nexus modules install mm_agent
uv pip install matplotlib

#add user_id and key
vim test_example.sh 

bash ./nexus/extensions/tasks/mm_agent/test_example.sh
```

