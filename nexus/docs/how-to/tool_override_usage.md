# Tool Meta Override - Usage Guide

## Overview

Tool metadata override allows you to dynamically modify schema-visible tool name, tool description, and parameter descriptions at runtime without changing the original tool definition.

## Configuration Structure

Overrides are stored in `ExecutionContext.tool_configs.meta_overrides`:

```python
tool_configs = {
    'meta_overrides': {
        '<tool_registry_name>': {
            'name': 'Schema-visible tool name',
            'description': 'New tool description',
            'params': {
                '<param_name>': {
                    'description': 'New parameter description'
                }
            }
        }
    },
    # Other tool configurations (API keys, etc.)
    'api_key': 'xxx',
    'timeout': 30,
    ...
}
```

## Priority

Configuration priority (high to low):
1. **Instance-level** (via `ToolExecutor.set_tool_override()`)
2. **Context-level** (via `ExecutionContext.tool_configs.meta_overrides`)
3. **Default** (from tool registration)

## Usage Examples

### Method 1: Via ExecutionContext (Recommended for global config)

```python
from nexus.core.context import ExecutionContext
from nexus.core.tool import ToolMetaOverride, ToolParamOverride

# Using dict format (simple)
ctx = ExecutionContext(
    tool_configs={
        'meta_overrides': {
            'swe_agent/bash': {
                'description': 'Execute bash commands in Docker container',
                'params': {
                    'command': {'description': 'Shell command (max 1000 chars)'},
                    'timeout': {'description': 'Timeout in seconds (default: 30)'}
                }
            },
            'search': {
                'description': 'Search the web using Bing API',
                'params': {
                    'query': {'description': 'Search query (2-500 characters)'}
                }
            }
        }
    }
)

# Using ToolMetaOverride (with validation)
ctx = ExecutionContext(
    tool_configs={
        'meta_overrides': {
            'swe_agent/bash': ToolMetaOverride(
                name='Bash',
                description='Execute bash in container',
                params={
                    'command': ToolParamOverride(
                        description='Command to execute'
                    )
                }
            ).model_dump()
        }
    }
)

async with ctx:
    # Overrides are automatically applied to all ToolExecutor instances
    executor = ToolExecutor([bash_tool, search_tool])
    schemas = executor.get_tool_schemas()  # Contains overridden descriptions
```

### Method 2: Via ToolExecutor (For dynamic/conditional overrides)

```python
from nexus.core.tool_executor import ToolExecutor

executor = ToolExecutor([bash_tool, search_tool])

# Set override for specific tool
executor.set_tool_override(
    'swe_agent/bash',
    name='Bash',
    description='Execute bash in isolated container',
    params={
        'command': 'Shell command (max 1000 chars)',
        'timeout': 'Timeout in seconds'
    }
)

# Get schemas with overrides applied
schemas = executor.get_tool_schemas()
```

### Method 3: Combined (Instance overrides context)

```python
# Context provides base overrides
ctx = ExecutionContext(
    tool_configs={
        'meta_overrides': {
            'swe_agent/bash': {
                'description': 'Base description from context',
                'params': {
                    'command': {'description': 'Base command description'}
                }
            }
        }
    }
)

async with ctx:
    executor = ToolExecutor([bash_tool])

    # Context override is active
    schemas = executor.get_tool_schemas()
    # → uses 'Base description from context'

    # Instance override takes precedence
    executor.set_tool_override('swe_agent/bash', description='Override from instance')
    schemas = executor.get_tool_schemas()
    # → uses 'Override from instance'
```

## Use Cases

### 1. Localization

```python
# Chinese descriptions
ctx = ExecutionContext(
    tool_configs={
        'meta_overrides': {
            'swe_agent/bash': {
                'description': '在隔离容器中执行 bash 命令',
                'params': {
                    'command': {'description': '要执行的 shell 命令'}
                }
            }
        }
    }
)
```

### 2. Context-Aware Descriptions

```python
# In coding task
if task_type == 'coding':
    executor.set_tool_override(
        'swe_agent/bash',
        name='Bash',
        description='Execute Python code in sandboxed environment',
        params={'command': 'Python code to execute'}
    )

# In system task
elif task_type == 'system':
    executor.set_tool_override(
        'swe_agent/bash',
        description='Run system commands in secure container'
    )
```

### 3. Dynamic Prompting Based on State

```python
class AdaptiveAgent:
    async def run(self):
        executor = ToolExecutor(self.tools)

        for step in range(max_steps):
            # Adjust tool descriptions based on current state
            if self.error_count > 3:
                executor.set_tool_override(
                    'swe_agent/bash',
                    params={
                        'command': '⚠️ CAUTION: Multiple errors detected! Double-check your command.'
                    }
                )

            # Generate schemas and call LLM
            schemas = executor.get_tool_schemas()
            response = await self.llm.chat(..., tools=schemas)
```

### 4. Environment-Specific Descriptions

```python
# Different descriptions for different runtime environments
if runtime_type == 'docker':
    meta_overrides = {
        'swe_agent/bash': {
            'description': 'Execute in Docker container (isolated)',
            'params': {
                'command': {'description': 'Command (Docker security limits apply)'}
            }
        }
    }
elif runtime_type == 'nix':
    meta_overrides = {
        'swe_agent/bash': {
            'description': 'Execute in Nix environment (reproducible)',
            'params': {
                'command': {'description': 'Command (Nix sandbox restrictions apply)'}
            }
        }
    }

ctx = ExecutionContext(
    tool_configs={'meta_overrides': meta_overrides}
)
```

### 5. Loading from Configuration File

```python
import json

# Load from config file
with open('tool_overrides.json') as f:
    config = json.load(f)

ctx = ExecutionContext(
    tool_configs={
        'meta_overrides': config['meta_overrides']
    }
)
```

## Configuration File Format

```json
{
  "meta_overrides": {
    "swe_agent/bash": {
      "description": "Execute bash commands in container",
      "params": {
        "command": {
          "description": "Shell command to execute (max 1000 chars)"
        },
        "timeout": {
          "description": "Execution timeout in seconds"
        }
      }
    },
    "search": {
      "description": "Search the web with custom engine",
      "params": {
        "query": {
          "description": "Search query string"
        }
      }
    }
  }
}
```

## API Reference

### ToolMetaOverride

```python
class ToolMetaOverride(BaseModel):
    name: str | None = None
    description: str | None = None
    params: dict[str, ToolParamOverride] = {}
```

### ToolParamOverride

```python
class ToolParamOverride(BaseModel):
    description: str | None = None
```

### ToolExecutor.set_tool_override()

```python
def set_tool_override(
    self,
    tool_name: str,
    *,
    name: str | None = None,
    description: str | None = None,
    params: dict[str, str | ToolParamOverride] | None = None,
) -> None:
    """Set metadata override for a tool.

    Args:
        tool_name: Tool registry name (e.g. `swe_agent/bash`)
        name: Override schema-visible tool name
        description: Override tool description
        params: Parameter overrides

    Raises:
        KeyError: If tool not found
    """
```

## Best Practices

1. **Use Context for Global Config**: Put common overrides in `ExecutionContext.tool_configs.meta_overrides`
2. **Use Instance for Dynamic Changes**: Use `set_tool_override()` for state-dependent modifications
3. **Partial Overrides**: You can override just name/description or just specific parameters
4. **Validation**: Use `ToolMetaOverride.model_validate()` to validate config from external sources
5. **Separation of Concerns**: Keep meta overrides separate from other tool configs (API keys, etc.)

## Limitations

- Overrides affect schema generation and schema-name resolution in `ToolExecutor.call()` / `parallel_call()`, but not the underlying tool implementation
- Cannot add/remove parameters or override parameter types/constraints
- Overrides are not persisted across ExecutionContext instances

## Future Enhancements

Potential future extensions:
- Parameter constraint overrides (min, max, enum values)
- Tool visibility control (hide/show tools)
- Override templates and presets
