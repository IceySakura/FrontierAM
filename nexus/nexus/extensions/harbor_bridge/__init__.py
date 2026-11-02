"""Harbor-Nexus Bridge: Seamlessly integrates harbor framework into nexus.

This module provides:
- GongfengEnvironment: harbor BaseEnvironment implementation backed by nexus gongfeng runtime
- HarborBridgeTask: nexus task that orchestrates harbor's full execution flow
- HarborDatasetReader: local T3-Harbor directories as Nexus Dataset Items
- Harbor Bridge Run Part and reusable Runtime/Agent/LLM profiles
- Legacy batch generator compatibility for existing run-batch workflows
"""
