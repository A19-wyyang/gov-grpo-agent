#!/usr/bin/env python
"""Compatibility patches for the pinned veRL/vLLM/Transformers/PEFT stack.

veRL correctly builds an HF config with ``attn_implementation=sdpa``, but its
FSDP engine does not forward that selection to ``from_pretrained``. Recent
Transformers releases then re-select the model's flash-attention default.
This small, idempotent install-time patch forwards the configured backend.
"""

from __future__ import annotations

from pathlib import Path

import verl


def main() -> None:
    target = (
        Path(verl.__file__).resolve().parent
        / "workers"
        / "engine"
        / "fsdp"
        / "transformer_impl.py"
    )
    source = target.read_text(encoding="utf-8")
    marker = "attn_implementation=self.model_config.override_config.get("
    if marker in source:
        print(f"veRL SDPA compatibility patch already present: {target}")
    else:
        old = """                    config=self.model_config.hf_config,
                    trust_remote_code=self.model_config.trust_remote_code,
"""
        new = """                    config=self.model_config.hf_config,
                    trust_remote_code=self.model_config.trust_remote_code,
                    attn_implementation=self.model_config.override_config.get(
                        "attn_implementation", "flash_attention_2"
                    ),
"""
        if old not in source:
            raise RuntimeError(
                f"Unsupported veRL layout; expected from_pretrained block not found in {target}"
            )
        target.write_text(source.replace(old, new, 1), encoding="utf-8")
        print(f"Patched veRL FSDP attention forwarding: {target}")

    # FSDP2 constructs untied models on the meta device. Without PEFT's
    # low-memory loading path, load_state_dict copies adapter tensors into meta
    # parameters (a no-op) and silently discards the SFT initialization.
    source = target.read_text(encoding="utf-8")
    old_peft_load = (
        "module = PeftModel.from_pretrained(module, local_adapter_path, "
        "is_trainable=True)"
    )
    legacy_peft_load = (
        "module = PeftModel.from_pretrained(\n"
        "                module, local_adapter_path, is_trainable=True, "
        "low_cpu_mem_usage=True\n"
        "            )"
    )
    new_peft_load = (
        "base_dtype = next(module.parameters()).dtype\n"
        "            module = PeftModel.from_pretrained(\n"
        "                module, local_adapter_path, is_trainable=True, "
        "low_cpu_mem_usage=True,\n"
        "                autocast_adapter_dtype=False,\n"
        "            )\n"
        "            module.to(base_dtype)"
    )
    if old_peft_load in source:
        target.write_text(
            source.replace(old_peft_load, new_peft_load, 1),
            encoding="utf-8",
        )
        print(f"Patched veRL meta-safe uniform-dtype PEFT loading: {target}")
    elif new_peft_load in source:
        print(f"veRL uniform-dtype PEFT adapter patch already present: {target}")
    elif legacy_peft_load in source:
        target.write_text(
            source.replace(legacy_peft_load, new_peft_load, 1),
            encoding="utf-8",
        )
        print(f"Upgraded veRL PEFT adapter dtype patch: {target}")
    else:
        raise RuntimeError(
            f"Unsupported veRL layout; PEFT adapter load not found in {target}"
        )

    # Transformers 4.56 includes AIMv2 natively. vLLM 0.9's Ovis shim tries
    # to register the same key and needs the idempotent flag.
    site_packages = Path(verl.__file__).resolve().parent.parent
    ovis_config = site_packages / "vllm" / "transformers_utils" / "configs" / "ovis.py"
    if ovis_config.exists():
        ovis_source = ovis_config.read_text(encoding="utf-8")
        old_register = 'AutoConfig.register("aimv2", AIMv2Config)'
        new_register = 'AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)'
        if old_register in ovis_source:
            ovis_config.write_text(
                ovis_source.replace(old_register, new_register, 1),
                encoding="utf-8",
            )
            print(f"Patched vLLM AIMv2 registration: {ovis_config}")
        else:
            print(f"vLLM AIMv2 compatibility patch already present: {ovis_config}")

    # veRL 0.8 targets the newer AsyncLLM helper added after vLLM 0.9.1.
    # vLLM 0.9.1 exposes the same state via OutputProcessor.
    vllm_server = (
        Path(verl.__file__).resolve().parent
        / "workers"
        / "rollout"
        / "vllm_rollout"
        / "vllm_async_server.py"
    )
    server_source = vllm_server.read_text(encoding="utf-8")
    old_drain = """    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()
"""
    new_drain = """    async def wait_for_requests_to_drain(self):
        drain = getattr(self.engine, "wait_for_requests_to_drain", None)
        if drain is not None:
            await drain()
            return
        while self.engine.output_processor.has_unfinished_requests():
            await asyncio.sleep(0.05)
"""
    if old_drain in server_source:
        vllm_server.write_text(
            server_source.replace(old_drain, new_drain, 1),
            encoding="utf-8",
        )
        print(f"Patched veRL request-drain compatibility: {vllm_server}")
    elif new_drain in server_source:
        print(f"veRL request-drain compatibility patch already present: {vllm_server}")
    else:
        raise RuntimeError(
            f"Unsupported veRL layout; request-drain method not found in {vllm_server}"
        )

    # veRL 0.8 creates and releases a BaseTool instance for every tool call.
    # Inject the trajectory request ID so stateful tools can reuse their
    # episode across multiple calls in the same rollout.
    tool_loop = (
        Path(verl.__file__).resolve().parent
        / "experimental"
        / "agent_loop"
        / "tool_agent_loop.py"
    )
    tool_loop_source = tool_loop.read_text(encoding="utf-8")
    old_tool_create = (
        '                instance_id, _ = await tool.create('
        'create_kwargs=kwargs.get("create_kwargs", {}))\n'
    )
    new_tool_create = """                create_kwargs = dict(kwargs.get("create_kwargs", {}))
                create_kwargs["_agent_request_id"] = agent_data.request_id
                instance_id, _ = await tool.create(create_kwargs=create_kwargs)
"""
    if old_tool_create in tool_loop_source:
        tool_loop.write_text(
            tool_loop_source.replace(old_tool_create, new_tool_create, 1),
            encoding="utf-8",
        )
        print(f"Patched veRL stateful tool request IDs: {tool_loop}")
    elif new_tool_create in tool_loop_source:
        print(f"veRL stateful tool request-ID patch already present: {tool_loop}")
    else:
        raise RuntimeError(
            f"Unsupported veRL layout; BaseTool create call not found in {tool_loop}"
        )

    # Some veRL log-prob preprocessing paths use the FlashAttention padding
    # helpers even when the model itself runs SDPA. Transformers ships
    # API-compatible pure PyTorch fallbacks, so use them when flash-attn is not
    # installed instead of forcing a source build on CUDA 11.8.
    attention_utils = (
        Path(verl.__file__).resolve().parent / "utils" / "attention_utils.py"
    )
    attention_source = attention_utils.read_text(encoding="utf-8")
    old_flash_import = (
        "        from flash_attn.bert_padding import "
        "index_first_axis, pad_input, rearrange, unpad_input\n"
    )
    legacy_flash_fallback = """        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ImportError:
            from einops import rearrange
            from transformers.modeling_flash_attention_utils import (
                _index_first_axis as index_first_axis,
                _pad_input as pad_input,
                _unpad_input as unpad_input,
            )
"""
    new_flash_import = legacy_flash_fallback.replace(
        "        try:\n",
        "        # gov-agent-rl: pure PyTorch fallback when flash-attn is absent\n"
        "        try:\n",
        1,
    )
    double_applied_fallback = legacy_flash_fallback.replace(
        old_flash_import, legacy_flash_fallback, 1
    )
    if double_applied_fallback in attention_source:
        attention_utils.write_text(
            attention_source.replace(double_applied_fallback, new_flash_import, 1),
            encoding="utf-8",
        )
        print(f"Repaired duplicate veRL SDPA padding patch: {attention_utils}")
    elif new_flash_import in attention_source:
        print(f"veRL SDPA padding fallback already present: {attention_utils}")
    elif legacy_flash_fallback in attention_source:
        attention_utils.write_text(
            attention_source.replace(legacy_flash_fallback, new_flash_import, 1),
            encoding="utf-8",
        )
        print(f"Marked existing veRL SDPA padding fallback: {attention_utils}")
    elif old_flash_import in attention_source:
        attention_utils.write_text(
            attention_source.replace(old_flash_import, new_flash_import, 1),
            encoding="utf-8",
        )
        print(f"Patched veRL SDPA padding fallback: {attention_utils}")
    else:
        raise RuntimeError(
            f"Unsupported veRL layout; FlashAttention import not found in {attention_utils}"
        )

    # PEFT 0.19 imports all Transformers tensor-parallel styles before it knows
    # whether the loaded model is tensor parallel. Transformers 4.56 does not
    # expose EmbeddingParallel, while this project uses FSDP and never needs
    # that code path. Gracefully skip TP-only sharding when the API is absent.
    peft_save_load = site_packages / "peft" / "utils" / "save_and_load.py"
    if peft_save_load.exists():
        peft_source = peft_save_load.read_text(encoding="utf-8")
        old_tp_import = """    from transformers.integrations.tensor_parallel import (
        ALL_PARALLEL_STYLES,
        ColwiseParallel,
        EmbeddingParallel,
        RowwiseParallel,
    )
"""
        new_tp_import = """    try:
        from transformers.integrations.tensor_parallel import (
            ALL_PARALLEL_STYLES,
            ColwiseParallel,
            EmbeddingParallel,
            RowwiseParallel,
        )
    except ImportError:
        # No TP API is needed for the FSDP model used by this project.
        return
"""
        if old_tp_import in peft_source:
            peft_save_load.write_text(
                peft_source.replace(old_tp_import, new_tp_import, 1),
                encoding="utf-8",
            )
            print(f"Patched PEFT optional tensor-parallel import: {peft_save_load}")
        elif new_tp_import in peft_source:
            print(f"PEFT tensor-parallel compatibility patch already present: {peft_save_load}")
    else:
        raise RuntimeError(
            f"Unsupported PEFT layout; tensor-parallel import not found in {peft_save_load}"
        )

    # The actor and vLLM are colocated. PyTorch's allocator can retain several
    # GB of activation blocks after the PPO step, making vLLM's sleep-mode
    # weight restore fail even though those blocks are no longer live.
    engine_workers = (
        Path(verl.__file__).resolve().parent / "workers" / "engine_workers.py"
    )
    engine_source = engine_workers.read_text(encoding="utf-8")
    old_actor_update = """    def update_actor(self, data: TensorDict) -> TensorDict:
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None
"""
    new_actor_update = """    def update_actor(self, data: TensorDict) -> TensorDict:
        output = self.actor.train_mini_batch(data=data)
        result = output.cpu() if output is not None else None
        # gov-agent-rl: release inactive PPO buffers before waking colocated vLLM
        aggressive_empty_cache(force_sync=True)
        return result
"""
    if old_actor_update in engine_source:
        engine_workers.write_text(
            engine_source.replace(old_actor_update, new_actor_update, 1),
            encoding="utf-8",
        )
        print(f"Patched veRL post-update GPU cache release: {engine_workers}")
    elif new_actor_update in engine_source:
        print(f"veRL post-update GPU cache patch already present: {engine_workers}")
    else:
        raise RuntimeError(
            f"Unsupported veRL layout; actor update method not found in {engine_workers}"
        )

    engine_source = engine_workers.read_text(encoding="utf-8")
    old_actor_load = """    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)
"""
    new_actor_load = """    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)
        # gov-agent-rl: release checkpoint deserialization buffers before vLLM wake-up
        aggressive_empty_cache(force_sync=True)
"""
    checkpoint_cache_block = (
        "        # gov-agent-rl: release checkpoint deserialization buffers "
        "before vLLM wake-up\n"
        "        aggressive_empty_cache(force_sync=True)\n"
    )
    double_actor_load = new_actor_load + checkpoint_cache_block
    if double_actor_load in engine_source:
        engine_workers.write_text(
            engine_source.replace(double_actor_load, new_actor_load, 1),
            encoding="utf-8",
        )
        print(f"Repaired duplicate veRL post-checkpoint cache patch: {engine_workers}")
    elif new_actor_load in engine_source:
        print(f"veRL post-checkpoint GPU cache patch already present: {engine_workers}")
    elif old_actor_load in engine_source:
        engine_workers.write_text(
            engine_source.replace(old_actor_load, new_actor_load, 1),
            encoding="utf-8",
        )
        print(f"Patched veRL post-checkpoint GPU cache release: {engine_workers}")
    else:
        raise RuntimeError(
            f"Unsupported veRL layout; actor checkpoint load method not found in {engine_workers}"
        )

    for patched_file in (
        target,
        vllm_server,
        tool_loop,
        attention_utils,
        peft_save_load,
        engine_workers,
    ):
        compile(
            patched_file.read_text(encoding="utf-8"),
            str(patched_file),
            "exec",
        )
    print("Compatibility patch syntax validation passed")


if __name__ == "__main__":
    main()
