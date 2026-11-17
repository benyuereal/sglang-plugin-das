from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace as S

from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs

register_envs()
from hcu_sglang.disagg import decode as D, scheduler as SC
import sglang.srt.disaggregation.fake.conn as FakeConn
from sglang.srt.managers import scheduler as native


class OfficialHostSchedulerDefaults(unittest.TestCase):
    """PD entry hooks must not depend on fork-only Scheduler members.

    `get_next_disagg_prefill_batch_to_run` calls `_begin_scheduler_iteration`
    before any disaggregation-mode check, and the prefill abort bookkeeping
    touched `_external_kv_{abort,cleanup}_rids` unconditionally. Neither name
    exists in this repository or in official sglang, so on the official
    v0.5.22 line every PD prefill launch died with AttributeError. The hooks
    now install a no-op default and initialise the two sets; these tests pin
    that behaviour and the "host implementation always wins" polarity.
    """

    def test_missing_iteration_hook_gets_a_no_op_default(self):
        had = hasattr(native.Scheduler, "_begin_scheduler_iteration")
        saved = getattr(native.Scheduler, "_begin_scheduler_iteration", None)
        if had:
            del native.Scheduler._begin_scheduler_iteration
        try:
            SC.register_hooks()
            self.assertTrue(hasattr(native.Scheduler, "_begin_scheduler_iteration"))
            self.assertIsNone(
                native.Scheduler._begin_scheduler_iteration(S())
            )
        finally:
            if had:
                native.Scheduler._begin_scheduler_iteration = saved
            elif hasattr(native.Scheduler, "_begin_scheduler_iteration"):
                delattr(native.Scheduler, "_begin_scheduler_iteration")

    def test_host_iteration_hook_is_not_overwritten(self):
        sentinel = object()
        had = hasattr(native.Scheduler, "_begin_scheduler_iteration")
        saved = getattr(native.Scheduler, "_begin_scheduler_iteration", None)
        native.Scheduler._begin_scheduler_iteration = sentinel
        try:
            SC.register_hooks()
            self.assertIs(native.Scheduler._begin_scheduler_iteration, sentinel)
        finally:
            if had:
                native.Scheduler._begin_scheduler_iteration = saved
            elif hasattr(native.Scheduler, "_begin_scheduler_iteration"):
                delattr(native.Scheduler, "_begin_scheduler_iteration")

    def test_init_disaggregation_creates_external_kv_sets(self):
        source = inspect.getsource(SC.init_disaggregation)
        self.assertIn("_external_kv_abort_rids", source)
        self.assertIn("_external_kv_cleanup_rids", source)


class ReceiverMetadataFiltering(unittest.TestCase):
    """`spec_metadata` may only reach receivers that declare it.

    `DecodePreallocQueue._send_kv_metadata` forwarded every metadata kwarg to
    `kv_receiver.send_metadata`. The HCU Mooncake receiver accepts
    `spec_metadata`/`device_kv_indices`, but upstream's FakeKVReceiver (used
    for the server warmup request) accepts neither and ignores its payload,
    so the first decode iteration raised TypeError and killed the scheduler.
    """

    def test_fake_receiver_does_not_accept_spec_metadata(self):
        sig = inspect.signature(FakeConn.FakeKVReceiver.send_metadata)
        self.assertNotIn("spec_metadata", sig.parameters)

    def test_unsupported_kwargs_are_dropped(self):
        receiver = S(
            send_metadata=lambda kv_indices, aux_index=None, decode_prefix_len=None: None
        )
        kept = D._accepted_metadata_kwargs(
            receiver, {"spec_metadata": {"pd_hidden": True}, "decode_prefix_len": 7}
        )
        self.assertEqual(kept, {"decode_prefix_len": 7})

    def test_var_keyword_receiver_keeps_everything(self):
        def send_metadata(kv_indices, aux_index=None, **kwargs):
            return None

        receiver = S(send_metadata=send_metadata)
        kwargs = {"spec_metadata": {"pd_hidden": True}}
        self.assertEqual(D._accepted_metadata_kwargs(receiver, kwargs), kwargs)

    def test_permissive_receiver_keeps_spec_metadata(self):
        def send_metadata(
            kv_indices,
            aux_index=None,
            state_indices=None,
            decode_prefix_len=None,
            device_kv_indices=None,
            spec_metadata=None,
        ):
            return None

        receiver = S(send_metadata=send_metadata)
        kwargs = {"spec_metadata": {"pd_hidden": True}}
        self.assertEqual(D._accepted_metadata_kwargs(receiver, kwargs), kwargs)

    def test_send_kv_metadata_is_hooked(self):
        self.assertIn("DecodePreallocQueue._send_kv_metadata", D._METHODS)


if __name__ == "__main__":
    unittest.main()
