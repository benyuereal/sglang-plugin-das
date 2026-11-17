# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import inspect,unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.models.deepseek.current_v4.execution import core as C

class ConstructionStopped(Exception):pass

class V4ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.models import deepseek_v4 as native, deepseek_v4_nextn as nextn
        from sglang.srt.plugins.hook_registry import HookRegistry
        from hcu_sglang.models.deepseek.deepseek_v4.hooks import decoder, model, causal_lm
        HookRegistry.reset();decoder.register_hooks();model.register_hooks();causal_lm.register_hooks();HookRegistry.apply_hooks()
        cls.public_model_init=staticmethod(native.DeepseekV4Model.__init__)
        cls.public_causal_init=staticmethod(native.DeepseekV4ForCausalLM.__init__)
        cls.decoder_alias=nextn.DeepseekV4DecoderLayer

    def test_nextn_constructor_alias_uses_current_platform_layer(self):
        self.assertIs(self.decoder_alias,C.DeepseekV4PlatformDecoderLayer)

    def test_hcu_model_uses_live_pp_group_and_five_streams(self):
        self.check_hcu_model_constructor(mock_platform=True)

    @unittest.skipUnless(C._is_hcu, 'requires HCU hardware')
    def test_hcu_model_constructor_with_actual_platform_dispatch(self):
        self.check_hcu_model_constructor(mock_platform=False)

    def check_hcu_model_constructor(self, *, mock_platform):
        config=SimpleNamespace(hidden_size=8,vocab_size=16,rms_norm_eps=1e-6,num_hidden_layers=1,hc_mult=4,hc_eps=1e-5,hc_pre_from_prev_sublayer=False,model_type='deepseek_v4',vision_n_layers=0)
        pp=SimpleNamespace(is_first_rank=True,is_last_rank=True);streams=[];calls=[];factory_signature=inspect.signature(C.make_pp_layers)
        def stream():
            obj=object();streams.append(obj);return obj
        def make_layers(*args,**kwargs):
            factory_signature.bind(*args,**kwargs)
            calls.append((args,kwargs));raise ConstructionStopped()
        def embedding(vocab,size,**kwargs):return torch.nn.Embedding(vocab,size)
        obj=object.__new__(C.NativeDeepseekV4Model)
        with ExitStack() as stack:
            stack.enter_context(patch.object(C.NativeDeepseekV4Model,'__init__',side_effect=AssertionError('old parent constructor was invoked')))
            if mock_platform:
                stack.enter_context(patch.object(C,'_is_cuda',False));stack.enter_context(patch.object(C,'_is_hcu',True))
            stack.enter_context(patch.object(C,'get_parallel',return_value=SimpleNamespace(pp_group=pp)))
            stack.enter_context(patch.object(C,'is_dp_attention_enabled',return_value=False))
            stack.enter_context(patch.object(C,'VocabParallelEmbedding',side_effect=embedding))
            stack.enter_context(patch.object(C.EngramLayout,'from_config',return_value=None))
            stack.enter_context(patch.object(torch,'get_device_module',return_value=SimpleNamespace(Stream=stream)))
            stack.enter_context(patch.object(C,'make_pp_layers',side_effect=make_layers))
            with self.assertRaises(ConstructionStopped):self.public_model_init(obj,config)
        self.assertIs(obj.pp_group,pp);self.assertEqual(len(streams),5)
        self.assertEqual(obj.alt_streams,streams);self.assertEqual(len(calls),1)
        self.assertIsNone(calls[0][1]['final_read'])
        self.assertIsNone(calls[0][1]['stage_facts'])
        self.assertIs(C.DeepseekV4DecoderLayer,C.DeepseekV4PlatformDecoderLayer)

    def test_native_model_public_forward_reaches_owned_body(self):
        model=object.__new__(C.NativeDeepseekV4Model)
        calls=[]
        marker=object()
        def body(owner,*args):
            calls.append((owner,args))
            return marker
        args=tuple(object() for _ in range(5))
        with patch.object(C,'_is_cuda',False),patch.object(C.NativeDeepseekV4Model,'_forward_impl',body):
            self.assertIs(model.forward(*args),marker)
        self.assertEqual(calls,[(model,args)])

    def test_native_model_public_forward_executes_owned_pp_body(self):
        model=object.__new__(C.NativeDeepseekV4Model)
        torch.nn.Module.__init__(model)
        model.pp_group=SimpleNamespace(is_first_rank=False,is_last_rank=False,world_size=3)
        model.hc_mult=2;model.hidden_size=3;model.dspark_layers_to_capture=None
        model.start_layer=0;model.end_layer=0;model.hc_pre_from_prev_sublayer=False
        model.use_fused_mhc_post_pre=False;model._can_run_tbo=lambda batch:False
        hidden=torch.arange(12.).reshape(2,6)
        proxy=C.PPProxyTensors({'hidden_states':hidden})
        fb=SimpleNamespace()
        with ExitStack() as stack:
            stack.enter_context(patch.object(C,'_is_cuda',False))
            stack.enter_context(patch.object(C,'_is_npu',False))
            stack.enter_context(patch.object(C,'is_cp_v2_active',return_value=False))
            stack.enter_context(patch.object(C,'dsa_use_prefill_cp',return_value=False))
            stack.enter_context(patch.object(C,'get_parallel',return_value=SimpleNamespace(attn_dp_size=1)))
            stack.enter_context(patch.object(C,'get_attn_backend',return_value=SimpleNamespace()))
            output=model.forward(torch.tensor([1,2]),torch.tensor([0,1]),fb,None,proxy)
        self.assertIsInstance(output,C.PPProxyTensors)
        torch.testing.assert_close(output['hidden_states'],hidden)
        self.assertEqual(output['hidden_states'].data_ptr(),hidden.data_ptr())

    def test_platform_mixin_initializes_state_after_base(self):
        calls=[]
        def base(self,*args,**kwargs):
            torch.nn.Module.__init__(self);calls.append((args,kwargs));self.hc_pre_from_prev_sublayer=False
        with patch.object(C.DeepseekV4DecoderLayerBase,'__init__',base):
            layer=C.DeepseekV4PlatformDecoderLayer('config',layer_id=3)
        self.assertEqual(calls,[(('config',),{'layer_id':3})])
        self.assertIsNone(layer._next_layer);self.assertIsNone(layer.local_boundary);self.assertIsNone(layer.next_boundary)

    def test_native_causal_instance_reaches_current_model_constructor(self):
        config=SimpleNamespace(model_type='deepseek_v4',hidden_size=8,vocab_size=16,vision_n_layers=0)
        pp=SimpleNamespace(is_first_rank=True,is_last_rank=True);calls=[]
        def model(*args,**kwargs):calls.append((args,kwargs));raise ConstructionStopped()
        obj=object.__new__(C.NativeDeepseekV4ForCausalLM)
        with patch.object(C.NativeDeepseekV4ForCausalLM,'__init__',side_effect=AssertionError('old parent constructor was invoked')),patch.object(C,'get_parallel',return_value=SimpleNamespace(pp_group=pp,tp_size=4)),patch.object(C.NativeDeepseekV4ForCausalLM,'determine_num_fused_shared_experts',lambda owner:setattr(owner,'num_fused_shared_experts',0)),patch.object(C,'DeepseekV4Model',side_effect=model):
            with self.assertRaises(ConstructionStopped):self.public_causal_init(obj,config)
        self.assertEqual(len(calls),1)

class MhcSharedTypesTests(unittest.TestCase):
    def test_target_attention_and_execution_share_state_identity(self):
        from sglang.srt.models import deepseek_v4_mhc as native
        from hcu_sglang.models.deepseek.current_v4 import attention
        for name in ['HcConfig','HcSubLayer','HcNormed','HcQuantized','AttnOutput','HcPending','HcState','HcNextBoundary']:
            self.assertIs(getattr(C.mhc,name),getattr(native,name))
        self.assertIs(attention.mhc.AttnOutput,C.mhc.AttnOutput)

if __name__=='__main__':unittest.main()
