from __future__ import annotations
from types import SimpleNamespace as NS
import sys
import types
import unittest
from unittest.mock import patch,Mock
import torch
import torch.nn.functional as F
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.layers.attention.linear import kda_backend as N
_OLD_CTOR=N.KDAAttnBackend.__init__
from sglang.srt.speculative import spec_utils as SU
_OLD_COMMIT=SU.commit_mamba_states_after_verify
_ALIAS=types.ModuleType('sglang.srt.speculative._kda_commit_consumer')
_ALIAS.commit=_OLD_COMMIT
sys.modules[_ALIAS.__name__]=_ALIAS
from hcu_sglang import register
from hcu_sglang.layers.attention.linear import kda_backend as P
from sglang.srt.plugins.hook_registry import HookRegistry
HookRegistry.reset();register();P.register_hooks();HookRegistry.apply_hooks()
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.configs.mamba_utils import KimiLinearStateShape,KimiLinearCacheParams,Mamba2StateDType
from sglang.srt.layers.attention.linear.utils import LinearAttnBackends,LinearAttnKernelBackend as Backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def fixture(device='cpu',spec=False,dim=16,track=False):
    shape=KimiLinearStateShape.create(tp_world_size=1,num_heads=2,head_dim=dim,conv_kernel_size=4,num_spec=2)
    params=KimiLinearCacheParams(shape=shape,layers=[0],dtype=Mamba2StateDType(conv=torch.bfloat16,temporal=torch.float32))
    pool=HybridReqToTokenPool(size=2,mamba_size=8 if track else 3,mamba_spec_state_size=2,max_context_len=64 if track else 16,device=device,enable_memory_saver=False,cache_params=params,mamba_layer_ids=[0],enable_mamba_extra_buffer=track,speculative_num_draft_tokens=2,speculative_eagle_topk=1,enable_overlap_schedule=False,enable_linear_replayssm_spec=spec,linear_replayssm_cache_len=4)
    pool.mamba_pool.clear_slots(torch.tensor([0,1,2],device=device))
    runner=NS(device=device,is_draft_worker=False,req_to_token_pool=pool,token_to_kv_pool=None,server_args=NS(enable_mis=False),model_config=NS(hf_text_config=NS(mamba_chunk_size=4)),linear_attn_backends=LinearAttnBackends(decode=Backend.TRITON,prefill=Backend.TRITON,verify=Backend.TRITON))
    with patch.object(N,'get_spec',return_value=NS(speculative_eagle_topk=1)),patch.object(P.H,'get_spec',return_value=NS(speculative_eagle_topk=1)),patch.object(P.H,'get_memory',return_value=NS(enable_unified_memory=False)),patch('sglang.srt.utils.common.require_mlp_sync',return_value=False):
        backend=N.KDAAttnBackend(runner)
    backend.forward_metadata=NS(query_start_loc=torch.tensor([0,1,2],dtype=torch.int32,device=device),mamba_cache_indices=torch.tensor([1,2],dtype=torch.int32,device=device),retrieve_next_token=None,retrieve_next_sibling=None,retrieve_parent_token=None,has_mamba_track_mask=False)
    weights=torch.zeros(6*dim,4,device=device,dtype=torch.bfloat16);weights[:,-1]=1
    layer=NS(layer_id=0,q_dim=2*dim,k_dim=2*dim,v_dim=2*dim,head_q_dim=dim,head_k_dim=dim,head_v_dim=dim,conv_weights=weights,bias=None,A_log=torch.zeros(2,device=device),dt_bias=torch.zeros(2*dim,device=device),beta_scale=1.,safe_gate=True,lower_bound=None)
    return runner,backend,layer


def recurrence(x,state,dim,steps):
    x=F.silu(x.float()).to(torch.bfloat16).float()
    q,k,v=x.split([2*dim]*3,dim=-1)
    q=q.view(-1,2,dim);k=k.view(-1,2,dim);v=v.view(-1,2,dim)
    q=q*torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)
    k=k*torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
    out=[];states=[]
    for step in range(steps):
        h=state*.5
        delta=(v[step]-(h*k[step].unsqueeze(-2)).sum(-1))*.5
        state=h+delta.unsqueeze(-1)*k[step].unsqueeze(-2)
        out.append((state*q[step].unsqueeze(-2)).sum(-1)*(dim**-.5));states.append(state.clone())
    return torch.stack(out),states


class KdaConsumerTests(unittest.TestCase):
    def test_real_config_pool_backend_constructor_selects_hcu_and_keeps_fp32_state(self):
        runner,backend,layer=fixture()
        self.assertTrue(runner.req_to_token_pool.mamba_pool.use_hcu_kda)
        self.assertTrue(backend.use_hcu_kda);self.assertIsNone(backend.kernel_dispatcher)
        self.assertEqual(runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.dtype,torch.float32)
        self.assertFalse(backend.supports_ragged_verify_graph)
        old=object.__new__(N.KDAAttnBackend)
        with patch.object(N,'get_spec',return_value=NS(speculative_eagle_topk=1)),patch.object(P.H,'get_spec',return_value=NS(speculative_eagle_topk=1)),patch.object(P.H,'get_memory',return_value=NS(enable_unified_memory=False)),patch('sglang.srt.utils.common.require_mlp_sync',return_value=False):
            _OLD_CTOR(old,runner)
        self.assertIsNotNone(old.kernel_dispatcher)
        self.assertTrue(callable(P._hcu_causal_conv1d_fn));self.assertTrue(callable(P._hcu_causal_conv1d_update))

    def test_generic_commit_preimport_consumer_is_rebound_and_non_mamba_fallback_stays_noop(self):
        self.assertIs(_ALIAS.commit,SU.commit_mamba_states_after_verify)
        with patch.object(SU,'mambaish_config',return_value=None):
            self.assertIsNone(_ALIAS.commit(NS(model_runner=NS(model_config=NS())),NS(),torch.tensor([1]),torch.tensor([[0]]),1))

    def test_hcu_tracked_prefill_metadata_runs_without_a_kernel_dispatcher(self):
        runner,backend,layer=fixture(track=True)
        self.assertIsNone(backend.kernel_dispatcher)
        fb=N.ForwardBatch(forward_mode=ForwardMode.EXTEND,batch_size=1,input_ids=torch.tensor([1,2,3,4]),req_pool_indices=torch.tensor([0]),seq_lens=torch.tensor([4]),out_cache_loc=torch.arange(4),seq_lens_sum=4,extend_num_tokens=4,extend_seq_lens=[4],extend_seq_lens_cpu=[4],extend_prefix_lens=torch.tensor([0]),extend_prefix_lens_cpu=[0],extend_start_loc=torch.tensor([0]),mamba_track_mask=torch.tensor([True]),mamba_track_indices=torch.tensor([1]),mamba_track_seqlens=torch.tensor([4]),mamba_track_seqlens_cpu=[4],mamba_prefill_track_mask_cpu=[True])
        with patch('sglang.srt.model_executor.forward_batch_info.mamba_cache_chunk_size',return_value=4):
            backend.init_forward_metadata(fb)
        metadata=backend.forward_metadata
        self.assertTrue(metadata.has_mamba_track_mask)
        self.assertEqual(metadata.conv_states_mask_indices.tolist(),[1])
        self.assertEqual(metadata.track_ssm_final_src.tolist(),metadata.mamba_cache_indices.tolist())
        self.assertEqual(metadata.num_state_checkpoints,0)

    def test_non_hcu_tracked_prefill_still_builds_the_flashinfer_checkpoint_plan(self):
        runner,backend,layer=fixture(track=True)
        fb=N.ForwardBatch(forward_mode=ForwardMode.EXTEND,batch_size=1,input_ids=torch.tensor([1,2,3,4]),req_pool_indices=torch.tensor([0]),seq_lens=torch.tensor([4]),out_cache_loc=torch.arange(4),seq_lens_sum=4,extend_num_tokens=4,extend_seq_lens=[4],extend_seq_lens_cpu=[4],extend_prefix_lens=torch.tensor([0]),extend_prefix_lens_cpu=[0],extend_start_loc=torch.tensor([0]),mamba_track_mask=torch.tensor([True]),mamba_track_indices=torch.tensor([1]),mamba_track_seqlens=torch.tensor([3]),mamba_track_seqlens_cpu=[3],mamba_prefill_track_mask_cpu=[True])
        plan=Mock()
        backend.use_hcu_kda=False
        backend.kernel_dispatcher=NS(extend_kernel=NS(uses_state_checkpoints=True))
        with patch('sglang.srt.model_executor.forward_batch_info.mamba_cache_chunk_size',return_value=4),patch('sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill.build_flashinfer_kda_checkpoint_plan',plan,create=True):
            backend.init_forward_metadata(fb)
        plan.assert_called_once()

    @unittest.skipUnless(torch.cuda.is_available(),'GPU required')
    def test_decode_executes_actual_hcu_kernel_and_updates_checkpoint_in_value_key_order(self):
        runner,backend,layer=fixture('cuda',dim=16)
        torch.manual_seed(4);x=torch.randn(2,96,device='cuda',dtype=torch.bfloat16)*.2
        a=torch.zeros(2,32,device='cuda');b=torch.zeros(2,2,device='cuda')
        batch=NS(mamba_track_mask=None,forward_mode=ForwardMode.DECODE)
        expected=[]
        for row in range(2):expected.append(recurrence(x[row:row+1],torch.zeros(2,16,16,device='cuda'),16,1))
        out=backend.forward_decode(layer,batch,x,a,b)
        torch.testing.assert_close(out.squeeze(0).float(),torch.stack([v[0][0] for v in expected]),rtol=.025,atol=.003)
        actual=runner.req_to_token_pool.mamba_pool.mamba_cache.temporal[0,torch.tensor([1,2],device='cuda')]
        torch.testing.assert_close(actual,torch.stack([v[1][0] for v in expected]),rtol=.025,atol=.003)
        self.assertGreater(actual.abs().sum().item(),0)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU required')
    def test_prefill_executes_hcu_chunk_kernel_and_zero_pads_inactive_rows(self):
        runner,backend,layer=fixture('cuda',dim=16)
        backend.forward_metadata.query_start_loc=torch.tensor([0,2],device='cuda',dtype=torch.int32)
        backend.forward_metadata.mamba_cache_indices=torch.tensor([1],device='cuda',dtype=torch.int32)
        torch.manual_seed(12);x=torch.randn(4,96,device='cuda',dtype=torch.bfloat16)*.2
        batch=NS(forward_mode=ForwardMode.EXTEND,extend_seq_lens_cpu=[2],extend_num_tokens=2,extend_prefix_lens=torch.zeros(1,device='cuda',dtype=torch.int32),mamba_track_mask=None)
        out=backend.forward_extend(layer,batch,x,torch.zeros(4,32,device='cuda'),torch.zeros(4,2,device='cuda'))
        expected,states=recurrence(x[:2],torch.zeros(2,16,16,device='cuda'),16,2)
        self.assertEqual(tuple(out.shape),(1,4,2,16));self.assertEqual(out[:,2:].count_nonzero().item(),0)
        torch.testing.assert_close(out[0,:2].float(),expected,rtol=.03,atol=.003)
        actual=runner.req_to_token_pool.mamba_pool.mamba_cache.temporal[0,1]
        torch.testing.assert_close(actual,states[-1],rtol=.03,atol=.003)
        self.assertEqual(runner.req_to_token_pool.mamba_pool.mamba_cache.temporal[0,2].abs().sum().item(),0)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU required')
    def test_verify_stages_inputs_then_generic_commit_folds_only_accepted_prefix_and_rolls_back_conv(self):
        runner,backend,layer=fixture('cuda',spec=True,dim=16)
        pool=runner.req_to_token_pool;state=pool.get_speculative_mamba2_params_all_layers()
        self.assertIsNone(state.intermediate_ssm)
        pool.req_index_to_mamba_index_mapping[:2]=torch.tensor([1,2],device='cuda')
        backend.forward_metadata.query_start_loc=torch.tensor([0,2,4],device='cuda',dtype=torch.int32)
        torch.manual_seed(9);x=torch.randn(4,96,device='cuda',dtype=torch.bfloat16)*.2
        batch=NS(forward_mode=ForwardMode.TARGET_VERIFY,spec_info=NS(draft_token_num=2),extend_seq_lens_cpu=None,extend_num_tokens=4,mamba_track_mask=None,mamba_track_indices=None,req_pool_indices=torch.tensor([0,1],device='cuda'),seq_lens=torch.tensor([1,1],device='cuda'))
        out=backend.forward_extend(layer,batch,x,torch.zeros(4,32,device='cuda'),torch.zeros(4,2,device='cuda'))
        self.assertEqual(tuple(out.shape),(1,4,2,16))
        checkpoint=state.temporal.clone()
        self.assertEqual(checkpoint.abs().sum().item(),0)
        self.assertGreater(state.replayssm_rawv.abs().sum().item(),0)
        expected=[]
        for row in range(2):expected.append(recurrence(x[row*2:row*2+2],torch.zeros(2,16,16,device='cuda'),16,2))
        torch.testing.assert_close(out.squeeze(0).float(),torch.cat([v[0] for v in expected]),rtol=.025,atol=.003)
        with patch.object(SU,'mambaish_config',return_value=NS()),patch.object(SU,'pp_spec_stable_rows_enabled',return_value=False):
            _ALIAS.commit(NS(model_runner=NS(model_config=NS(),req_to_token_pool=pool)),batch,torch.tensor([1,2],device='cuda'),torch.tensor([[0,1],[2,3]],device='cuda'),2)
        actual=state.temporal[0,torch.tensor([1,2],device='cuda')]
        torch.testing.assert_close(actual,torch.stack([expected[0][1][0],expected[1][1][1]]),rtol=.025,atol=.003)
        self.assertEqual(state.temporal[0,0].abs().sum().item(),0)
        for row,step in enumerate([0,1]):
            torch.testing.assert_close(state.conv[0][0,row+1],state.intermediate_conv_window[0][0,row,step])

if __name__=='__main__':unittest.main()
