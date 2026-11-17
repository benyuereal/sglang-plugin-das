from types import SimpleNamespace as NS
from unittest.mock import patch
from contextlib import ExitStack
import unittest
import torch
from torch import nn
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang import speculative_dspark_worker as S
BEFORE=(S.M.DeepseekV4ForCausalLMDSpark,S.I.TargetHiddenKvInjector,S.W.DSparkWorkerV2)
S.register_hooks()
M,I,W=S.M,S.I,S.W

class Projection(nn.Module):
    def __init__(self,weight):
        super().__init__();self.weight=nn.Parameter(weight);self.quant_method=NS()
    def forward(self,x):return torch.nn.functional.linear(x,self.weight),None

class Norm(nn.Module):
    def __init__(self,n):super().__init__();self.weight=nn.Parameter(torch.ones(n));self.eps=1e-6
    def forward(self,x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+self.eps)*self.weight

class Pool:
    swa_req_ring_size=None
    uniform_fp8=False
    def __init__(self):self.data=torch.zeros(32,2);self.calls=[]
    def translate_loc_from_full_to_swa(self,loc):return loc
    def set_swa_key_buffer_radix_fused_norm_rope(self,**kw):
        self.calls.append(kw)
        self.data[kw['swa_loc'].long()]=kw['kv']

class DsParkHiddenConsumerTests(unittest.TestCase):
    @unittest.skipUnless(S._is_hcu, 'requires HCU hardware')
    def test_tp_draft_constructor_with_actual_platform_dispatch(self):
        cfg=NS(hidden_size=2,num_hidden_layers=8,hc_mult=1,hc_eps=1e-6,rms_norm_eps=1e-6,n_shared_experts=0,vocab_size=16,dspark_target_layer_ids=[6,7],dspark_markov_rank=4,dspark_block_size=2,dspark_noise_token_id=0)
        class StageReached(Exception):pass
        with patch.object(M,'model_parallel_is_initialized',return_value=True),patch.object(M,'get_parallel',return_value=NS(pp_rank=0,pp_size=1)),patch.object(M,'get_disagg',return_value=NS(disaggregation_mode='null')),patch.object(M,'is_shared_experts_fusion_disabled',return_value=False),patch.object(M,'DSparkV4Stage',side_effect=StageReached) as stage:
            with self.assertRaises(StageReached):M.DeepseekV4ForCausalLMDSpark(cfg)
        self.assertEqual(stage.call_count,1)

    @unittest.skipUnless(S._is_hcu, 'requires HCU hardware')
    def test_tp_draft_forward_preserves_platform_stage_state(self):
        from contextlib import nullcontext
        calls=[]
        class Stage(nn.Module):
            def forward(self,positions,x,batch,pre):
                calls.append(pre)
                return x+2,'state'
        model=object.__new__(M.DeepseekV4ForCausalLMDSpark);nn.Module.__init__(model)
        model.stages=nn.ModuleList([Stage(),Stage()]);model.hc_pre_from_prev_sublayer=False
        values=torch.tensor([[1.,3.]])
        with patch.object(M,'get_global_expert_distribution_recorder',return_value=NS(disable_this_region=nullcontext)):
            result=model(torch.tensor([0]),torch.tensor([0]),NS(),input_embeds=values)
        self.assertEqual(calls,[None,'state'])
        torch.testing.assert_close(result.hidden_states,values+4)

    def model(self):
        model=object.__new__(M.DeepseekV4ForCausalLMDSpark);nn.Module.__init__(model)
        model.config=NS(hidden_size=2);model.num_target_features=2
        stage=nn.Module();stage.main_proj=Projection(torch.tensor([[1.,2.,3.,4.],[4.,3.,2.,1.]]));stage.main_norm=Norm(2)
        stage.self_attn=nn.Module();stage.self_attn.wkv=Projection(torch.tensor([[2.,0.],[0.,3.]]))
        stage.self_attn.layer_id=0;stage.self_attn.kv_norm=Norm(2);stage.self_attn.eps=1e-6
        stage.self_attn.freqs_cis=torch.ones(16,1,dtype=torch.complex64)
        model.stages=nn.ModuleList([stage]);model._partial_main_proj=None;model._partial_feature_indices=()
        return model

    def test_native_classes_preserve_identity_and_ring_commit_masks(self):
        self.assertIs(M.DeepseekV4ForCausalLMDSpark,BEFORE[0]);self.assertIs(I.TargetHiddenKvInjector,BEFORE[1]);self.assertIs(W.DSparkWorkerV2,BEFORE[2])
        layout=S.build_ring_commit_inject_layout(req_pool_indices=torch.tensor([2,4]),prefix_lens=torch.tensor([5,9]),block_pos_offsets=torch.arange(3),commit_lens=torch.tensor([2,1]),stride=3,ring_stride=8)
        self.assertEqual(layout.swa_loc.tolist(),[21,22,-1,33,-1,-1])
        self.assertEqual(layout.positions.tolist(),[5,6,7,9,10,11])

    def test_worker_streaming_chunks_reach_real_draft_projection_and_kv_writer(self):
        model=self.model();pool=Pool();row=torch.tensor([[5,14,17,9,11,3]])
        target_runner=NS(device=torch.device('cpu'),req_to_token_pool=NS(req_to_token=row))
        draft_runner=NS(token_to_kv_pool=pool)
        injector=I.TargetHiddenKvInjector(draft_model=model,draft_model_runner=draft_runner,model_runner=target_runner,device='cpu',verify_num_draft_tokens=3,block_pos_offsets=torch.arange(3))
        worker=object.__new__(W.DSparkWorkerV2)
        worker.device='cpu';worker.model_runner=target_runner;worker._kv_injector=injector
        req=NS(kv=NS(req_pool_idx=0));hidden=torch.tensor([[1.,2.,3.,4.],[4.,3.,2.,1.],[1.,1.,1.,1.]])
        with patch.object(torch.cuda,'is_available',return_value=False),patch.object(M,'get_platform',return_value=NS(is_blackwell=False)),patch.object(M,'is_unified_kv_triton',return_value=False):
            self.assertIsNone(worker.inject_pd_hidden_chunk(req,hidden[:2],1))
            self.assertIsNone(worker.inject_pd_hidden_chunk(req,hidden[2:],3))
        projected=model.stages[0].main_norm(model.stages[0].main_proj(hidden)[0])
        expected=model.stages[0].self_attn.wkv(projected)[0]
        torch.testing.assert_close(pool.data[torch.tensor([14,17,9])],expected)
        self.assertEqual(pool.calls[0]['positions'].tolist(),[1,2]);self.assertEqual(pool.calls[1]['positions'].tolist(),[3])
        self.assertEqual(pool.data[torch.tensor([0,5,11,31])].abs().sum().item(),0)

    def test_partial_projection_zero_fills_nonlocal_features_and_validates_shape(self):
        model=self.model();local=torch.tensor([[5.,6.]])
        model.prepare_target_hidden_partial([1])
        self.assertIsNone(model._partial_main_proj)
        actual=model.project_target_hidden_partial(local,[1])
        torch.testing.assert_close(actual,model.stages[0].main_proj(torch.tensor([[0.,0.,5.,6.]]))[0])
        with self.assertRaisesRegex(ValueError,'feature dim'):model.project_target_hidden_partial(torch.ones(1,3),[1])
        with self.assertRaisesRegex(ValueError,'out of range'):model.project_target_hidden_partial(local,[2])

    def test_model_constructor_creates_weightless_lifecycle_pp_rank(self):
        cfg=NS(hidden_size=2,num_hidden_layers=8,hc_mult=1,hc_eps=1e-6,rms_norm_eps=1e-6,n_shared_experts=0,vocab_size=16,dspark_target_layer_ids=[6,7],dspark_markov_rank=4,dspark_block_size=2,dspark_noise_token_id=0)
        with patch.object(M,'model_parallel_is_initialized',return_value=True),patch.object(M,'get_parallel',return_value=NS(pp_rank=0,pp_size=2)),patch.object(M,'get_disagg',return_value=NS(disaggregation_mode='prefill')),patch.object(M,'is_shared_experts_fusion_disabled',return_value=False):
            model=M.DeepseekV4ForCausalLMDSpark(cfg)
        self.assertTrue(model.is_lifecycle_only);self.assertEqual(len(model.stages),0)
        self.assertEqual(list(model.parameters()),[])

    def test_worker_constructor_installs_lifecycle_state_without_injector_or_proposer(self):
        cfg=NS(hidden_size=2,num_hidden_layers=8,hc_mult=1,hc_eps=1e-6,rms_norm_eps=1e-6,n_shared_experts=0,vocab_size=16,dspark_target_layer_ids=[6,7],dspark_markov_rank=4,dspark_block_size=2,dspark_noise_token_id=0)
        ps=NS(pp_rank=0,pp_size=2,tp_rank=0,tp_size=1,attn_dp_enabled=False,attn_tp_size=1,pp_group=NS(is_last_rank=False))
        with patch.object(M,'model_parallel_is_initialized',return_value=True),patch.object(M,'get_parallel',return_value=ps),patch.object(M,'get_disagg',return_value=NS(disaggregation_mode='prefill')),patch.object(M,'is_shared_experts_fusion_disabled',return_value=False):
            draft=M.DeepseekV4ForCausalLMDSpark(cfg)
        mc=NS(hf_text_config=cfg,hf_config=cfg,vocab_size=16,num_hidden_layers=8)
        target=NS(device='cpu',random_seed=1,model_runner=NS(model_config=mc,spec_aux_config=NS(dflash_target_layer_ids=[6,7])))
        bundle=NS(draft_worker=NS(),draft_model_runner=NS(model_config=mc),draft_model=draft)
        with patch.object(W,'get_schedule',return_value=NS(page_size=2)),patch.object(W,'get_parallel',return_value=ps),patch.object(W,'draft_is_deepseek_v4',return_value=True),patch.object(W,'get_disagg',return_value=NS(disaggregation_mode='prefill')),patch.object(W,'get_exec',return_value=NS(graph=NS(cuda_graph_config=NS(decode=NS(backend=W.Backend.DISABLED))))),patch.object(W,'get_pp_stage_load_group',return_value=None),patch.object(W,'build_draft_tp_worker',return_value=bundle),patch.object(W,'mambaish_config',return_value=None),patch.object(W,'get_spec',return_value=NS(speculative_num_draft_tokens=3)):
            worker=W.DSparkWorkerV2(NS(),0,1234,target)
        self.assertTrue(worker.is_lifecycle_only_pp_prefill_rank)
        self.assertIsNone(worker._kv_injector);self.assertIsNone(worker._proposer)
        self.assertEqual(worker.weight_update_runners(),[])
        self.assertFalse(worker.carries_confidence)

    def test_injector_rejects_location_mismatch_before_model_mutation(self):
        model=self.model();pool=Pool();injector=I.TargetHiddenKvInjector(draft_model=model,draft_model_runner=NS(token_to_kv_pool=pool),model_runner=NS(device='cpu'),device='cpu',verify_num_draft_tokens=3,block_pos_offsets=torch.arange(3))
        with self.assertRaisesRegex(ValueError,'one hidden row'):
            injector.inject_target_hidden(target_hidden=torch.ones(2,4),cache_loc=torch.tensor([1]),positions=torch.tensor([1,2]))
        self.assertEqual(pool.calls,[]);self.assertEqual(pool.data.abs().sum().item(),0)

if __name__=='__main__':unittest.main()
