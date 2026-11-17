import unittest

import torch

from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()

from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class TopkV2GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset()
        register()
        HookRegistry.apply_hooks()
        from hcu_sglang.models.deepseek.current_v4.hcu_backend import topk
        cls.api=topk

    def data(self):
        scores=torch.stack([torch.arange(-2048,2048).roll(i*137) for i in range(4)]).to(device='cuda',dtype=torch.float32)
        lens=torch.tensor([0,7,1019,4096],device='cuda',dtype=torch.int32)
        pages=(torch.arange(1024,device='cuda').reshape(4,256)+11).to(torch.int32)
        out=torch.full((4,128),-777,device='cuda',dtype=torch.int32)
        raw=torch.empty_like(out)
        metadata=torch.full((5,2),-1,device='cuda',dtype=torch.int32)
        return scores,lens,pages,out,raw,metadata

    def run_kernel(self,scores,lens,pages,out,raw,metadata):
        self.api.plan_topk_v2(lens,out=metadata)
        self.api.topk_transform_paged_v2(scores,lens,pages,out,16,metadata,raw)

    def verify(self,scores,lens,pages,out,raw,metadata):
        for row,length in enumerate(lens.cpu().tolist()):
            count=min(out.shape[1],length)
            selected=scores[row,:length].topk(count,sorted=False).indices
            slots=(pages[row].long()[selected//16]*16+selected%16).to(torch.int32)
            self.assertEqual(int((out[row]<0).sum()),out.shape[1]-count)
            self.assertEqual(int((raw[row]<0).sum()),out.shape[1]-count)
            torch.testing.assert_close(out[row][out[row]>=0].sort().values,slots.sort().values,rtol=0,atol=0)
            torch.testing.assert_close(raw[row][raw[row]>=0].sort().values,selected.to(torch.int32).sort().values,rtol=0,atol=0)

    def test_real_jit_paged_topk_and_raw_indices(self):
        data=self.data()
        self.run_kernel(*data)
        torch.cuda.synchronize()
        self.verify(*data)

    def test_graph_replay_changes_scores_lengths_and_page_table(self):
        data=self.data()
        for _ in range(3):self.run_kernel(*data)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):self.run_kernel(*data)
        scores,lens,pages,out,raw,metadata=data
        for length in [17,2011]:
            scores.copy_(scores.roll(193,-1))
            lens.fill_(length)
            pages.add_(7)
            out.fill_(-777)
            raw.fill_(-777)
            graph.replay()
            torch.cuda.synchronize()
            self.verify(*data)


if __name__=='__main__':unittest.main()
