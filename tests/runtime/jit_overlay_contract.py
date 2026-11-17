from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from hcu_sglang import jit_overlay_loader as C
from sglang.kernels.jit.utils import load_jit as early_load
from sglang.kernels.jit.utils.compile import loader as N, spec as S
from sglang.kernels.jit.utils.compile.paths import DEFAULT_INCLUDE,KERNEL_PATH
from sglang.srt.plugins.hook_registry import HookRegistry


class CapturedSpec(Exception):pass


class JitOverlayContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();C.register_hooks();HookRegistry.apply_hooks()

    def capture(self,**kwargs):
        specs=[]
        def key(spec,build_file):specs.append((spec,build_file));raise CapturedSpec()
        with patch.object(N.cache,'compute_build_key',side_effect=key):
            with self.assertRaises(CapturedSpec):early_load('owned_resource_contract',**kwargs)
        self.assertEqual(len(specs),1);return specs[0]

    def test_actual_native_buildspec_uses_owned_source_and_first_header_root(self):
        spec,ninja=self.capture(cuda_files=['deepseek_v4/c_plan.cuh'],cuda_wrappers=[('plan','CompressPlanKernel<1>::run')])
        self.assertEqual(Path(spec.cuda_files[0]),C._OVERLAY_ROOT/'csrc/deepseek_v4/c_plan.cuh')
        self.assertEqual(spec.include_paths[0],str(C._OVERLAY_ROOT/'include'))
        self.assertIn(str(C._OVERLAY_ROOT/'csrc/deepseek_v4/c_plan.cuh'),spec.translation_units()[0].source)
        self.assertIn('cuda.cu',ninja)

    def test_overlay_flags_precede_loader_private_default_includes(self):
        with tempfile.TemporaryDirectory() as folder:
            cached_include = str(Path(folder) / "adapter-cache" / "include")
            overlay_flag = f"-I{C._OVERLAY_ROOT / 'include'}"
            with patch.object(N, "DEFAULT_INCLUDE", [cached_include]):
                spec, ninja = self.capture(
                    cuda_files=["deepseek_v4/c_plan.cuh"],
                    extra_cuda_cflags=["-DKEEP_CALLER_FLAG"],
                )
            self.assertEqual(spec.include_paths[0], cached_include)
            flags = next(line for line in ninja.splitlines() if line.startswith("cudaflags ="))
            self.assertLess(flags.index(overlay_flag), flags.index(f"-I{cached_include}"))
            self.assertIn("-DKEEP_CALLER_FLAG", flags)

    def test_explicit_external_source_is_preserved_and_unowned_source_uses_effective_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            external=Path(folder)/'external.cuh';external.write_text('int external;\n')
            spec,_=self.capture(cuda_files=[str(external)])
            self.assertEqual(Path(spec.cuda_files[0]),external)
        candidates=[p for p in (KERNEL_PATH/'csrc').rglob('*.cuh') if not (C._OVERLAY_ROOT/'csrc'/p.relative_to(KERNEL_PATH/'csrc')).is_file()]
        self.assertTrue(candidates)
        relative=candidates[0].relative_to(KERNEL_PATH/'csrc').as_posix()
        expected=Path(S.resolve_sources([relative])[0])
        spec,_=self.capture(cuda_files=[relative]);self.assertEqual(Path(spec.cuda_files[0]),expected)

    def test_header_preference_is_idempotent_and_preserves_native_include_entries(self):
        before=list(DEFAULT_INCLUDE);C.prefer_overlay_headers();C.prefer_overlay_headers()
        self.assertEqual(DEFAULT_INCLUDE.count(str(C._OVERLAY_ROOT/'include')),1)
        self.assertEqual(DEFAULT_INCLUDE[1:],before[1:])

    def test_non_hcu_effective_original_and_caller_keyword_lists_are_preserved(self):
        original=Mock(return_value='native');files=['deepseek_v4/c_plan.cuh'];includes=['caller-include']
        with patch.object(C,'is_hcu',return_value=False):
            self.assertEqual(C.around_load_jit(original,'name',cuda_files=files,extra_include_paths=includes),'native')
        self.assertIs(original.call_args.kwargs['cuda_files'],files);self.assertIs(original.call_args.kwargs['extra_include_paths'],includes)
        original=Mock(return_value='owned')
        C.around_load_jit(original,'name',cuda_files=files,extra_include_paths=includes)
        self.assertEqual(files,['deepseek_v4/c_plan.cuh']);self.assertEqual(includes,['caller-include'])


if __name__=='__main__':unittest.main()
