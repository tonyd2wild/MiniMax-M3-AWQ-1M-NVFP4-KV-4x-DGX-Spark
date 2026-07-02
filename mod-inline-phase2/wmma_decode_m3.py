# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LaNarde "Tony" DeAngelo (github.com/tonyd2wild) — 2Wild
#
# Custom WMMA (tensor-core) flash-decode kernel for the MiniMax-M3 packed
# nvfp4 KV cache on GB10/sm121 — adapted from the GB10-proven MiMo
# ``wmma_decode.py`` (nvfp4-kv-diffkv recipe, measured 2.3x over the Triton
# inline-dequant path in the long-context decode regime, rel-err ~quant noise).
#
# M3 adaptations vs MiMo:
#   * head sizes: Hk = Hv = 128 (MiMo was DiffKV 192/128); G = 16 q-heads
#     per kv-head is invariant at any TP (64/4 global).
#   * cache layout: TWO planes (K = kv_cache[:,0], V = kv_cache[:,1]) of
#     uint8 [num_blocks, BS, NKVH, 72] each; row = [fp4 64B | fp8 scales 8B].
#     MiMo packed K+V in ONE 180-byte row. The kernel therefore takes two
#     base pointers + an explicit block stride (the planes are strided views
#     of the same buffer: block stride = 2*BS*NKVH*72 elements).
#   * BS (KV page size) templated over {16, 32, 64, 128} (M3 full-attention
#     layers may use any multiple-of-16 page; sparse layers pin 128).
#
# Gated by VLLM_M3_WMMA_DECODE (default OFF for M3): offline bench on GB10
# showed the Triton inline-dequant path BEATS this kernel at every context
# length tested (16K/64K/262K, 2-4x) for M3's TP=4 shape — NKVH=1 per rank
# gives the split-K grid only (1, NSPLIT, num_seqs) blocks of ONE warp each,
# unlike MiMo (NKVH=2, Hk=192) where it won 2.3x. Kept for experimentation
# (correctness verified: rel ~1.6e-3 vs Triton on bs 16/64/128 + MTP q_len 3).
# Falls back (returns False) for any unsupported shape / SWA / sinks /
# softcap / prefill, so the engine stays correct either way.
import os

import torch

_M = None
_OK = None  # tri-state: None=untried, True=compiled, False=failed (stop retrying)
_CALLS = 0  # invocation counter for verification
_DBG = 0    # one-time gate-decision diagnostics

# MiniMax-M3 per-rank shapes (head_dim 128 both sides; GQA 64/4 -> G=16).
_HK, _HV, _SB = 128, 128, 72
_G = 16

# --- cudagraph-capturable config -------------------------------------------------
# NSPLIT is FIXED to a process-constant so the kernel grid (grid.y == NSPLIT) is
# static per batch size -> a CUDA graph can capture the launch. Empty per-split
# ranges (j0>=j1) are no-ops and L<=0 writes neutral partials, so a fixed-large
# NSPLIT is numerically IDENTICAL for short contexts, just with empty blocks.
_NSPLIT_MAX = max(8, min(512, int(os.environ.get("VLLM_M3_WMMA_NSPLIT", "512"))))
# Generous per-rank decode batch cap for the static partial-stat scratch (grows
# in eager warmup only; never inside a capture).
_MAX_BATCH = max(1, int(os.environ.get("VLLM_M3_WMMA_MAX_BATCH", "64")))

_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
using namespace nvcuda;
__device__ __forceinline__ float e2m1(unsigned int n){
  const float L[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
  return L[n&15]; }
__device__ __forceinline__ float fp8d(unsigned char b){ __nv_fp8_e4m3 v; *reinterpret_cast<unsigned char*>(&v)=b; return (float)v; }
#define MT 16
#define NT 16
// Hk == Hv == 128, SB == 72 for M3.  K and V live in SEPARATE planes of the
// same buffer; kc/vc are the plane base pointers and blkStride the element
// stride between consecutive physical blocks WITHIN a plane.
template<int Hk,int Hv,int G,int BS,int SB>
__global__ void wmma_dec_m3(const __nv_bfloat16* __restrict__ q,
    const unsigned char* __restrict__ kc,const unsigned char* __restrict__ vc,
    const int* __restrict__ bt,const int* __restrict__ seqused,
    float* __restrict__ pm,float* __restrict__ pl,float* __restrict__ pa,
    int NQH,int NKVH,int NSPLIT,int maxblk,long long blkStride,float scale){
  int kh=blockIdx.x, sp=blockIdx.y, seq=blockIdx.z, lane=threadIdx.x;
  int L=seqused[seq];
  const int Kfb=Hk/2;                 // 64 fp4 data bytes
  const int KS=Kfb;                   // fp8 scales at row offset 64
  const int Vfb=Hv/2;
  const int VS=Vfb;
  const int* bt_s = bt + (size_t)seq*maxblk;
  const __nv_bfloat16* q_s = q + (size_t)seq*NQH*Hk;
  __shared__ __nv_bfloat16 Qs[MT*Hk];
  __shared__ __nv_bfloat16 KVt[NT*Hk];
  __shared__ float Ssh[MT*NT];
  __shared__ __nv_bfloat16 Psh[MT*NT];
  for(int i=lane;i<MT*Hk;i+=32){ int r=i/Hk,c=i%Hk;
    Qs[i]=(r<G)? q_s[((size_t)(kh*G+r))*Hk+c] : __float2bfloat16(0.f); }
  __shared__ float rm[MT], rl[MT];
  __shared__ float acc[G*Hv];
  for(int i=lane;i<MT;i+=32){ rm[i]=-1e30f; rl[i]=0.f; }
  for(int i=lane;i<G*Hv;i+=32) acc[i]=0.f;
  __syncthreads();
  wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> qf[Hk/16];
  #pragma unroll
  for(int kcf=0;kcf<Hk/16;kcf++) wmma::load_matrix_sync(qf[kcf], Qs+kcf*16, Hk);
  if(L<=0){ // empty seq: write neutral partials
    int hb=(seq*NQH);
    for(int r=lane;r<G;r+=32){ size_t idx=(size_t)((hb+kh*G+r)*NSPLIT+sp); pm[idx]=-1e30f; pl[idx]=0.f; }
    for(int i=lane;i<G*Hv;i+=32){ int r=i/Hv,d=i%Hv; size_t idx=(size_t)((hb+kh*G+r)*NSPLIT+sp); pa[idx*Hv+d]=0.f; }
    return; }
  int per=((L+NSPLIT-1)/NSPLIT + NT-1)/NT*NT; int j0=sp*per; int j1=min(L,j0+per);
  const int BSL=(BS==16)?4:(BS==32)?5:(BS==64)?6:7;
  for(int jt=j0; jt<j1; jt+=NT){
    int nv=min(NT,j1-jt);
    const int KW=Hk/8;
    for(int i=lane;i<NT*KW;i+=32){ int n=i/KW,w=i%KW; int base=n*Hk+8*w;
      if(n<nv){ int p=jt+n; long long phys=bt_s[p>>BSL];
        const unsigned char* rb=kc+(size_t)phys*blkStride+((size_t)((p&(BS-1))*NKVH)+kh)*SB;
        unsigned int pk=*reinterpret_cast<const unsigned int*>(rb+4*w); float s=fp8d((rb+KS)[w>>1]);
        #pragma unroll
        for(int t2=0;t2<8;t2++) KVt[base+t2]=__float2bfloat16(e2m1(pk>>(4*t2))*s);
      } else { for(int t2=0;t2<8;t2++) KVt[base+t2]=__float2bfloat16(0.f); } }
    __syncthreads();
    wmma::fragment<wmma::accumulator,16,16,16,float> cf; wmma::fill_fragment(cf,0.f);
    #pragma unroll
    for(int kcf=0;kcf<Hk;kcf+=16){
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bfr;
      wmma::load_matrix_sync(bfr, KVt+kcf, Hk);
      wmma::mma_sync(cf, qf[kcf/16], bfr, cf); }
    wmma::store_matrix_sync(Ssh, cf, NT, wmma::mem_row_major);
    __syncthreads();
    for(int r=lane;r<MT;r+=32){
      float mr=rm[r], mloc=-1e30f;
      for(int n=0;n<nv;n++){ float s=Ssh[r*NT+n]*scale; Ssh[r*NT+n]=s; mloc=fmaxf(mloc,s); }
      float mnew=fmaxf(mr,mloc); float corr=__expf(mr-mnew); float lsum=rl[r]*corr;
      for(int n=0;n<nv;n++){ float p=__expf(Ssh[r*NT+n]-mnew); Psh[r*NT+n]=__float2bfloat16(p); lsum+=p; }
      for(int n=nv;n<NT;n++) Psh[r*NT+n]=__float2bfloat16(0.f);
      if(r<G) for(int d=0;d<Hv;d++) acc[r*Hv+d]*=corr;
      rm[r]=mnew; rl[r]=lsum; }
    __syncthreads();
    const int VW=Hv/8;
    for(int i=lane;i<NT*VW;i+=32){ int n=i/VW,w=i%VW; int base=n*Hv+8*w;
      if(n<nv){ int p=jt+n; long long phys=bt_s[p>>BSL];
        const unsigned char* rb=vc+(size_t)phys*blkStride+((size_t)((p&(BS-1))*NKVH)+kh)*SB;
        unsigned int pv=*reinterpret_cast<const unsigned int*>(rb+4*w); float s=fp8d((rb+VS)[w>>1]);
        #pragma unroll
        for(int t2=0;t2<8;t2++) KVt[base+t2]=__float2bfloat16(e2m1(pv>>(4*t2))*s);
      } else { for(int t2=0;t2<8;t2++) KVt[base+t2]=__float2bfloat16(0.f); } }
    __syncthreads();
    for(int dc=0; dc<Hv; dc+=16){
      wmma::fragment<wmma::accumulator,16,16,16,float> af2; wmma::fill_fragment(af2,0.f);
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> pa_;
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> vb_;
      wmma::load_matrix_sync(pa_, Psh, NT);
      wmma::load_matrix_sync(vb_, KVt+dc, Hv);
      wmma::mma_sync(af2, pa_, vb_, af2);
      wmma::store_matrix_sync(Ssh, af2, 16, wmma::mem_row_major);
      __syncthreads();
      for(int i=lane;i<G*16;i+=32){ int r=i/16,d=i%16; acc[r*Hv+dc+d]+=Ssh[i]; }
      __syncthreads(); } }
  int hb=(seq*NQH);
  for(int r=lane;r<G;r+=32){ size_t idx=(size_t)((hb+kh*G+r)*NSPLIT+sp); pm[idx]=rm[r]; pl[idx]=rl[r]; }
  for(int i=lane;i<G*Hv;i+=32){ int r=i/Hv,d=i%Hv; size_t idx=(size_t)((hb+kh*G+r)*NSPLIT+sp); pa[idx*Hv+d]=acc[r*Hv+d]; }
}
template<int Hv>
__global__ void fa_reduce(const float* pm,const float* pl,const float* pa,__nv_bfloat16* o,int NSPLIT){
  int h=blockIdx.x, lane=threadIdx.x; const int VD=Hv/32; float m=-1e30f,l=0.f,a[VD];
  #pragma unroll
  for(int i=0;i<VD;i++) a[i]=0.f;
  for(int s=0;s<NSPLIT;s++){ size_t idx=(size_t)h*NSPLIT+s; float ms=pm[idx];
    float mn=fmaxf(m,ms),c1=__expf(m-mn),c2=__expf(ms-mn);
    #pragma unroll
    for(int i=0;i<VD;i++) a[i]=a[i]*c1+pa[idx*Hv+lane*VD+i]*c2;
    l=l*c1+pl[idx]*c2; m=mn; }
  float inv = (l>0.f)? 1.f/l : 0.f;
  #pragma unroll
  for(int i=0;i<VD;i++) o[(size_t)h*Hv+lane*VD+i]=__float2bfloat16(a[i]*inv);
}
// Static partial-stat scratch (pm/pl/pa), allocated ONCE per device and reused
// so no torch::empty happens inside a CUDA graph capture.
static torch::Tensor g_pm, g_pl, g_pa;
static long g_cap_rows = 0, g_cap_nsplit = 0;
static void* g_cap_dev = nullptr;
static void ensure_scratch(long want_rows,int NSPLIT,int Hv,const torch::TensorOptions& fopt,void* dev){
  if(g_pm.defined() && want_rows<=g_cap_rows && NSPLIT==g_cap_nsplit && dev==g_cap_dev) return;
  if(want_rows < g_cap_rows) want_rows = g_cap_rows;   // never shrink
  if(want_rows < 1) want_rows = 1;
  g_pm = torch::empty({want_rows,NSPLIT},fopt);
  g_pl = torch::empty({want_rows,NSPLIT},fopt);
  g_pa = torch::empty({want_rows,NSPLIT,Hv},fopt);
  g_cap_rows = want_rows; g_cap_nsplit = NSPLIT; g_cap_dev = dev;
}
// out: caller-provided [num_seqs*NQH, Hv] bf16 buffer; fa_reduce writes the
// final result directly into it.  min_rows pre-grows the static scratch to
// its ceiling on the FIRST call (no capture-time alloc).
void run(torch::Tensor q,torch::Tensor kcache,torch::Tensor vcache,torch::Tensor bt,
         torch::Tensor seqused,torch::Tensor out,int NKVH,int NSPLIT,double scale,
         int BS,int min_rows,long long blk_stride){
  int num_seqs=q.size(0),NQH=q.size(1); const int Hv=128; int maxblk=bt.size(1);
  int rows=num_seqs*NQH;
  long want_rows = (rows>min_rows)? (long)rows : (long)min_rows;
  auto fopt=torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
  ensure_scratch(want_rows,NSPLIT,Hv,fopt,q.device().has_index()? (void*)(intptr_t)q.device().index() : (void*)0);
  float* pm_p = g_pm.data_ptr<float>();
  float* pl_p = g_pl.data_ptr<float>();
  float* pa_p = g_pa.data_ptr<float>();
  dim3 g(NKVH,NSPLIT,num_seqs);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static bool once=false;
  if(!once){
    cudaFuncSetAttribute((const void*)&wmma_dec_m3<128,128,16,16,72>,  cudaFuncAttributePreferredSharedMemoryCarveout,100);
    cudaFuncSetAttribute((const void*)&wmma_dec_m3<128,128,16,32,72>,  cudaFuncAttributePreferredSharedMemoryCarveout,100);
    cudaFuncSetAttribute((const void*)&wmma_dec_m3<128,128,16,64,72>,  cudaFuncAttributePreferredSharedMemoryCarveout,100);
    cudaFuncSetAttribute((const void*)&wmma_dec_m3<128,128,16,128,72>, cudaFuncAttributePreferredSharedMemoryCarveout,100);
    once=true; }
  auto qp=(const __nv_bfloat16*)q.data_ptr();
  auto kp=kcache.data_ptr<unsigned char>(); auto vp=vcache.data_ptr<unsigned char>();
  auto bp=bt.data_ptr<int>(); auto sp=seqused.data_ptr<int>();
  float fscale=(float)scale;
  if(BS==128)
    wmma_dec_m3<128,128,16,128,72><<<g,32,0,stream>>>(qp,kp,vp,bp,sp,pm_p,pl_p,pa_p,NQH,NKVH,NSPLIT,maxblk,blk_stride,fscale);
  else if(BS==64)
    wmma_dec_m3<128,128,16,64,72><<<g,32,0,stream>>>(qp,kp,vp,bp,sp,pm_p,pl_p,pa_p,NQH,NKVH,NSPLIT,maxblk,blk_stride,fscale);
  else if(BS==32)
    wmma_dec_m3<128,128,16,32,72><<<g,32,0,stream>>>(qp,kp,vp,bp,sp,pm_p,pl_p,pa_p,NQH,NKVH,NSPLIT,maxblk,blk_stride,fscale);
  else
    wmma_dec_m3<128,128,16,16,72><<<g,32,0,stream>>>(qp,kp,vp,bp,sp,pm_p,pl_p,pa_p,NQH,NKVH,NSPLIT,maxblk,blk_stride,fscale);
  fa_reduce<128><<<rows,32,0,stream>>>(pm_p,pl_p,pa_p,(__nv_bfloat16*)out.data_ptr(),NSPLIT);
}
'''


def _compile():
    global _M, _OK
    if _OK is not None:
        return _OK
    try:
        from torch.utils.cpp_extension import load_inline
        _M = load_inline(
            name="wmma_decode_m3_nvfp4",
            cpp_sources="void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int,int,double,int,int,long long);",
            cuda_sources=_CUDA, functions=["run"], verbose=False,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_121,code=sm_121", "--use_fast_math"],
        )
        _OK = True
    except Exception as e:  # pragma: no cover
        import sys
        print(f"[wmma_decode_m3] compile FAILED, falling back to Triton: {e}", file=sys.stderr)
        _OK = False
    return _OK


_QLEN_CAP = 3  # decode + MTP verify (q_len = num_spec_tokens+1). This is a flash-DECODE
               # kernel: INCORRECT for prefill (multiple new tokens attending to short
               # context). Prefill chunks must go to Triton; we additionally require
               # seqused > q_len on the eager path.


def try_wmma_decode(q, k_cache, v_cache, out, seqused_k, block_table, softmax_scale,
                    num_kv_heads, head_size, block_size, sinks, softcap,
                    window_left, cu_seqlens_q, max_seqlen_q, force=False):
    """Return True if the WMMA kernel handled this call (out written); else False.

    Handles decode (q_len=1) AND MTP/speculative decode (q_len <= 3) via
    per-query-token expansion: query token t of a sequence attends to the
    causal prefix ending at its position (seq_len = seqused_k - q_len + 1 + t).
    """
    global _DBG

    def _dbg(reason):
        global _DBG
        if _DBG < 12:
            _DBG += 1
            try:
                with open("/tmp/wmma_m3_trace.log", "a") as _f:
                    _f.write(f"REJECT={reason} q={tuple(q.shape)} kvh={num_kv_heads} "
                             f"hs={head_size} bs={block_size} "
                             f"cache_last={k_cache.shape[-1] if hasattr(k_cache, 'shape') else '?'} "
                             f"mq={max_seqlen_q} win={window_left} sinks={sinks is not None} "
                             f"softcap={softcap} dt={q.dtype}\n")
            except Exception:
                pass

    if not force and os.environ.get("VLLM_M3_WMMA_DECODE", "0") != "1":
        return False  # default OFF: Triton inline wins on M3 shapes (see header)
    if (head_size != _HK or block_size not in (16, 32, 64, 128)
            or k_cache.shape[-1] != _SB or v_cache.shape[-1] != _SB
            or q.shape[1] != num_kv_heads * _G):
        _dbg("shape"); return False
    if sinks is not None or softcap not in (0.0, None) or window_left >= 0:
        _dbg("feature(sink/softcap/window)"); return False
    if q.dtype != torch.bfloat16 or cu_seqlens_q is None:
        _dbg("dtype/cu_none"); return False
    if max_seqlen_q is None or max_seqlen_q > _QLEN_CAP:   # prefill -> Triton
        _dbg("max_seqlen_q"); return False
    # Plane layout: rows contiguous within a block; block stride may include
    # the sibling plane (2x factor) -- passed explicitly to the kernel.
    NKVH = num_kv_heads
    if (k_cache.stride(3) != 1 or k_cache.stride(2) != _SB
            or k_cache.stride(1) != NKVH * _SB
            or v_cache.stride(3) != 1 or v_cache.stride(2) != _SB
            or v_cache.stride(1) != NKVH * _SB
            or k_cache.stride(0) != v_cache.stride(0)):
        _dbg("strides"); return False
    capturing = torch.cuda.is_current_stream_capturing()
    if _OK is None and capturing:
        # Never JIT-compile inside a CUDA graph capture; this graph falls
        # back to Triton (correct, just slower) until an eager call compiles.
        _dbg("compile_during_capture"); return False
    if not _compile():
        _dbg("compile"); return False
    # Prefill detection (seqused <= q_len => first prefill chunk, no prior
    # context) needs a GPU->CPU sync -- ILLEGAL during capture but also
    # unnecessary there (vLLM only captures pure decode batches).
    if not capturing:
        _lo = int(seqused_k.min().item())   # eager-only sync
        if _lo <= int(max_seqlen_q):
            _dbg("prefill(seqused<=mq)"); return False
    dev = q.device
    total_q = q.shape[0]
    cu = cu_seqlens_q.to(torch.int64)
    num_seqs = cu.shape[0] - 1
    q_lens = cu[1:] - cu[:-1]                              # [num_seqs]
    rows = torch.arange(total_q, device=dev, dtype=torch.int64)
    seq_idx = torch.bucketize(rows, cu[1:], right=True)    # token row -> seq idx
    seq_idx = seq_idx.clamp_(max=num_seqs - 1)
    t = rows - cu[seq_idx]
    su_full = seqused_k.to(torch.int64)[seq_idx]
    su = (su_full - q_lens[seq_idx] + 1 + t).to(torch.int32).contiguous()
    bt = block_table.to(torch.int32)[seq_idx].contiguous()  # per-token table
    NSPLIT = _NSPLIT_MAX   # FIXED -> static grid -> cudagraph-capturable
    blk_stride = int(k_cache.stride(0))
    min_rows = _MAX_BATCH * q.shape[1]
    qc = q.contiguous()
    if out.is_contiguous():
        out_flat = out.view(total_q * q.shape[1], _HV)
        _M.run(qc, k_cache, v_cache, bt, su, out_flat, num_kv_heads, NSPLIT,
               float(softmax_scale), int(block_size), int(min_rows), blk_stride)
    else:
        out_flat = torch.empty((total_q * q.shape[1], _HV), dtype=out.dtype, device=dev)
        _M.run(qc, k_cache, v_cache, bt, su, out_flat, num_kv_heads, NSPLIT,
               float(softmax_scale), int(block_size), int(min_rows), blk_stride)
        out.copy_(out_flat.view(total_q, q.shape[1], _HV))
    global _CALLS
    _CALLS += 1
    if _CALLS == 1:   # one-time confirmation the kernel is live
        try:
            with open("/tmp/wmma_m3_trace.log", "a") as _f:
                _f.write(f"KERNEL_ACTIVE total_q={total_q} kvh={num_kv_heads} NSPLIT={NSPLIT}\n")
        except Exception:
            pass
    return True
